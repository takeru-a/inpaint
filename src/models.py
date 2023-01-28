import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else: 
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + landmark(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=4, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        self.tv_loss = TVLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )



    def process(self, images, landmarks, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs

        outputs = self(images, landmarks, masks)

        
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(torch.cat((dis_input_real, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        tv_loss = self.tv_loss(outputs*masks+images*(1-masks))
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, landmarks, masks):
        # print(landmarks.shape, masks.shape)
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, landmarks), dim=1)
        conv1 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0)
        conv1 = nn.DataParallel(conv1,device_ids=[0])
        conv2 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        conv2 = nn.DataParallel(conv2,device_ids=[0])
        conv3 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        conv3 = nn.DataParallel(conv3,device_ids=[0])
        # print(inputs.shape)
        scaled_masks_quarter = F.interpolate(conv1(masks), size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                     mode='bilinear', align_corners=True)
        scaled_masks_half = F.interpolate(conv2(masks), size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                     mode='bilinear', align_corners=True)
        # print(scaled_masks_half.shape,scaled_masks_quarter.shape)
        masks = conv3(masks)
        outputs = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter)
        # in: [rgb(3) + landmark(1)]
        return outputs

    def backward(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

    def backward_joint(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


from .networks import MobileNetV2

def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx,other=torch.ones(absx.shape).cuda())
    r = 0.5 *((absx-1)*minx + absx)
    return r

def loss_landmark_abs(y_true, y_pred):
    loss = torch.mean(abs_smooth(y_pred - y_true))
    return loss

def loss_landmark(landmark_true, landmark_pred, points_num=68):
    landmark_loss = torch.norm((landmark_true-landmark_pred).reshape(-1,points_num*2),2,dim=1,keepdim=True)
    return torch.mean(landmark_loss)

#todo 損失関数変更
# 口角の距離の差分 L2
def loss_mouth_corner_distance(mcd_true, mcd_pred):
    # print(mcd_pred[0][54],mcd_true[0][54][0])
    dis = torch.pow(torch.dist(mcd_true[0][54], mcd_true[0][56]) - torch.dist(mcd_pred[0][54].to(torch.float64),mcd_pred[0][56].to(torch.float64)),2)
    dis += torch.pow(torch.dist(mcd_true[0][48], mcd_true[0][58]) - torch.dist(mcd_pred[0][48].to(torch.float64),mcd_pred[0][58].to(torch.float64)),2)
    return dis/2

# 唇の距離の差分 L2
def loss_lip_distance(ld_true, ld_pred):
    dis = torch.pow(torch.dist(ld_true[0][51], ld_true[0][57].to(torch.float64)) - torch.dist(ld_pred[0][51],ld_pred[0][57].to(torch.float64)),2)
    dis += torch.pow(torch.dist(ld_true[0][62], ld_true[0][66].to(torch.float64)) - torch.dist(ld_pred[0][62],ld_pred[0][66].to(torch.float64)),2)
    return dis/2

# labelの情報: 表情の差分　 Softmax-with-Loss one-hot
# def loss_state(x, label_class):
#     # softmax
#     y = torch.exp(x) / torch.sum(torch.exp(x), axis=1, keepdims=True)
#     return -torch.sum(label_class * torch.log(y))

class LandmarkDetectorModel(nn.Module):
    def __init__(self, config):
        super(LandmarkDetectorModel, self).__init__()
        self.mbnet = MobileNetV2(points_num=config.LANDMARK_POINTS)
        self.name = 'landmark_detector'
        self.iteration = 0
        self.config = config

        self.landmark_weights_path = os.path.join(config.PATH, self.name + '.pth')
        print(self.landmark_weights_path)
        if len(config.GPU) > 1:
            self.mbnet = nn.DataParallel(self.mbnet, config.GPU)

        self.optimizer = optim.Adam(
            params=self.mbnet.parameters(),
            lr=self.config.LR,
            weight_decay=0.000001
        )


    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'detector': self.mbnet.state_dict()
        }, self.landmark_weights_path)

    def load(self):
        if os.path.exists(self.landmark_weights_path):
            print('Loading landmark detector...')

            if torch.cuda.is_available():
                data = torch.load(self.landmark_weights_path)
            else:
                data = torch.load(self.landmark_weights_path, map_location=lambda storage, loc: storage)

            self.mbnet.load_state_dict(data['detector'])
            self.iteration = data['iteration']
            print('Loading landmark detector complete!')

    def forward(self, images, masks, label):
        images_masked = images* (1 - masks).float() + masks
        #ラベル数が2で、出力される特徴量の次元数が16
        # (0: not smiling, 1:smiling)
        embedding = nn.Embedding(2, 16)
        embedding = nn.DataParallel(embedding,device_ids=[0])
        output_label = embedding(label)
        #　サイズ調整
        output_label = output_label.view(-1, 1, 1, 16)
        # print(output_label.shape)
        # self.config.INPUT_SIZE = 256
        output_label = output_label.repeat(1, 1, 256, 16)
        # 破損顔画像とEmbeddingした特徴量を連結
        # print(images_masked.shape, output_label.shape)
        output = torch.cat((images_masked, output_label), dim=1)
        landmark_gen, pre_label = self.mbnet(output)
        landmark_gen *= self.config.INPUT_SIZE

        return landmark_gen, pre_label
    
    def process(self, images, masks, landmark_gt, label):
        self.iteration += 1
        self.optimizer.zero_grad()

        images_masked = images*(1-masks)+masks
        landmark_gen , pre_label = self(images_masked, masks, label)
        landmark_gen = landmark_gen.reshape((-1, self.config.LANDMARK_POINTS, 2))
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)
        
        # new
       #口角の距離の差分 2点        
        loss += loss_mouth_corner_distance(landmark_gen, landmark_gt)
       #唇の距離の差分 　2点
        loss += loss_lip_distance(landmark_gen, landmark_gt)
        # State Loss: スマイリングのラベル情報が正しいかどうかを判断する損失関数
        # print(pre_label.to(torch.long),label.view(-1, 1).to(torch.long))
        logs = [("loss", loss.item())]
        # print(pre_label.to(torch.long).dtype,label.view(-1, 1).to(torch.long).dtype)
        loss += nn.BCEWithLogitsLoss(reduction='mean')(pre_label.to(torch.float64), label.view(-1, 1).to(torch.float64))
        logs = [("loss", loss.item())]
        return landmark_gen, loss, logs, pre_label
 
    def process_aug(self, images, masks, landmark_gt, label):
        self.optimizer.zero_grad()
        images_masked = images*(1-masks)+masks
        landmark_gen, pre_label = self(images_masked, masks, label)
        # ? x 68 x 2
        landmark_gen = landmark_gen.reshape(-1,self.config.LANDMARK_POINTS,2)
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)
       # new
       #口角の距離の差分 2点        
        loss += loss_mouth_corner_distance(landmark_gen, landmark_gt)
       #唇の距離の差分 　2点
        loss += loss_lip_distance(landmark_gen, landmark_gt)
        # State Loss: スマイリングのラベル情報が正しいかどうかを判断する損失関数
        loss += nn.BCEWithLogitsLoss(pre_label, label)
        logs = [("loss_aug", loss.item())]

        return landmark_gen, loss, logs, pre_label



    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
