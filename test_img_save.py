from PIL import Image  ### 画像処理ライブラリPillow をインポート
y_count = 0
no_count = 0
with open("test_attr.txt","r") as f:    ### 属性ファイルを開く
     for i in range(20000):   ### 全部で202,599枚処理する
         line = f.readline()   ### 1行データ読み込み
         line = line.split() ### データを分割
        #  print(len(line))
                  ### 何枚目を処理しているかスクリーン表示
                         
         if line[7]=="1" and line[28]=="1" and line[21]=="1" and line[32]=="1" and line[40]=="1":  ### 男性で、笑顔で、鼻がとんがっている、でかい、唇が分厚い、若い
            image = Image.open("./img_align_celeba/"+line[0])
            image.save("./my_test_img/y_imgs/"+line[0])
            y_count += 1
         elif line[7]=="-1" and line[28]=="-1" and line[21]=="1" and line[32]=="-1" and line[40]=="1":  ### 男性で、笑顔で、鼻が小さい、唇が薄い、若い
            image = Image.open("./img_align_celeba/"+line[0])
            image.save("./my_test_img/no_imgs/"+line[0])
            no_count +=1
            
         if y_count > 5 and no_count > 5:
            break