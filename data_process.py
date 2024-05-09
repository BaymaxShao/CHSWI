import numpy as np
import PIL.Image as Image
import os
path = r'/home/zxk/ClothWild_RELEASE-main/data/MSCOCO/parses/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/'
path_save = r'../data/MSCOCO/parses/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations_process/'
# 返回指定路径的文件夹名称
dirs = os.listdir(path) # 循环遍历该目录下的照片
# for dir in dirs:
#     image_name = path +




    # img = Image.open(path+"/"+dir)
    # img = np.array(img)
    # img = img.astype(np.uint8)
    # for j in range(img.shape[0]):
    #     for i in range(img.shape[1]):
    #         if img[j,i] == 1 or img[j,i] == 2 or img[j,i] == 3 or img[j,i] == 4 or img[j,i] == 8 or img[j,i] == 11 or img[j,i] == 13 or img[j,i] == 14 or img[j,i] == 15 or img[j,i] == 16 or img[j,i] == 17 or img[j,i] == 18 or img[j,i] == 19:
    #             img[j,i] = 0
    # image = Image.fromarray(img,'P')
    # image.save("save1.png")
    # colormap = [0,0,0]+[0,0,0]+[0,0,0]+[0,0,0]+[255,0,0]+[0,0,255]+[0,255,0]+[0,255,255]*248
    # image.putpalette(colormap)
    # image.save('rgb_image.png')

# ceshi
# img = Image.open('1.png')
# img = np.array(img)
# img = img.astype(np.uint8)
# for j in range(img.shape[0]):
#     for i in range(img.shape[1]):
#         if img[j,i] == 1 or img[j,i] == 2 or img[j,i] == 3 or img[j,i] == 4 or img[j,i] == 8 or img[j,i] == 11 or img[j,i] == 13 or img[j,i] == 14 or img[j,i] == 15 or img[j,i] == 16 or img[j,i] == 17 or img[j,i] == 18 or img[j,i] == 19:
#             img[j,i] = 0
# image = Image.fromarray(img,'P')
# image.save("save1.png")
# colormap = [0,0,0]+[0,0,0]+[0,0,0]+[0,0,0]+[255,0,0]+[0,0,255]+[0,255,0]+[0,255,255]*248
# image.putpalette(colormap)
# image.save('rgb_image.png')


# src= Image.open("filename.png")
# mat = np.array(src)
# print(len(mat))
# mat = mat.astype(np.uint8)
# dst = Image.fromarray(mat, 'P')
# # palette = dst.getpalette()  # 获取调色板
# # print(palette)
# # bin_colormap = np.random.randint(0, 255, (256, 3))      # 可视化的颜色
# bin_colormap=[0,0,0]+[0,0,0]+[0,0,0]+[0,0,0]+[255,0,0]+[0,0,255]+[0,255,0]+[0,255,255]*248
# # bin_colormap = bin_colormap.astype(np.uint8)
# # np.save('my_array', bin_colormap)
# # print(bin_colormap.astype)
# # print(bin_colormap)
# dst.putpalette(bin_colormap)
# dst.save('new5cccccc.png')
# src.close()