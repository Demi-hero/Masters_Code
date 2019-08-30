import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
np.random.seed(42)
xmin = -10
xmax = 10
ymin = -10
ymax = 11
while 0:
    rgb = 1
    dim_tuple = (9,64, 64, 3)
    im_path1 = "D:\Documents\Comp Sci Masters\Project_Data\Data\Results\GAN_Assesments\StyleGan_Full\gen_0_vs_lbl_rid_587739097529647193_NM.png"
    im_path2 = "D:\Documents\Comp Sci Masters\Project_Data\Data\Results\GAN_Assesments\StyleGan_Full\gen_1_vs_lbl_rid_587742774558195782_NM.png"
    im_path3 = "D:\Documents\Comp Sci Masters\Project_Data\Data\Results\GAN_Assesments\StyleGan_Full\gen_2_vs_lbl_rid_587727178988847263_M.png"
    pths = [im_path1, im_path2, im_path3]

    img_array = np.zeros(dim_tuple, dtype=float)
    cntr = 0
    for pic in pths:
        img = cv2.imread(pic, 1)
        for i in range(3):
            part = img[:, 64*i:64*(i+1), :]
            img_array[cntr] = part.astype("float32")
            cntr += 1

    r12 = np.concatenate(img_array[:3], axis=1)
    r22 = np.concatenate(img_array[3:6], axis=1)
    r32 = np.concatenate(img_array[6:9], axis=1)

    fin = np.concatenate([r12,r22,r32])

    x = Image.fromarray(np.uint8(fin))
    x.save("Style_Full_Comp.jpg")



    def read_in_tiled_img(path, dim_tuple, rgb =1 ):
        image = cv2.imread(path, rgb)
        img_array = np.zeros(dim_tuple, dtype=float)
        cntr = 0
        for j in range(3):
            img = image[:, 64*j:64*(j+1), :]
            img_array[cntr] = img.astype("float32") / 255.0
            cntr += 1
        return img_array

    #image_reshaped = image.reshape(dim_tuple)
    #images_array = image_reshaped.astype('float32') / 255.0
    #for i in range(25):
        #cv2.imshow('image',images_array[:, :, i, :])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    #print(images_array.shape)

array = np.linspace(-10, 10, num=100)
a = []
b = []
for i in array:
    y = -(i+1)**2+10
    a.append(i)
    b.append(y)
plt.plot(a,b)
c = []
d = []

for i in array:
    y = -(2.5*i-10)**2+10
    c.append(i)
    d.append(y)
plt.plot(c,d)
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])
plt.title= 'Test parabola'
plt.show()