from Image_Generator import Merger_Generator
import fid
from keras.models import model_from_json
import os
import numpy as np
import pandas as pd
import cv2
colour_channels = 3
image_tuple = (64, 64, colour_channels)

## Play around with CV2 resize functions. It needs to be at least 75x75 probably best if resized to 128x128

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_image_tensor_on_path(path_list, dim_tuple, extra_path_details=""):
    RGB = 1

    array_tuple = (len(path_list),) + dim_tuple
    images_array = np.zeros(array_tuple, dtype=float)
    print("Reading in files by path")
    for i in range(len(path_list)):
        if i % 1000 == 0:
            print("Read in {i} images".format(i=i))
        img_path = os.path.join(extra_path_details, path_list.iloc[i])
        image =cv2.imread(img_path, RGB)
        image_reshaped = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
    return images_array


def read_in_gan_json_model(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded = json_file.read()
    json_file.close()
    model = model_from_json(loaded)
    model.load_weights(weight_path)
    return model

json_path = 'D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\DCGan\Saved_Model\galaxy_dcgan_generator.json'
weight_path = 'D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\DCGan\Saved_Model\galaxy_dcgan_generator_weights.hdf5'
merger_maker = read_in_gan_json_model(json_path, weight_path)

noise = np.random.normal(0,1, (5 * 5, 100))

ass = merger_maker.predict(noise)

for i in range(25):
    rep = np.repeat(ass[i:i+1, :, :, :],2)
    rep = np.reshape(rep, (128, 128, 6))
    print(rep)


#ass = np.repeat(ass, 2)
#ass = np.reshape(ass, (25, 128, 128, 3))
#print(ass.shape)

#CWD = "../../Data/"
#csv_root    = "__CSV__"
#images_csv = "GZ1_Full_Expert_Paths.csv"
#id_path = os.path.join(CWD, csv_root, images_csv)
#image_ids = pd.read_csv(id_path)
#image_ids = image_ids[image_ids.EXPERT == 'M']
# real_images = create_image_tensor_on_path(image_ids.Paths, image_tuple, extra_path_details="..")

#fd = fid.FrechetInceptionDistance(merger_maker, (-1,1))
#gan_fid = fd(real_images, noise)

#print(gan_fid)