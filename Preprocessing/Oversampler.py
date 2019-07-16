import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import cv2
import os
seed = 42
np.random.RandomState(seed=seed)
colour_channels = 3
image_ids = pd.read_csv("../../Data/__CSV__/GZ1_Full_Expert_Paths.csv")
image_ids = image_ids[:2000]

ind_start = len(image_ids[image_ids.EXPERT == "NM"])

def oversampler(image_ids,image_lables,seed=42):
    ros = RandomOverSampler(random_state=seed)
    resample, relabel = ros.fit_resample(image_ids,image_lables)
    resample = pd.DataFrame(resample, columns=["OBJID", "Source_Lable", "EXPERT", "Path"])
    return resample[len(image_ids):]

image_tuple = (64, 64, colour_channels)
def create_image_tensor_on_path(path_list,dim_tuple):
    RGB = 1

    array_tuple = (len(path_list),) + dim_tuple
    images_array = np.zeros(array_tuple, dtype=float)
    print("Reading in files by path")
    for i in range(len(path_list)):
        if i % 1000 == 0:
            print("Read in {i} images".format(i=i))
        img_path = os.path.join(path_list.iloc[i])
        image =cv2.imread(img_path, RGB)
        image_reshaped = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
    return images_array


oversamples = create_image_tensor_on_path(duplicates.Path, image_tuple)

#np.random.shuffle(resample)


#resample.to_csv("../../Data/__CSV__/GZ1_Full_Expert_Paths_oversampled.csv")

