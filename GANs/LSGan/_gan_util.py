import numpy as np
import os
import cv2


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
        images_array[i] = image_reshaped.astype('float32') / 175.5 - 1
    return images_array

def biniriase_lables(lables):
    length = len(lables)
    result = np.zeros((length,), dtype=int)
    for i in range(0, length):
        if lables[i] == 'M': result[i] = 1
    return result