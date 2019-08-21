import pandas as pd
import numpy as np
import os
import cv2
# from keras.datasets import mnist
from keras.models import model_from_json
from AdaIN import AdaInstanceNormalization
from PIL import Image
import sys
#import time
#from datetime import datetime
#import pdb
#from _NOC_Utils import create_image_tensor_on_path

from _utils_CNN import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
"""
###################################################################################################

colour_channels = 3 #int(sys.argv[1])  What should be registering here. My IDE says its out of index range. Used a magic 3.
script_name = sys.argv[0]
CWD = "../Data/"
images_root = "__IMAGES__"
csv_root    = "__CSV__"

###################################################################################################
trial_name = 'Conv128_GZ1_Validation_BW-64x'

if colour_channels == 3:
    trial_name = 'Conv128_GZ1_Validation_RGB-64x_2'

image_tuple   = (64, 64, colour_channels)
images_folder = "Blended_Image_Catalouge_tiff"
images_csv = "GZ1_Full_Expert.csv"
###################################################################################################
# ==================================================================================================
print_header(script_name)
# ==================================================================================================
output_folder = directory_check(CWD, 'CNN_' + trial_name, preserve=False)

# comment out if not testing
images_folder = "Test"

# Images and Labels loading:
images_folder = os.path.join(CWD, images_root, images_folder)
images_csv    = os.path.join(CWD, csv_root, images_csv)
images_labels = pd.read_csv(images_csv)
images_labels = images_labels[["OBJID","EXPERT"]]

# comment out if not testing
images_labels = images_labels[:1000]

images_IDs    = np.array(images_labels['OBJID'], dtype=str)
images_array  = read_images_tensor(images_IDs, images_folder, image_tuple)

merger_subset = (images_labels[images_labels.EXPERT == "M"])
merger_subset = list(merger_subset.OBJID)
print(len(merger_subset))

# This is where we want our Oversampling / Data Augment Methods
images_labels2, images_array2 = image_oversampler(images_labels, merger_subset,
                                                  images_array, CWD, times_sampled=3)
print(images_labels2.head(5))

# Need to recreate the ID list after the suffle.
# images_IDs = np.array(images_labels['OBJID'], dtype=str)


############
Thoughts on shufeling 
############
def shuffle_in_unison(a, b, seed=42):
    # try and get a way of locking the state.
    # Remember this only takes NP arrays
    np.random.RandomState(seed=seed)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


#np1 = np.array([[1,2,3],[4,5,6],[7, 8, 9]])
#np2 = np.array([[1,2,3],[4,5,6],[7,8,9]])

#shuffle_in_unison(np1,np2, seed=1000)

#print(np1)
#print(np2)


df2 = df.copy()
print(len(df))
df = df.OBJID.to_numpy(dtype="str")
df2 = df2.OBJID.to_numpy(dtype="str")

shuffle_in_unison(df,df2)

output = sum(df == df2)
print(output)



#This section was where I made the path tensor maker

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
        if image is None:
            print(img_path)
        image_reshaped = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
    return images_array

d = "D:\Documents\Comp Sci Masters\Project_Data\Data\__CSV__\GZ1_Full_Expert__Augment_Paths.csv"
paths = pd.read_csv(d)
paths = paths[:100]
image_tuple   = (64, 64, 3)
imgs = create_image_tensor_on_path(paths["Paths"], image_tuple)

d = "D:\Documents\Comp Sci Masters\Project_Data\Data\__CSV__\GZ1_Full_Expert__Paths.csv"
paths = pd.read_csv(d)

extras = undersample(paths, paths.EXPERT, list(paths), final_dataset_size=10000, minority_class_size=2733,
                     seed=42)
print(extras)


def augmentation_oversample(augmented_data, test_ids, samples_needed, num_of_augments=20):
    #This is important for some reason...
    added_data = 0
    final_data = pd.DataFrame(columns=['OBJID', 'Source_Lables', 'EXPERT', 'Paths'])
    while added_data < samples_needed:
        for test_id in test_ids:
            check_id = str(test_id) + "_0"
            if check_id in list(augmented_data.OBJID):
                print("I was found in the list")
                #augments = np.random.randint(0, num_of_augments)
                augments= [0, 1]
                for augs in range(0, augments+1):
                    used_id = str(test_id) + "_{}".format(augs)
                    data = augmented_data[augmented_data.OBJID == used_id]
                    final_data = final_data.append(data)
                    added_data += 1
                if added_data > samples_needed:
                    break
    return final_data

test_row = {"OBJID": "00005",
           "Source_Lables": "M",
           "EXPERT": "M",
           "Paths": "Path"}
df = pd.DataFrame({"OBJID": ["00001_0", "00002_0", "00002_1", "00003_0", "00004_0", "00005_0", "00005_1"],
           "Source_Lables": ["M", "M", "M", "M", "M", "M", "M"],
           "EXPERT": ["M", "M", "M", "M", "M", "M", "M"],
           "Paths": ["Path1", "Path2_0", "Path2_1", "Path3", "Path4", "Path5_0", "Path5_1"]},
                  columns=['OBJID', 'Source_Lables', 'EXPERT', 'Paths'])

test_ids = ['00005', "00002"]

op = augmentation_oversample(df, test_ids, num_of_augments=1)
print(op)
"""

latent_size = 512
im_size = 64
def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

#Noise Sample
def noiseImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1])


def loadModel(name, num):  # Load a Model

    file = open("GANs/StyleGAN/Models/" + name + ".json", 'r')
    json = file.read()
    file.close()

    mod = model_from_json(json, custom_objects={'AdaInstanceNormalization': AdaInstanceNormalization})
    mod.load_weights("GANs/StyleGAN/Models/" + name + "_" + str(num) + ".h5")

    return mod

sgan_gen = loadModel("gen", 0)
sgan_gen.summary()
for i in range(1, 25):
    n = noise(32)
    n2 = noiseImage(32)
    im2 = sgan_gen.predict([n, n2, np.ones([32, 1])])

    r12 = np.concatenate(im2[i-1:i], axis=1)
    r22 = np.concatenate(im2[8:16], axis=1)
    r32 = np.concatenate(im2[16:24], axis=1)
    # r43 = np.concatenate(im3[:8], axis=1)

    # c1 = np.concatenate([r12, r22, r32], axis=0)

    x = Image.fromarray(np.uint8(r12 * 255))

    x.save(f"test{i}.jpg")