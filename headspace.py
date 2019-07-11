import pandas as pd
import numpy as np
import os
import sys
#import time
#from datetime import datetime
#import pdb

from _utils_CNN import *

# from Convolutional_CNN import Conv128_3_NN

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

###################################################################################################

colour_channels = 3  # int(sys.argv[1])
script_name = sys.argv[0]
CWD = "../Data/"
images_root = "__IMAGE__"
csv_root    = "__CSV__"

###################################################################################################
trial_name = 'Conv128_GZ1_Validation_BW-64x'

if colour_channels == 3: trial_name = 'Conv128_GZ1_Validation_RGB-64x'

image_tuple   = (64, 64, colour_channels)
images_folder = "Blended_Image_Catalouge"
images_csv = "GZ1_Full_Expert.csv"
###################################################################################################
# ==================================================================================================
print_header(script_name)
# ==================================================================================================
output_folder = directory_check(CWD, 'CNN_' + trial_name, preserve=False)

# Images and Labels loading:
images_folder = os.path.join(CWD, images_root, images_folder)
images_csv    = os.path.join(CWD, csv_root, images_csv)
images_labels = pd.read_csv(images_csv)

images_IDs    = np.array(images_labels['OBJID'], dtype=str)

merger_subset = (images_labels[images_labels.Label == "M"])
merger_subset = list(merger_subset.OBJID)

images_array  = read_images_tensor(images_IDs, images_folder, image_tuple, file_extension=".tiff")
print(images_labels.shape)
print(images_array.shape)
# This is where we want our oversampling/Data Augment Methods to go?
images_labels, images_array = image_oversampler(images_labels, merger_subset[1:3],
                                                images_array, CWD, times_sampled=14)
# Need to recreate the ID list after the suffle.
images_IDs    = np.array(images_labels['OBJID'], dtype=str)

print(images_labels['OBJID'])
