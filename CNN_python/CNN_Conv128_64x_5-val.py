# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:21:14 2018

@author: psxmaji
"""

import pandas as pd
import numpy as np
import os
import sys
#import time
#from datetime import datetime
#import pdb

from _utils_CNN import *

from Convolutional_CNN import Conv128_3_NN

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

###################################################################################################

colour_channels = int(sys.argv[1])
script_name = sys.argv[0] 
CWD = "../../Data/"
images_root = "__IMAGES__"
csv_root    = "__CSV__"

###################################################################################################
trial_name = 'Conv128_GZ1_Validation_BW-64x'

if colour_channels == 3 : trial_name = 'Conv128_GZ1_Validation_RGB-64x'

image_tuple   = (64, 64, colour_channels)
images_folder = "GZ1_Validation_Full_64x_png" 
images_csv = "GZ1_Validation_Full.csv"
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
images_array  = read_images_tensor(images_IDs, images_folder, image_tuple) 

merger_subset = (images_labels[images_labels.Label == "M"])
merger_subset = list(merger_subset.OBJID)

# This is where we want our Oversampling / Data Augment Methods
images_labels, images_array = image_oversampler(images_labels, merger_subset,
                                                images_array, CWD, times_sampled=14)
# Need to recreate the ID list after the suffle.
images_IDs    = np.array(images_labels['OBJID'], dtype=str)

partitions = [i for i in range(1, 6)]

for test_partition in partitions: 
    
    # Training, Validation and Test partitions:
    (images_train, labels_train, labels_train_bin, images_test, labels_test,
     labels_test_bin) = data_split_CNN_test(images_array, images_labels, test_partition, n_splits=5)
    
    (CNN_train_images, CNN_train_labels, CNN_val_images,
     CNN_val_labels) = data_split_CNN_training(images_train, labels_train_bin, train_val_ratio=0.7)
    
    print('\n\n  >> ' + trial_name)
    print('\n  -> Partition ' + str(test_partition) + '\n')
    
    # Autoencoder setting and training:
    CNN = Conv128_3_NN(image_tuple)
    CNN.train(CNN_train_images, CNN_train_labels, CNN_val_images, CNN_val_labels, epochs=100)
    
    # Output log file:
    train_time = CNN.trial_log(output_folder, trial_name, test_partition=test_partition)
    
    # Test predictions and classification results: 
    predictions = CNN.model.predict(images_test)
    
    classification_performance(CNN_train_images, predictions, labels_test_bin, train_time, output_folder,
                               trial_name, test_partition)
    
    save_predictions_CNN(labels_test, predictions, test_partition, output_folder, trial_name)

#==================================================================================================
print_header(script_name, end=True)
#==================================================================================================
