import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
import cv2
import os
import sys
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from _NOC_Utils import *
from Convolutional_CNN import Conv128_3_NN

colour_channels = 3
image_tuple = (64, 64, colour_channels)
np.random.seed(56)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###################################################################################################

colour_channels = 3 #int(sys.argv[1])  What should be registering here. My IDE says its out of index range. Used a magic 3.
script_name = sys.argv[0]
CWD = "../../Data/"
images_root = "__IMAGES__"
csv_root    = "__CSV__"

###################################################################################################
trial_name = 'Conv128_GZ1_Validation_BW-64x'

if colour_channels == 3:
    trial_name = 'Conv128_GZ1_Validation_RGB-64x_Trial'

image_tuple   = (64, 64, colour_channels)
images_folder = "Blended_Image_Catalouge_tiff"
images_csv = "GZ1_Full_Expert_Paths.csv"
###################################################################################################
# ==================================================================================================
print_header(script_name)
# ==================================================================================================
output_folder = directory_check(CWD, 'CNN_' + trial_name, preserve=False)

# read in file
id_path = os.path.join(CWD, csv_root, images_csv)
image_ids = pd.read_csv(id_path)

# Uncomment for when running tests
image_ids = image_ids[:1000]

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
for test_partition in range(1, 6):

    # create the train/test/val trio
    train, test, val = cross_fold_train_test_split(image_ids, test_partition)

    # Uncomment to perform undersampling
    # train = undersample(train, train.EXPERT, list(train), reduction=0.7)
    # train = shuffle(train)

    # Uncomment to perform oversampling
    # extras = oversample(train, train.EXPERT, list(train))
    # train = pd.concat([train, extras], axis=0)
    # train = shuffle(train)

    # Binary Lable Creation can I make this bit smaller?
    train_binary_labels = np.array(train['EXPERT'], dtype=str)
    val_binary_labels = np.array(val["EXPERT"], dtype=str)
    test_binary_lables = np.array(test["EXPERT"], dtype=str)
    train_binary_labels = convert_labels_expert(train_binary_labels, double_column=True)
    val_binary_labels = convert_labels_expert(val_binary_labels, double_column=True)
    test_binary_lables = convert_labels_expert(test_binary_lables, double_column=True)

    # the tensor read ins.
    train_images = create_image_tensor_on_path(train.Paths, image_tuple, extra_path_details="..")
    val_images = create_image_tensor_on_path(val.Paths, image_tuple, extra_path_details="..")
    test_images = create_image_tensor_on_path(test.Paths, image_tuple, extra_path_details="..")

    # the machine learning part
    print('\n\n  >> ' + trial_name)
    print('\n  -> Partition ' + str(test_partition) + '\n')

    # Autoencoder setting and training:
    CNN = Conv128_3_NN(image_tuple)
    # Look in to Epoc Value. Once
    CNN.train(train_images, train_binary_labels, val_images, val_binary_labels, epochs=1)

    # Output log file:
    train_time = CNN.trial_log(output_folder, trial_name, test_partition=test_partition)

    # Test predictions and classification results:
    predictions = CNN.model.predict(test_images)

    classification_performance(train_images, predictions, test_binary_lables, train_time, output_folder,
                               trial_name, test_partition)

    save_predictions_CNN(test, predictions, test_partition, output_folder, trial_name)



    rpreds = round_to_single_column(predictions)
    rtbl = round_to_single_column(test_binary_lables)

    fpr, tpr, thresholds = roc_curve(rtbl, rpreds)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (test_partition, roc_auc))

plot_mean_roc_curve(tprs, mean_fpr, aucs, output_folder, trial_name)


