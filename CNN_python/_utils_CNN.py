# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:24:51 2018

@author: psxmaji
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import sys
from datetime import datetime
from sklearn.utils import shuffle

matplotlib.use('Agg') 
plt.switch_backend('agg')

# =============================================================================
# *********************** F U N C T I O N S ***********************************
# =============================================================================
# (1.1) Function to save in .csv format the autoencoder results (review)


def save_features_csv (encoded_imgs, images_labels, output_folder, trial_name) : 

    if len(encoded_imgs) != len(images_labels) : 
    
        print('\n >> ERROR _utils_AE(1.1): features tensor and labels array length do not match!!\n')
        sys.exit()

    if len(encoded_imgs.shape) > 2 :
        
        num_features = np.prod(encoded_imgs.shape[1:])
        num_imgs     = len(encoded_imgs)
        
        enc_imgs_flat = np.zeros((num_imgs, num_features), dtype=float)
        
        for i in range(0, num_imgs) : 
            
            enc_imgs_flat[i] = encoded_imgs[i].flatten()
            
    else: enc_imgs_flat = encoded_imgs
    
    features = pd.DataFrame(enc_imgs_flat, index=None)
    result = images_labels.join(features)
    
    result_path = os.path.join(output_folder, trial_name + '_feats.csv')
    result.to_csv(result_path, index=None, float_format='%.4g')

    return


# (2.1) Function to read greyscale images (OK)
def read_images_flat (IDs_array, folder_path, dim_tuple, file_extension='.png', normalisation=True) :
    
    """
    This function reads the sets of 1,000 greyscale images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height).
    (Channels is supposed to be equal to '1').
    """

    images_array = np.zeros((len(IDs_array), dim_tuple[0] * dim_tuple[1]), dtype=float)           
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])

    #print('\nLoading images', end='')
    print('\nLoading images...\n')

    for i in range(0, len(IDs_array)) : 
        
        if i % 1000 == 0 : 
            
            #print('.', end='', flush=True)
            print(' -> ' + str(i) + ' images done...')
            
            folder = folders[int(i / 1000)]
        
        image_name = IDs_array[i] + file_extension
        image_path = os.path.join(folder_path, folder, image_name)
        
        images_array[i] = cv2.imread(image_path, 0).flatten()
        
    print('\n\nImages load successful!!\n')

    if normalisation == True : images_array = images_array.astype('float32') / 255.0
    
    return images_array


#(2.2) Function to read RGB images (OK)
def read_images_tensor (IDs_array, folder_path, dim_tuple, file_extension='.tiff') :
    
    """
    This function reads the sets of 1,000 (RGB) images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height, channels).
    """

    array_tuple = (len(IDs_array),) + dim_tuple
    RGB = 1
            
    images_array = np.zeros(array_tuple, dtype=float)
    
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])
    
    if dim_tuple[-1] == 1 : RGB = 0

    print('\nLoading images', end='')
    print('\nLoading images...\n')

    for i in range(0, len(IDs_array)) :

        if i % 1000 == 0 :
            
            #print('.', end='', flush=True)
            print(' -> ' + str(i) + ' images done...')
            
            folder = folders[int(i / 1000)]
        
        image_name = IDs_array[i] + file_extension
        image_path = os.path.join(folder_path, folder, image_name)
        
        image           = cv2.imread(image_path, RGB)
        image_reshaped  = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
                
    print('\n\nImages load successful!\n')
          
    return images_array


#(2.3) Function to read RGB images (OK)
def read_images_tensor_by_ID (IDs_array, IDs_array_to_read, folder_path, dim_tuple, file_ext='.png') :
    
    """
    This function reads the sets of 1,000 (RGB) images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height, channels).
    """

    array_tuple = (len(IDs_array_to_read),) + dim_tuple
    RGB = 1
    if dim_tuple[-1] == 1 : RGB = 0
            
    images_array = np.zeros(array_tuple, dtype=float)
    
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])
    
    print('\nLoading images...\n')
    
    for i in range(0, len(IDs_array_to_read)) : 
        
        if i % 1000 == 0 and i != 0 : print(' -> ' + str(i) + ' images done...')
        
        image_index = np.where(IDs_array == IDs_array_to_read[i])
        image_index = image_index[0][0]
        
        image_folder = folders[int(image_index / 1000)]
                    
        image_name = IDs_array_to_read[i] + file_ext
        image_path = os.path.join(folder_path, image_folder, image_name)
        
        image = cv2.imread(image_path, RGB)
        image_reshaped = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
                
    print('\n\nImages load successful!\n')
          
    return images_array


# (17.1)  shuffles 2 variable length items identically
def shuffle_in_unison(a, b, seed=42):
    # try and get a way of locking the state.
    np.random.RandomState(seed=seed)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


# (17.2) oversample supplied class(es) in the image tensor
def image_oversampler(id_array,id_subset, image_tensor_array, trace_output, seed=42, times_sampled=14):
    # Takes a corresponding array and tensor. Also needs a subset of the array.
    # Returns an shuffled and upsampled version of both maintaining correspondence

    id_array = id_array.to_numpy()
    temp_array = id_array
    temp_tensor = image_tensor_array
    trace_file = pd.DataFrame(columns=["ID", "times_sampled"])
    rows_added = times_sampled
    for i in range(0,len(id_subset)):
        id_index = np.where(id_array == id_subset[i])
        id_index = id_index[0][0]
        print("On index {}".format(i))
        print("Adding {times_sampled} new rows. {rows_added} total rows added".format(times_sampled=times_sampled,
                                                                                      rows_added=rows_added))
        repeats = 0
        while repeats < times_sampled:
            # does v-stack work here?
            temp_tensor = np.concatenate((temp_tensor, image_tensor_array[id_index:id_index+1]))
            temp_array = np.vstack((temp_array, id_array[i]))
            repeats += 1
        trace_file.loc[i] = [list(id_array[i])[1], times_sampled]
        rows_added += times_sampled
    shuffle_in_unison(temp_array, temp_tensor)
    trace_file.to_csv("{trace_output}/oversampling_trace.csv".format(trace_output=trace_output))

    return pd.DataFrame(temp_array, columns=["OBJID", "EXPERT"]), temp_tensor



# (5) Function to concatenate words for file names
def concat_str (str_list, file_ext='', separator='_') : 
    
    result = str_list[0]
    
    for i in range(1, len(str_list)) : 
        
        if type(str_list[i]) != str : str_list[i] = str(str_list[i])
        
        if str_list[i] != '' : result = result + separator + str_list[i]
        
    if file_ext != '' : result = result + file_ext
    
    return result


# (6) Function to create a directory for output files with several controls
def directory_check (root_path, directory_name, preserve=True) :
    
    list_dir = os.listdir(root_path)
    directory_path = os.path.join(root_path, directory_name)
    
    if preserve == False :
        
        if directory_name not in list_dir : os.makedirs(directory_path)
            
    else: 
            
        done = False
        length = len(directory_name)
        n = 3
        
        while done == False : 
                  
            if directory_name in list_dir and len(directory_name) == length : 
                
                directory_name += '(2)'
                
            elif directory_name in list_dir :  
                
                index_parenthesis = directory_name.rindex('(') + 1
                
                directory_name = directory_name[0:index_parenthesis] + str(n) + ')'
                n += 1
                
            else : 
                
                directory_path = os.path.join(root_path, directory_name)
                os.makedirs(directory_path)
                done = True
                        
    return directory_path


# (9) Function to print informative headers to the screen while running the scripts
def print_header (script_name, end=False) : 
 
    now = datetime.now()

    if end == True : 
        
        print('\n\n=============================================================')
        print('\n>> Running of ' + str(script_name) + ' FINISHED!!')
        print('\n', now.strftime("%x"), now.strftime("%X"))
        print('=============================================================\n')
        
    else :
        
        print('\n\n=============================================================')
        print('\n   >> Running ' + str(script_name) + '...')
        print('\n', now.strftime("%x"), now.strftime("%X"))
        print('=============================================================\n')
        
    return


# (10) Function to extract labels separated by '_' from a long informative string
def extract_tags (long_string_tag, position='all', extension=True) : 
    
    if extension == True : long_string_tag = long_string_tag[0:-4]
   
    tags   = []
    result = ''
    
    j = 0
    for i in range(0, len(long_string_tag)) : 
        
        if long_string_tag[i] == '_' : 
            
            tags.append(long_string_tag[j:i])
            j = i + 1
            
    tags.append(long_string_tag[j:])
    
    result = tags 
    
    if position != 'all' and position < len(tags) : result = tags[position] 
    
    return result 


# (12) Function to convert the LM expert flags to binary [0,1,-1] == [el,sp,ERROR]
def convert_labels_expert (expert_array, double_column=False) : 
        
    length = len(expert_array)
    
    if double_column == True:
        
        result = np.zeros((length, 2), dtype=int)
        
        for i in range(0, length) : 
                        
            if expert_array[i] == 'M' : result[i, 1] = 1
            
            else : result[i, 0] = 1
            
    else : 
        
        result = np.zeros((length,), dtype=int)
        
        for i in range(0, length) : 
                        
            if expert_array[i] == 'M': result[i] = 1
                           
    return result


# (13) Function to trim an image keeping it centred
def crop_image_RGB (images_array_original, desired_size) : 

    if images_array_original.shape[1] != images_array_original.shape[2] : 

        print('\n >> ERROR utils_AE(13): the images are not square scaled!!\n')
        sys.exit()
        
    elif images_array_original.shape[1] <= desired_size : 

        print('\n >> ERROR utils_AE(13): incorrect parameter values!!\n')
        sys.exit()
            
    sqr_dim = images_array_original.shape[1]
    
    n_crop = sqr_dim - desired_size
    
    chunk = int(n_crop / 2.0)
    
    pixels_to_crop = [i for i in range(0, chunk)] + [i for i in range(chunk + desired_size, sqr_dim)]

    images_array_original = np.delete(images_array_original, pixels_to_crop, axis=1)
    images_array_original = np.delete(images_array_original, pixels_to_crop, axis=2)  

    return images_array_original     


# (14.1) Function to split whole features and labels arrays into training and validation sets
def data_split_AE_test (images_array, labels_pd, test_partition, n_splits=5) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    return (images_train, labels_train, images_test, labels_test)


# (14.2) Function to split whole features and labels arrays into training and validation sets
def data_split_CNN_test_expert (images_array, labels_pd, test_partition, n_splits=5) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    labels_train_binary = np.array(labels_train['EXPERT'], dtype=str)
    labels_train_binary = convert_labels_expert(labels_train_binary, double_column=True)
    labels_test_binary = np.array(labels_test['EXPERT'], dtype=str)
    labels_test_binary = convert_labels_expert(labels_test_binary, double_column=True) 
    
    return (images_train, labels_train, labels_train_binary,
            images_test, labels_test, labels_test_binary)
    

# (14.3) Function to split whole features and labels arrays into training and validation sets
def data_split_CNN_test_amateur (images_array, labels_pd, test_partition, n_splits=5) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
        
    labels_train_binary = np.transpose(np.vstack((np.array(labels_train['EL_RAW'], dtype=int),
                                                  np.array(labels_train['CS_RAW'], dtype=int))))
    
    labels_test_binary  = np.transpose(np.vstack((np.array(labels_test['EL_RAW'], dtype=int),
                                                  np.array(labels_test['CS_RAW'], dtype=int))))
    
    return (images_train, labels_train, labels_train_binary,
            images_test, labels_test, labels_test_binary)


# (15.1) Function to split training features and labels arrays into training/validation sets
def data_split_AE_training (images_array_train, train_val_ratio=0.7) : 
    
    train_partition = int(train_val_ratio * len(images_array_train))
    #np.random.shuffle(images_array_train)
    
    images_train      = images_array_train[0 : train_partition]
    images_validation = images_array_train[train_partition :]
    
    return (images_train, images_validation)


# (15.2) Function to split training features and labels arrays into training/validation sets
def data_split_CNN_training (images_array_train, labels_array_train, train_val_ratio=0.7) :
    
    if len(images_array_train) != len(labels_array_train) : 
    
        print('\n >> ERROR utils_CNN(15.2): images and labels arrays length do not match!!\n')
        sys.exit()
    
    train_partition = int(train_val_ratio * len(images_array_train))
    #np.random.shuffle(images_array_train)
    
    images_train = images_array_train[0 : train_partition]
    images_val   = images_array_train[train_partition :]
    
    labels_train = labels_array_train[0 : train_partition] 
    labels_val   = labels_array_train[train_partition :] 
    
    return images_train, labels_train, images_val, labels_val


# (16) Function to convert seconds into dd:hh:mm:ss
def convert_seconds (seconds) : 
    
    time = float(seconds)
    
    days = int(time // (24 * 3600))
    time = time % (24 * 3600)
    
    hours = str(int(time // 3600))      
    time %= 3600
    
    minutes = str(int(time // 60))
    time %= 60
    
    seconds = str(round(time, 2))
    
    if days != 0 : result = '(' + str(days) + 'd' + hours + 'h' + minutes + 'm' + seconds + 's)'
    
    else : result = '(' + hours + 'h' + minutes + 'm' + seconds + 's)' 
    
    return result


# (3.2) Function to print to a .txt and .csv files all runnning information and classifier performance
def classification_performance (features_train, predictions_array, labels_test, exec_time, output_path, output_name, partition) : 
     
    predictions_array = round_to_single_column(predictions_array)
    labels_test       = round_to_single_column(labels_test)
    
    CM = Confusion_Matrix(predictions_array, labels_test)
    print(CM)
    accuracy = Acc(CM)
    precision = Precision(CM)
    recall = Recall(CM)
    f1_score = F1_score(CM)
    g_mean = Geometric_Mean(CM)
    exec_time = round(exec_time, 2)
    exec_time_str = convert_seconds(exec_time)
    
    list_dir = os.listdir(output_path)  
    csv_metrics_name = output_name + '_classification.csv'
    csv_metrics_path = os.path.join(output_path, csv_metrics_name)
    
    metrics_tags = ['Test_part', 'N_train', 'N_test', 'Train_time(s)', 'Train_time',
                    'Accuracy', 'Precision', 'Recall', 'F1_score', 'Geometric_Mean']
    new_raw_data = [partition, len(features_train), len(labels_test), exec_time, exec_time_str,
                    accuracy, precision, recall, f1_score, g_mean]

    new_row = pd.DataFrame(data=[new_raw_data], columns=metrics_tags)
   
    if csv_metrics_name not in list_dir : 
        
        csv_metrics = new_row
        
    else : 
        
        csv_metrics = pd.read_csv(csv_metrics_path)    
        csv_metrics = csv_metrics.append(new_row, ignore_index=True)
        
    csv_metrics.to_csv(csv_metrics_path, index=False, float_format='%.4f')
            
    if partition == 5 : global_statistics(output_path, output_name)

    return 



# (3.3) Function to print to a .txt and .csv files all runnning information and classifier performance
def save_predictions (data_test, predictions_array, partition, classifier_tag, output_path, output_name) : 
    
    csv_name = output_name + '_test_results_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)
    
    classifier_tag = classifier_tag 
    
    data_test[classifier_tag] = np.array(predictions_array, dtype=int)
    
    data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'EXPERT', classifier_tag]]
        
    data_test.to_csv(csv_path, index=False)


    return



# (3.4) Function to print to a .txt and .csv files all runnning information and classifier performance
def save_predictions_CNN (data_test, predictions_array, partition, output_path, output_name) : 
    
    csv_name = output_name + '_test_results_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)

    data_test['CNN_NM'] = np.array(predictions_array[:, 0])
    data_test['CNN_M'] = np.array(predictions_array[:, 1])

    #print(data_test)

    #data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'EXPERT', 'CNN_EL', 'CNN_SP']]
    #data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'CNN_EL', 'CNN_SP']]
    #data_test = data_test[["'OBJID', 'EXPERT' , 'CNN_EL', 'CNN_SP'"]]
       
    data_test.to_csv(csv_path, index=False)


    return



# (3.4) Function to compute the mean and std of the csv file originated in (3)
def global_statistics (output_file_dir, output_filename) : 
    
    #csv_name = os.path.basename(output_file_dir) + '.csv'
    
    csv_name = output_filename + '_classification.csv'

    csv_path = os.path.join(output_file_dir, csv_name)
    csv_file = pd.read_csv(csv_path)
    
    acc = csv_file['Accuracy']
    pre = csv_file['Precision']
    rec = csv_file['Recall']
    f1s = csv_file['F1_score']
    gom = csv_file['Geometric_Mean']
    time = csv_file['Train_time(s)']
    
    meta_metrics_tags = ['Acc_mean', 'Acc_std', 'Prec_mean', 'Prec_std', 'Rec_mean', 'Rec_std',
                         'F1_mean', 'F1_std', 'Geometric_Mean_Mean', 'Geometric_Mean_std','Train_time(s)',
                         'Train_time(s)_std']
    meta_metrics_data = [acc.mean(), acc.std(), pre.mean(), pre.std(), rec.mean(), rec.std(),
                         f1s.mean(), f1s.std(),gom.mean(), gom.std(), time.mean(), time.std()]
    
    ### IMPORTANT!!: This code implements (n - 1)-std function
    
    csv_result = pd.DataFrame(data=[meta_metrics_data], columns=meta_metrics_tags)
    
    csv_result_path = os.path.join(output_file_dir, '_' + output_filename + '_classification_summary.csv')
    csv_result.to_csv(csv_result_path, index=False, float_format='%.4f')
    
    return



# (13) Function to convert a two-column binary vector to a single column binary vector
def round_to_single_column (two_columns_array) : 
    
    length = len(two_columns_array)
    result = np.zeros((length,), dtype=int)
    
    for i in range(0, length) : 
        
        if two_columns_array[i, 1] > two_columns_array[i, 0]   : result[i] = 1
        
    return result



# (2.1) Function to evaluate predictions' Confusion Matrix 
def Confusion_Matrix (predictions_array, labels_array) : 
    
    if len(predictions_array) != len(labels_array) : 
        
        print('\n >> ERROR utils_experiments(2.1): Label arrays length do not match!\n')
        sys.exit()  
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(0, len(predictions_array)) : 
        
        if predictions_array[i] == 1 and labels_array[i] == 1 : TP += 1
        if predictions_array[i] == 1 and labels_array[i] == 0 : FP += 1
        if predictions_array[i] == 0 and labels_array[i] == 0 : TN += 1
        if predictions_array[i] == 0 and labels_array[i] == 1 : FN += 1
        
    
    return (TP, FN, FP, TN)



# (2.2) Function to evaluate predictions' Accuracy from Confusion Matrix
def Acc (confusion_matrix) : 
    
    trues = confusion_matrix[0] + confusion_matrix[3]
    total = trues + confusion_matrix[1] + confusion_matrix[2]
    
    result = (trues * 1.0) / total
    
    return round(result, 4)



# (2.3) Function to evaluate predictions' Precison from Confusion Matrix
def Precision (confusion_matrix) : 
    
    TP = confusion_matrix[0] * 1.0
    FP = confusion_matrix[2] * 1.0
    
    all_Positive = TP + FP
    
    if all_Positive != 0.0 : precision = TP / all_Positive
    
    else : precision = 0.0
    
    return round(precision, 4)



# (2.4) Function to evaluate predictions' Recall from Confusion Matrix
def Recall (confusion_matrix) : 
    
    TP = confusion_matrix[0] * 1.0
    FN = confusion_matrix[1] * 1.0
    
    denom = TP + FN
    
    if denom != 0.0 : recall = TP / denom
    
    else : recall = 0.0
    
    return round(recall, 4)



# (2.5) Function to evaluate predictions' F1 Score from Confusion Matrix 
def F1_score (confusion_matrix) : 
    
    precision = Precision(confusion_matrix)
    recall    = Recall(confusion_matrix)
    
    denom = precision + recall
    
    if denom != 0.0 : F1_score  = (2 * precision * recall) / denom
    
    else : F1_score = 0.0
    
    return round(F1_score, 4)


def Geometric_Mean(confusion_matrix):
    TP = confusion_matrix[0] * 1.0
    FN = confusion_matrix[1] * 1.0
    FP = confusion_matrix[2] * 1.0
    TN = confusion_matrix[3] * 1.0

    g_mean = np.sqrt( (TP/(TP+FN)) * (TN/(FP*TN)))

    return round(g_mean, 4)

