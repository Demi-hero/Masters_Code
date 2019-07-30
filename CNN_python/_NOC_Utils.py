import numpy as np
import pandas as pd
import cv2
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from datetime import datetime


def cross_fold_train_test_split(image_ids, partition, n_splits=5, test_size=0.2):

    data_partitions = [len(x) for x in np.array_split(image_ids, n_splits)]
    start_point = sum(data_partitions[0: (partition - 1)])
    end_point = start_point + data_partitions[partition - 1]

    training_set = pd.concat([image_ids.iloc[0:start_point], image_ids.iloc[end_point:]],axis=0)
    validation = image_ids.iloc[start_point : end_point]

    training, testing = train_test_split(training_set, test_size=test_size)
    return training, testing, validation


def oversample(image_ids, image_lables, column_names, seed=42):
    ros = RandomOverSampler(random_state=seed)
    resample, relabel = ros.fit_resample(image_ids, image_lables)
    resample = pd.DataFrame(resample, columns=column_names)
    return resample[len(image_ids):]


def undersample(image_ids, image_lables, column_names, reduction=0.5, seed=42):
    minor = len(image_ids[image_ids.EXPERT == 'M'])
    final = (len(image_ids)-minor) * reduction
    rus = RandomUnderSampler(sampling_strategy=(minor/final), random_state=seed)
    resample, relable = rus.fit_resample(image_ids, image_lables)
    resample = pd.DataFrame(resample, columns=column_names)
    return resample


def convert_labels_expert(expert_array, double_column=False):
    length = len(expert_array)
    if double_column:
        result = np.zeros((length, 2), dtype=int)
        for i in range(0, length):
            if expert_array[i] == 'M':
                result[i, 1] = 1
            else:
                result[i, 0] = 1
    else:
        result = np.zeros((length,), dtype=int)
        for i in range(0, length):
            if expert_array[i] == 'M': result[i] = 1
    return result


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


def convert_seconds(seconds):
    time = float(seconds)

    days = int(time // (24 * 3600))
    time = time % (24 * 3600)

    hours = str(int(time // 3600))
    time %= 3600

    minutes = str(int(time // 60))
    time %= 60

    seconds = str(round(time, 2))

    if days != 0:
        result = '(' + str(days) + 'd' + hours + 'h' + minutes + 'm' + seconds + 's)'

    else:
        result = '(' + hours + 'h' + minutes + 'm' + seconds + 's)'

    return result


def print_header(script_name, end=False):
    now = datetime.now()

    if end == True:

        print('\n\n=============================================================')
        print('\n>> Running of ' + str(script_name) + ' FINISHED!!')
        print('\n', now.strftime("%x"), now.strftime("%X"))
        print('=============================================================\n')

    else:

        print('\n\n=============================================================')
        print('\n   >> Running ' + str(script_name) + '...')
        print('\n', now.strftime("%x"), now.strftime("%X"))
        print('=============================================================\n')

    return


def directory_check(root_path, directory_name, preserve=True):
    list_dir = os.listdir(root_path)
    directory_path = os.path.join(root_path, directory_name)

    if preserve == False:

        if directory_name not in list_dir: os.makedirs(directory_path)

    else:

        done = False
        length = len(directory_name)
        n = 3

        while done == False:

            if directory_name in list_dir and len(directory_name) == length:

                directory_name += '(2)'

            elif directory_name in list_dir:

                index_parenthesis = directory_name.rindex('(') + 1

                directory_name = directory_name[0:index_parenthesis] + str(n) + ')'
                n += 1

            else:

                directory_path = os.path.join(root_path, directory_name)
                os.makedirs(directory_path)
                done = True

    return directory_path


def classification_performance(features_train, predictions_array, labels_test, exec_time, output_path, output_name,
                               partition):
    predictions_array = round_to_single_column(predictions_array)
    labels_test = round_to_single_column(labels_test)

    CM = Confusion_Matrix(predictions_array, labels_test)
    accuracy = Acc(CM)
    precision = Precision(CM)
    recall = Recall(CM)
    f1_score = F1_score(CM)
    g_mean = Geometric_Mean(CM)
    rac = roc_auc_score(labels_test, predictions_array)
    exec_time = round(exec_time, 2)
    exec_time_str = convert_seconds(exec_time)

    list_dir = os.listdir(output_path)
    csv_metrics_name = output_name + '_classification.csv'
    csv_metrics_path = os.path.join(output_path, csv_metrics_name)

    metrics_tags = ['Test_part', 'N_train', 'N_test', 'Train_time(s)', 'Train_time',
                    "TP", "FP", "TN", "FN",'Accuracy', 'Precision', 'Recall', 'F1_score', 'Geometric_Mean', "Roc_Auc"]
    new_raw_data = [partition, len(features_train), len(labels_test), exec_time, exec_time_str,
                    CM[0], CM[2], CM[3], CM[1], accuracy, precision, recall, f1_score, g_mean, rac]

    new_row = pd.DataFrame(data=[new_raw_data], columns=metrics_tags)

    if csv_metrics_name not in list_dir:

        csv_metrics = new_row

    else:

        csv_metrics = pd.read_csv(csv_metrics_path)
        csv_metrics = csv_metrics.append(new_row, ignore_index=True)

    csv_metrics.to_csv(csv_metrics_path, index=False, float_format='%.4f')

    if partition == 5: global_statistics(output_path, output_name)

    return


# (3.3) Function to print to a .txt and .csv files all runnning information and classifier performance
def save_predictions(data_test, predictions_array, partition, classifier_tag, output_path, output_name):
    csv_name = output_name + '_test_results_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)

    classifier_tag = classifier_tag

    data_test[classifier_tag] = np.array(predictions_array, dtype=int)

    data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'EXPERT', classifier_tag]]

    data_test.to_csv(csv_path, index=False)

    return


# (3.4) Function to print to a .txt and .csv files all runnning information and classifier performance
def save_predictions_CNN(data_test, predictions_array, partition, output_path, output_name):
    csv_name = output_name + '_test_results_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)

    data_test['CNN_NM'] = np.array(predictions_array[:, 0])
    data_test['CNN_M'] = np.array(predictions_array[:, 1])

    # print(data_test)

    # data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'EXPERT', 'CNN_EL', 'CNN_SP']]
    # data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'CNN_EL', 'CNN_SP']]
    # data_test = data_test[["'OBJID', 'EXPERT' , 'CNN_EL', 'CNN_SP'"]]

    data_test.to_csv(csv_path, index=False)

    return


# (3.4) Function to compute the mean and std of the csv file originated in (3)
def global_statistics(output_file_dir, output_filename):
    # csv_name = os.path.basename(output_file_dir) + '.csv'

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
                         'F1_mean', 'F1_std', 'Geometric_Mean_Mean', 'Geometric_Mean_std', 'Train_time(s)',
                         'Train_time(s)_std']
    meta_metrics_data = [acc.mean(), acc.std(), pre.mean(), pre.std(), rec.mean(), rec.std(),
                         f1s.mean(), f1s.std(), gom.mean(), gom.std(), time.mean(), time.std()]

    ### IMPORTANT!!: This code implements (n - 1)-std function

    csv_result = pd.DataFrame(data=[meta_metrics_data], columns=meta_metrics_tags)

    csv_result_path = os.path.join(output_file_dir, '_' + output_filename + '_classification_summary.csv')
    csv_result.to_csv(csv_result_path, index=False, float_format='%.4f')

    return


# (13) Function to convert a two-column binary vector to a single column binary vector
def round_to_single_column(two_columns_array):
    length = len(two_columns_array)
    result = np.zeros((length,), dtype=int)

    for i in range(0, length):

        if two_columns_array[i, 1] > two_columns_array[i, 0]: result[i] = 1

    return result


# (2.1) Function to evaluate predictions' Confusion Matrix
def Confusion_Matrix(predictions_array, labels_array):
    if len(predictions_array) != len(labels_array):
        print('\n >> ERROR utils_experiments(2.1): Label arrays length do not match!\n')
        sys.exit()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(0, len(predictions_array)):

        if predictions_array[i] == 1 and labels_array[i] == 1: TP += 1
        if predictions_array[i] == 1 and labels_array[i] == 0: FP += 1
        if predictions_array[i] == 0 and labels_array[i] == 0: TN += 1
        if predictions_array[i] == 0 and labels_array[i] == 1: FN += 1

    return (TP, FN, FP, TN)


# (2.2) Function to evaluate predictions' Accuracy from Confusion Matrix
def Acc(confusion_matrix):
    trues = confusion_matrix[0] + confusion_matrix[3]
    total = trues + confusion_matrix[1] + confusion_matrix[2]

    result = (trues * 1.0) / total

    return round(result, 4)


# (2.3) Function to evaluate predictions' Precison from Confusion Matrix
def Precision(confusion_matrix):
    TP = confusion_matrix[0] * 1.0
    FP = confusion_matrix[2] * 1.0

    all_Positive = TP + FP

    if all_Positive != 0.0:
        precision = TP / all_Positive

    else:
        precision = 0.0

    return round(precision, 4)


# (2.4) Function to evaluate predictions' Recall from Confusion Matrix
def Recall(confusion_matrix):
    TP = confusion_matrix[0] * 1.0
    FN = confusion_matrix[1] * 1.0

    denom = TP + FN

    if denom != 0.0:
        recall = TP / denom

    else:
        recall = 0.0

    return round(recall, 4)


# (2.5) Function to evaluate predictions' F1 Score from Confusion Matrix
def F1_score(confusion_matrix):
    precision = Precision(confusion_matrix)
    recall = Recall(confusion_matrix)

    denom = precision + recall

    if denom != 0.0:
        F1_score = (2 * precision * recall) / denom

    else:
        F1_score = 0.0

    return round(F1_score, 4)


def Geometric_Mean(confusion_matrix):
    TP = confusion_matrix[0] * 1.0
    FN = confusion_matrix[1] * 1.0
    FP = confusion_matrix[2] * 1.0
    TN = confusion_matrix[3] * 1.0

    if TP + FN == 0 or FP + TN == 0:
        g_mean = 0
    else:
        g_mean = np.sqrt((TP/(TP + FN)) * (TN / (FP + TN)))
    return round(g_mean, 4)
