import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def cross_fold_train_test_split(image_ids, partition, n_splits = 5, test_size=0.2):


    data_partitions = [len(x) for x in np.array_split(image_ids, n_splits)]
    start_point = sum(data_partitions[0: (partition - 1)])
    end_point = start_point + data_partitions[partition - 1]

    training_set = pd.concat([image_ids.iloc[0:start_point], image_ids.iloc[end_point:]],axis=0)
    validation = image_ids.iloc[start_point : end_point]

    training, testing = train_test_split(training_set, test_size=test_size)
    return  training, testing, validation


image_ids = pd.read_csv("../../Data/__CSV__/GZ1_Full_Expert_Paths.csv")
n_splits = 5
for test_partition in range(1,6):
    train, test, val = cross_fold_train_test_split(image_ids,test_partition)
    print(len(train), len(test), len(val))
# Remove the Testing set dependant on the partition


"""
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


def data_split_cnn_test_expert(images_locs, labels_pd, test_partition, n_splits=5):
    if test_partition > n_splits or test_partition < 1:
        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()

    if len(images_locs) != len(labels_pd):
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()

    data_partitions = [len(x) for x in np.array_split(images_locs, n_splits)]
    start_point = sum(data_partitions[0: (test_partition - 1)])
    end_point = start_point + data_partitions[test_partition - 1]

    images_train = np.vstack((images_array[0: start_point], images_array[end_point:]))
    images_test = images_array[start_point: end_point]

    labels_selection = [i for i in range(0, start_point)] + \
                       [i for i in range(end_point, len(labels_pd))]

    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))

    labels_test = labels_pd[start_point: end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))

    labels_train_binary = np.array(labels_train['EXPERT'], dtype=str)
    labels_train_binary = convert_labels_expert(labels_train_binary, double_column=True)
    labels_test_binary = np.array(labels_test['EXPERT'], dtype=str)
    labels_test_binary = convert_labels_expert(labels_test_binary, double_column=True)

    return (images_train, labels_train, labels_train_binary,
            images_test, labels_test, labels_test_binary)


def data_split_cnn_training(images_array_train, labels_array_train, train_val_ratio=0.7):
    if len(images_array_train) != len(labels_array_train):
        print('\n >> ERROR utils_CNN(15.2): images and labels arrays length do not match!!\n')
        sys.exit()

    train_partition = int(train_val_ratio * len(images_array_train))
    # np.random.shuffle(images_array_train)

    images_train = images_array_train[0: train_partition]
    images_val = images_array_train[train_partition:]

    labels_train = labels_array_train[0: train_partition]
    labels_val = labels_array_train[train_partition:]

    return images_train, labels_train, images_val, labels_val

"""