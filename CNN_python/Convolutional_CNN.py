# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:30:33 2019

@author: psxmaji
"""
import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg') 
plt.switch_backend('agg')

from _utils_CNN import convert_seconds

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model
from keras import backend as K

K.clear_session()

#==================================================================================================
#================================ Convolutional Neural Network ====================================
#==================================================================================================        
""" 
Convolutional Neural Network (CNN) made up of four consecutive Conv--Pooling pairs of layers. 
ReLU activation functions in all layers except the last one, which implements Sigmoid.
MaxPooling used. 

""" 


class Conv128_3_NN : 
    
    def __init__ (self, input_tuple) : 
        
        self.input_tuple = input_tuple
        
        K.clear_session()
        
        self.build()
        
    
    def build (self) : 
        
        input_img = Input(shape=self.input_tuple) # (64, 64, 1)
        
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # (64, 64, 8)
        
        x = MaxPooling2D((2, 2), padding='same')(x)                          # (32, 32, 8)
        
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (16, 16, 32)

        x = MaxPooling2D((2, 2), padding='same')(x)                          # (8, 8, 32)
        
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (8, 8, 64)
        
        encoded = MaxPooling2D((2, 2), padding='same')(x)                    # (2, 2, 64) = 256 feats.
        
        x = Flatten()(encoded)
        
        x = Dense(256, activation='relu')(x)  # (2, 2, 64)
                
        x = Dense(128, activation='relu')(x)  # (8, 8, 32)
        
        output = Dense(2, activation='softmax')(x)
    
    
        self.model = Model(inputs=input_img, outputs=output)

    def train(self, train_images, train_labels, val_images, val_labels, optimizer='sgd',
              loss='mean_squared_error', epochs=100, batch_size=256, shuffle=True):
        
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_images = train_images
        self.train_labels = train_labels
        
        self.val_images = val_images
        self.val_labels = val_labels
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        
        start_time = time.time()
        record = self.model.fit(self.train_images, self.train_labels, epochs=self.epochs, shuffle=self.shuffle,
                                 validation_data=(self.val_images, self.val_labels))
        end_time = time.time() 

        self.history = record.history.copy()
        self.exec_time = end_time - start_time
        
        
    def trial_log (self, output_folder_path, trial_name, test_partition=0, csv_global=False) :
                
        self.output_folder_path = output_folder_path
        self.trial_name = trial_name
        self.test_partition = test_partition
        
        output_path = os.path.join(self.output_folder_path, self.trial_name + '_' + str(self.test_partition))
        runtime_str = convert_seconds(self.exec_time)
        train_time = round(self.exec_time, 2)

        
        # History CSV:
        result_history = pd.DataFrame(data=self.history)
        history_csv_path = output_path + '_history.csv'
        result_history.to_csv(history_csv_path, index=None, float_format='%.4f')
       
        # History plot:
        plt.ioff()
        plt.figure()
        
        n = len(self.history['loss'])
        x = np.arange(1, n + 1, 1)
        plt.plot(x, self.history['loss'], label='loss')
        plt.plot(x, self.history['val_loss'], label='val_loss')
        
        step = int(n / 10) + 1
            
        plt.xticks([i for i in x], [str(i) if i % step == 0 else '' for i in x])
        plt.title(self.trial_name, fontsize=12)
        plt.legend()
        plt.grid()
        
        figure_path = output_path + '_training.png'
        plt.savefig(figure_path)
        
        # Training times CSV: 
        if csv_global == True : 
            
            list_dir = os.listdir(self.output_folder_path)
            
            csv_name = self.trial_name + '_training_global.csv'
            csv_path = os.path.join(self.output_folder_path, csv_name)
            
            columns_tags = ['Test_partition', 'Train_time(s)', 'Train_time', 'N_epochs']
            new_row_data = [int(self.test_partition), train_time, runtime_str, self.epochs]
            
            new_row = pd.DataFrame(data=[new_row_data], columns=columns_tags)  
                    
            if csv_name not in list_dir : 
                
                csv_global = new_row 
                
            else : 
                
                csv_global = pd.read_csv(csv_path)                   
                csv_global = csv_global.append(new_row, ignore_index=True)
                           
            csv_global.to_csv(csv_path, index=False) # float_format='%.4f'            
                
        # Log .txt file:        
        text_file = output_path + '_log.txt'
        
        file = open(text_file, 'w')

        file.write('\nOutput file generated from: ' + str(sys.argv[0]))       
        now = datetime.now()
        file.write('\r\nDate and time: ' + now.strftime("%x") + ' ' + now.strftime("%X")) 
        file.write('\r\n')        
        file.write('\r\nModel: Conv128+3 CNN')
        file.write('\r\nInput images: ' + str(self.input_tuple))
        file.write('\r\n')
        file.write('\r\nOptimizer: ' + str(self.optimizer))
        file.write('\r\nLoss func.: ' + str(self.loss))
        file.write('\r\nEpochs: ' + str(self.epochs))
        file.write('\r\nBatch size: ' + str(self.batch_size))
        file.write('\r\nShuffle: ' + str(self.shuffle))
        file.write('\r\n')
        file.write('\r\nTrain      : ' + str(len(self.train_images)) + ' examples')
        file.write('\r\nValidation : ' + str(len(self.val_images)) + ' examples')
        file.write('\r\nTotal: ' + str(len(self.train_images) + len(self.val_images)) + ' examples')
        file.write('\r\n')
        file.write('\r\nExecution time: ' + str(round(self.exec_time, 2)) + ' ' + runtime_str)
        file.write('\r\n')
        file.write('\r\nCNN structure summary:\n')
        file.write('\r\n')
        
        self.model.summary(print_fn=lambda x : file.write(x + '\r\n'))
        
        file.close()
        
        return train_time

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def encoded_images(self, img_array):
        enc_im = self.encoder.predict(img_array)
        return enc_im
###################################################################################################        
###################################################################################################        
###################################################################################################