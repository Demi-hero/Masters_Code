import numpy as np
import pandas as pd
import os
from keras.models import model_from_json
import cv2
import sys


class Dcgan_Merger_Generator:

    def __init__(self, model_path, weight_path, single_file=False, image_grid=(5, 5)):
        if single_file:
            self.model_path = model_path
            # create the single file model when I need it
        else:
            self.model_path = model_path
            self.weight_path = weight_path
            self.image_grid = image_grid
            self.Merger_Maker = self.read_in_gan_json_model()

    def read_in_gan_json_model(self):
        json_file = open(self.model_path, 'r')
        loaded = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded)
        self.model.load_weights(self.weight_path)
        return self.model

    def dcgan_images(self):
        r = self.image_grid[0]
        c = self.image_grid[1]
        noise = np.random.normal(0,1, (r * c, 100))
        images = self.model.predict(noise)
        return images

    def dcgan_human_images(self):
        images = self.dcgan_images()
        images = (images + 1) * 127.5
        return images

    def dcgan_cnn_imgs(self):
        images = self.dcgan_human_images()
        images = images.astype('float32') / 255.0
        return images
