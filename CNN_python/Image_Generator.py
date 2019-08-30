import numpy as np
import pandas as pd
import os
from keras.models import model_from_json
import cv2
import sys

def noise(n, latent_size):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

#Noise Sample
def noiseImage(n, im_size):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1])

class Merger_Generator:

    def __init__(self, model_path, weight_path, single_file=False, image_grid=(5, 5), latent_dims=100):
        self.image_grid = image_grid
        self.latent_dims = latent_dims
        if single_file:
            self.model_path = model_path
            # create the single file model when I need it
        else:
            self.model_path = model_path
            self.weight_path = weight_path
            self.Merger_Maker = self.read_in_gan_json_model()

    def read_in_gan_json_model(self):
        json_file = open(self.model_path, 'r')
        loaded = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded)
        self.model.load_weights(self.weight_path)
        return self.model

   #playing with fire
    def predict(self, inpt, resize_tpl=(128,128)):
        imgs = self.model.predict(inpt)
        for i in range(0, imgs.shape[0]):
            imgs[i] = cv2.resize(imgs[i, :, :, :], dsize=resize_tpl, interpolation=cv2.INTER_CUBIC)
        return imgs

   # The 3 functions for returning images from single input generators
    def si_images(self):
        r = self.image_grid[0]
        c = self.image_grid[1]
        noise = np.random.normal(0,1, (r * c, self.latent_dims))
        images = self.model.predict(noise)
        return images

    def si_human_images(self):
        images = self.si_images()
        images = (images + 1) * 127.5
        return images

    def si_cnn_imgs(self):
        images = self.si_human_images()
        images = images.astype('float32') / 255.0
        return images

    def img_generator(self):
        return self.si_cnn_imgs()


if __name__ == "__main__":
    from PIL import Image
    json_path = 'D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\SGan\Saved_Model\galaxy_sgan_generator.json'
    weight_path = 'D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\SGan\Saved_Model\galaxy_sgan_generator_weights.hdf5'
    merger_maker = Merger_Generator(json_path, weight_path)
    for i in range(100):
        im2 = merger_maker.si_human_images()
        r12 = np.concatenate(im2[:5], axis=1)
        r22 = np.concatenate(im2[5:10], axis=1)
        r32 = np.concatenate(im2[10:15], axis=1)
        r43 = np.concatenate(im2[15:20], axis=1)
        r44 = np.concatenate(im2[20:25], axis=1)
        c1 = np.concatenate([r12, r22, r32, r43, r44], axis=0)
        x = Image.fromarray(np.uint8(c1))
        x.save("S_IMG/s_img_{}.jpg".format(i))
