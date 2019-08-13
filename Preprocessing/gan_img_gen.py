import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json
import cv2
import sys


# Target Locations
ref_string = "..\\Data\\__CSV__"
ref_file = "GZ1_Full_Expert__Augment_Paths.csv"
GAN_path = "GANs\\DCGan"
model_file = "Saved_Model"
js = "galaxy_dcgan_generator.json"
weights = "galaxy_dcgan_generator_weights.hdf5"
output = os.path.join(GAN_path, "generator_data")
# defining the ID of the first image
ref_path = os.path.join(ref_string, ref_file)
ref_ids = pd.read_csv(ref_path)
id = max(ref_ids.OBJID) + 1
my_dpi = 96

# multiply the batches value by 25 to find the final volume of 64x64 images created
batches = 5
def read_in_gan_json_model(GAN_path, model_file, model_name, weight_name):
    # creates a model from a json file and h5 weight file in the same directory
    js = os.path.join(GAN_path, model_file, model_name)
    weigths = os.path.join(GAN_path, model_file, weight_name)
    json_file = open(js, "r")
    loaded = json_file.read()
    json_file.close()
    model = model_from_json(loaded)
    model.load_weights(weigths)
    return model


model = read_in_gan_json_model(GAN_path, model_file, js, weights)
model.summary()

r, c = 5, 5
i = 0
df = pd.DataFrame(columns=['OBJID', 'Source_Lables', 'EXPERT', 'Paths'])

while i < batches:
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = model.predict(noise)
    x = 0
    file_path = "../Data/__IMAGES__/GAN_Images/DCGAN"
    file_name = "{id}.tiff".format(id=id)
    while x <= 24:
        file_path = "../Data/__IMAGES__/GAN_Images/DCGAN"
        file_name = "{id}.tiff".format(id=id)
        output = os.path.join(file_path, file_name)
        image = (gen_imgs[x,:,:] + 1) * 127.5
        image = image.astype('uint8')
        cv2.imwrite(output, image)
        x += 1
        id += 1
    i += 1
    print("{i} image batches generated".format(i=i))






