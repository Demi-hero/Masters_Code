import numpy as np
import matplotlib.pyplot as plt
import os
import json
from keras.models import model_from_json, load_model

toppath = "GANs\\DCGan\\Saved_Model"
js = "galaxy_dcgan_generator.json"
weights = "galaxy_dcgan_generator_weights.hdf5"

def read_in_gan_json_model(model_path, model_name, weight_name):
    # creates a model from a json file and h5 weight file in the same directory
    js = os.path.join(model_path, model_name)
    weigths = os.path.join(model_path, weight_name)
    json_file = open(js, "r")
    loaded = json_file.read()
    json_file.close()
    model = model_from_json(loaded)
    model.load_weights(weigths)
    return model


model = read_in_gan_json_model(toppath, js, weights)
#model = load_model("GANs/DCGan/Saved_Model/galaxy_dcgan_generator_full_system.hdf5")
model.summary()
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

r, c = 5, 5
noise = np.random.normal(0, 1, (r*c, 100))

gen_imgs = model.predict(noise)

gen_imgs = 0.5 * gen_imgs + 0.5

x = 0
img = plt.imshow(gen_imgs[x, :, :])
plt.show()
#img.axis('off')



