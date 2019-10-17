import os
import cv2
import numpy as np
import pandas as pd
from Image_Generator import Merger_Generator
from Convolutional_CNN import Conv128_3_NN
from _NOC_Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### File Locations, Set these for your File structure ######
# Magic Numbers
colour_channels = 3
image_tuple = (64, 64, colour_channels)
dt = (25, 64, 64, colour_channels)
# Multipy the shape of an encoded image together to get this value
flat_encode = 8*8*8
white_lines = np.ones(image_tuple)


# Generator Paths
json_path = 'D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\DCGan\Saved_Model\galaxy_dcgan_generator.json'
weight_path = 'D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\DCGan\Saved_Model\galaxy_dcgan_generator_weights.hdf5'

# Classifier Paths
class_weight_path = "saved_model/Test_weights.hdf5"

# img paths
data_directory = "../../Data/"
img_root = "__IMAGES__"
img_folder = "Real_vs_Gen"
input_img = 'dc_gan.jpg'
input_path = os.path.join(data_directory, img_root, input_img)
gen_list = ["DCGAN", "SGAN", "StyleGAN", 'Other']
gen_type = gen_list[-1]
output_file = 'DCGAN'
img_out_path = os.path.join(data_directory, img_root, img_folder, output_file)

# csv Paths
csv_root = "__CSV__"
csv_dest = "EDs"
csv_in_file = "GZ1_Full_Expert_Paths.csv"
csv_out_file = "{}_Generated_Eds.csv".format(gen_type)
csv_path = os.path.join(data_directory, csv_root, csv_in_file)
csv_out = os.path.join(data_directory, csv_root, csv_dest,csv_out_file)



#######################
###### UTIL FUNCTIONS #####

def encoded_flatner(encoded_tensor, flat_shape):
    encoded_flat_array = np.zeros((len(encoded_tensor), flat_shape), dtype=float)
    for i in range(len(encoded_tensor)):
        encoded_flat_array[i] = np.reshape(encoded_tensor[i], flat_shape)
    return encoded_flat_array

def read_in_tiled_img(path, dim_tuple, rgb =1 ):
    image = cv2.imread(path, rgb)
    img_array = np.zeros(dim_tuple, dtype=float)
    cntr = 0
    for i in range(5):
        for j in range(5):
            img = image[64*i:64*(i+1), 64*j:64*(j+1), :]
            img_array[cntr] = img.astype("float32")
            cntr += 1
    return img_array

###########################
# Create the Image Generator if doing it on fresh data
if gen_type == "DCGAN" or gen_type == "SGAN":
    merger_maker = Merger_Generator(json_path, weight_path)
    generate = 1
elif gen_type == "StyleGAN":
    merger_maker = WGAN(lr=0.0003, silent = False)
    merger_maker.load(129)
    generate = 1
else:
    gan_img = read_in_tiled_img(input_path, dt)
    gan_img = gan_img / 255
    generate = 0

if generate:
    gan_img = merger_maker.img_generator()
    gan_img = gan_img[:25, :, :, :]
# Create the Classifier
classifier = Conv128_3_NN(image_tuple)
classifier.load_weights(class_weight_path)

# Read in the Merger Galaxy Images normalised to 0-1.
data = pd.read_csv(csv_path)
data = data[data.EXPERT == 'M']#[:25]
data_tens = create_image_tensor_on_path(data.Paths, image_tuple, extra_path_details="..")


# Encode all n of these Galaxies.
enc_img = classifier.encoded_images(data_tens)

# Reshape them all into the nescissary array
flat_mergers = encoded_flatner(enc_img, flat_encode)

# Encode the 25 gened images
gan_enc_img = classifier.encoded_images(gan_img)
flat_gan_enc = encoded_flatner(gan_enc_img,flat_encode)
dist = 10000
ind = 0
ind_list= []
dists = []
# For Each gan img calculate the ed against real images. Keep the lowest distance.
for gan_ind in range(len(flat_gan_enc)):
    for merg_ind in range(len(flat_mergers)):
        ed = np.sqrt(np.sum((flat_gan_enc[gan_ind] - flat_mergers[merg_ind]) ** 2))
        if ed < dist:
            dist = ed
            ind = merg_ind

    ind_list.append(ind)
    dists.append(dist)
    dist = 10000
    ind = 0
# Takes the list of real img indexs, uses those and listed index positions to compaare them to the gans.
# names them with ED in the name.
labs = []
for ind, val, in enumerate(ind_list):
    gi = ind
    ri = val
    gimg = gan_img[gi, :, :, :]
    rimg = data_tens[ri, :, :, :]
    imgs = np.hstack((gimg, white_lines, rimg))
    imgs = imgs * 255
    rid = data.iloc[ri, 0]
    rlbl = data.iloc[ri, 2]
    labs.append(rlbl)
    # imwrite needs 0-255 but imshow can handle 0-1s ... that seems like a flaw
    img_out = os.path.join(img_out_path, "gen_{gid}_vs_lbl_rid_{rid}_{rlbl}.png".format(rid=rid, gid=ind, rlbl=rlbl))
    cv2.imwrite(img_out, imgs)

dict = {"OBJID": ind_list, "GANID": list(range(1, 26)), "Expert Lable":labs, "Euclidian Distance": dists}
output = pd.DataFrame(data=dict)
output.to_csv(csv_out, index=False)
