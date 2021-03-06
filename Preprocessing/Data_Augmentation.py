import numpy as np
import pandas as pd
import cv2
import os
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from _preproces_utils import zero_placer
import sys

# new image paths
CWD = "..\\..\\Data"
CSV = "__CSV__"
IMAGES = "__IMAGES__\\image_catalogue"
file_name = "GZ1_Full_Expert_Paths.csv"
output_image_library = "GZ1_Expert_Merger_64x_tiff_Aug_only"
output = os.path.join(CWD, IMAGES, output_image_library)
# set the random seed
seed_value = 42
total_img_gens = 20
# read in the csv and crate base image path
csv_path = os.path.join(CWD, CSV, file_name)
source_file = pd.read_csv(csv_path)
path_file = pd.DataFrame(columns=['OBJID', 'Source_Lables', 'EXPERT', 'Paths'])
merger_subset = source_file[source_file.EXPERT == "M"]
path_list = list(merger_subset.Paths)
total = len(path_list) * total_img_gens
ids = list(source_file.OBJID)

generator = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True,
                               vertical_flip=True, rotation_range=170, zoom_range=[0.5, 1.0])

lower = 1
upper = 1000
counter = 0
cur_out_file = "{output}/{zp_lower}{lower}-{zp_upper}{upper}".format(output=output, zp_lower=zero_placer(lower),
                                                                     lower=lower, zp_upper=zero_placer(upper),
                                                                     upper=upper)
img_cat = os.path.join(cur_out_file)
os.mkdir(img_cat)

np.random.seed(seed_value)
print("Starting Image Generation")
print("Generating {zeroes}{value}st Transformation ".format(zeroes=zero_placer(lower), value=lower))
itters = 0
while itters < total:
    for oid, path in zip(merger_subset.OBJID, merger_subset.Paths):
        path = os.path.join("..", path)
        img = load_img(path)
        img_array = img_to_array(img)
        sample = np.expand_dims(img_array, 0)
        transformer = generator.flow(sample, batch_size=1)
        for i in range(total_img_gens):
            # generate batch of images
            batch = transformer.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # make the output path
            new_id = "{ID}_{change}".format(ID=oid, change=i)
            destination = os.path.join(img_cat, "{new_id}.tiff".format(new_id=new_id))
            # write out the image to the output
            cv2.imwrite(destination, image)
            counter += 1
            itters += 1
            # attach it to the csv
            path_file = path_file.append({"OBJID": new_id, "Source_Lables": "M", "EXPERT": "M",
                                          "Paths": destination[3:]}, ignore_index=True)
            # check to see if we have filled a file if yes make a new one.
            if counter >= 1000:
                counter = 0
                lower += 1000
                upper += 1000
                print("Generating the {zeroes}{value}th Transformed Image".format(zeroes=zero_placer(lower),
                                                                                  value=lower))
                cur_out_file = "{output}/{zp_lower}{lower}-{zp_upper}{upper}".format(output=output,
                                                                                     zp_lower=zero_placer(lower),
                                                                                     lower=lower,
                                                                                     zp_upper=zero_placer(upper),
                                                                                     upper=upper)
                img_cat = os.path.join(cur_out_file)
                os.mkdir(img_cat)
        if itters > total:
            break
csv_output = os.path.join(CWD, CSV, "GZ1_Full_Expert_Augmented_Only_Paths.csv")
path_file.to_csv(csv_output, index=False)
