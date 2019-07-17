import numpy as np
import pandas as pd
import cv2
import os
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot
# new image paths
CWD = "../../Data"
CSV ="__CSV__"
IMAGES ="__IMAGES__/image_catalogue"
file_name = "GZ1_Full_Expert_Paths.csv"
image_library = "GZ1_Expert_Merger__Augmented_64x_tiff"

# read in the csv and crate base image path

csv_path = os.path.join(CWD, CSV, file_name)
path_file = pd.read_csv(csv_path)

merger_subset = path_file[path_file.EXPERT == "M"]
path_list = list(merger_subset.Paths)

ids = list(path_file.OBJID)
new_id = max(ids) + 1
generator = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True,
                               vertical_flip=True, rotation_range=170,zoom_range=[0.5, 1.0])

for path in path_list[:2]:
    path = os.path.join("..", path)
    print(path)
    img = load_img(path)
    img_array = img_to_array(img)
    sample = np.expand_dims(img_array, 0)
    transformer = generator.flow(sample, batch_size=1)
    for i in range(9):
        # generate batch of images
        batch = transformer.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # make the new file path
        pyplot.imshow(image)
    # show the figure
    pyplot.show()





