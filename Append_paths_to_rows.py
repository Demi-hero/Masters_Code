import pandas as pd
import numpy as np
import os


# This section adds A File Path to the end of a provided CSV
image_ids = pd.read_csv("../Data/__CSV__/GZ1_Full_Expert.csv")

CWD = "..\\Data"
images_root = "__IMAGES__"
csv_root    = "__CSV__"
images_folder = "Blended_Image_Catalouge_tiff"
images_csv = "GZ1_Full_Expert.csv"
file_extension = ".tiff"
folder_path = os.path.join(CWD, images_root, images_folder)
print(folder_path)
folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])
images_IDs = np.array(image_ids['OBJID'], dtype=str)
image_paths = list()
print(image_paths)
for i in range(0, len(image_ids.OBJID)):
    if i % 1000 == 0:
        folder = folders[int(i / 1000)]

    image_name = images_IDs[i] + file_extension
    image_path = os.path.join(folder_path, folder, image_name)
    image_paths.append(image_path)

image_ids["Paths"] = image_paths
outpath = os.path.join(CWD,csv_root, "GZ1_Full_Expert_Paths.csv")
image_ids.to_csv(outpath, index=False)