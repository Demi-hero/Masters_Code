import pandas as pd
from Merger_Catalouge_Creation._preproces_utils import create_catalouge

source = "../Data/image_catalogue"
ids = pd.read_csv("../Data/GZ1_Full_Expert.csv")
output = "../Data/Blended_Image_Catalouge/"

create_catalouge(source,output,list(ids["OBJID"]))
