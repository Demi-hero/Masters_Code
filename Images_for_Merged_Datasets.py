import pandas as pd

from _preproces_utils import mass_migration

source = "../Data/image_catalogue"
keys = "../Data/Merged_Datasets"
output = "../Data/_IMAGES_"

key_20 = pd.read_csv(f"{keys}/final_20.csv")
#key_35 = pd.read_csv(f"{keys}/final_35.csv")
#key_50 = pd.read_csv(f"{keys}/final_50.csv")

key_20 = key_20.tolist()

print(key_20)
