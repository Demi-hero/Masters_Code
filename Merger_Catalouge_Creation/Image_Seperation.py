import pandas as pd
from _preproces_utils import mass_migration

# Paths for the data to follow. These both need to exist
core_path = "D:/Documents/Comp Sci Masters/Project_Data/Data"
source_path = f"{core_path}/GZ1_T2_64x_tiff"
output_path = f"{core_path}/GZ1_Expert_Merger_64x_tiff"


# Get needed IDs from source doc
id_data = pd.read_csv(f"{core_path}/GZ1_MG-T2_cross-search.csv")

id_data_fin = id_data[id_data.IN_T2_1 == 1]
id_data_fin = id_data_fin[id_data.IN_T2_2 == 0]

ok_id = id_data_fin["OBJID_1"].tolist()

# Helpers

mass_migration(source_path, output_path, ok_id)
