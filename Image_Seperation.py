import pandas as pd
import os
import shutil

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


def zero_placer(base, overall_length=8):
    length = len(str(base))
    return "0" * (overall_length - length)


def mass_migration(origin, output, key):
    # Moves files from one place to another if their name appears in the key.
    # Groups them by batches of 1000
    lower = 1
    upper = 1000
    counter = 0
    os.mkdir(f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
    for root, dirs, files in os.walk(origin):
        try:
            for file in files:
                if int(os.path.splitext(file)[0]) in key:
                    source = os.path.join(root, file)
                    shutil.copy(source, f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
                    counter += 1
                    print(f"{file} added")
                    if counter >= 1000:
                        counter = 0
                        lower += 1000
                        upper += 1000
                        os.mkdir(f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
        except FileNotFoundError:
            print(f"File {file} not found")


mass_migration(source_path, output_path, ok_id)
