import os
import shutil
import sys

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
                    if counter >= 1000:
                        counter = 0
                        lower += 1000
                        upper += 1000
                        os.mkdir(f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
        except FileNotFoundError:
            print(f"File {file} not found")


def create_catalouge(origin,output, catalouge_id):
    # Approaches the mass migration probelem from the other angle
    # Don't use this if you can avoid it. Its super slow.
    lower = 1
    upper = 1000
    counter = 0
    os.mkdir(f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
    for value in catalouge_id:
        for root, dirs, files in os.walk(origin):
            try:
                for file in files:
                    if int(os.path.splitext(file)[0]) == value:
                        source = os.path.join(root, file)
                        shutil.copy(source, f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
                        counter += 1
                        if counter >= 1000:
                            print(f"Directory {output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper} "
                                  f"filled with {counter} images")
                            counter = 0
                            lower += 1000
                            upper += 1000
                            os.mkdir(f"{output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}")
                            print(f"New Directory {output}/{zero_placer(lower)}{lower}-{zero_placer(upper)}{upper}made")
            except FileNotFoundError:
                print(f"File {file} not found")


def expert_label_renamer(row):
    if row['EXPERT'] == "M":
        # Gives the Label Merger Galaxy
        return "E"
    elif row['EXPERT'] == "L":
        # Give the label Singular Galaxy
        return "S"
    else:
        pass


def move_and_name_change(source,destination, suffix):
    # Copy files from one place to another changing their name to img (int).suffix
    counter = 1
    for root, dirs,files in os.walk(source):
        try:
            for file in files:
                src = os.path.join(root, file)
                dest = os.path.join(destination,f"im ({counter}).{suffix}")
                shutil.copy(src, dest)
                counter += 1
                if counter % 1000 == 0:
                    print(f"{counter} images moved")
        except FileNotFoundError:
            print(f"File {file} not found")
