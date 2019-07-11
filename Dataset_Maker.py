import pandas as pd
from _preproces_utils import expert_label_renamer

core_path = "../Data/__CSV__"
output = "../Data/"
expert = pd.read_csv(f"{core_path}/GZ1_Trunc_Expert.csv")
merger = pd.read_csv(f"{core_path}/GZ1_Trunc_Merger.csv")


expert["Label"] = expert.apply(lambda row: expert_label_renamer(row), axis=1)
merger["Label"] = "M"
merger["OBJID"] = merger["OBJID_1"]
expert = expert[["OBJID", "Label"]]
merger = merger[["OBJID", "Label"]]
output = expert.append(merger).sample(frac=1, random_state=42).reset_index()

output = output[["OBJID","Label"]]
print(output.head(10))

output.to_csv(f"{core_path}/GZ1_Full_Expert2.csv")
