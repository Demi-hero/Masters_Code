import pandas as pd

core_path = "../Data"
output = "../Data/Merged Datasets"
expert      = pd.read_csv(f"{core_path}/GZ1_Trunc_Expert.csv")
oversampl20 = pd.read_csv(f"{core_path}/GZ1_Oversampled_Datasets/20_pc_oversampled_Mergers.csv")
oversampl35 = pd.read_csv(f"{core_path}/GZ1_Oversampled_Datasets/35_pc_oversampled_Mergers.csv")
oversampl50 = pd.read_csv(f"{core_path}/GZ1_Oversampled_Datasets/50_pc_oversampled_Mergers.csv")

expert = expert.OBJID

final_20 = expert.append(oversampl20.OBJID_1).sample(frac=1).reset_index(drop=True)
final_35 = expert.append(oversampl35.OBJID_1).sample(frac=1).reset_index(drop=True)
final_50 = expert.append(oversampl50.OBJID_1).sample(frac=1).reset_index(drop=True)

combi = [final_20, final_35, final_50]
names = ['final_20', 'final_35', 'final_50']
for items,names in zip(combi,names):
    items.to_csv(f"{output}/{names}.csv", header=True, index=False)

