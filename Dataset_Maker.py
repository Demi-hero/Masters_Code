import pandas as pd

core_path = "../Data"

expert      = pd.read_csv(f"{core_path}/GZ1_Trunc_Expert.csv")
oversampl20 = pd.read_csv(f"{core_path}/GZ1_Oversampled_Datasets/20_pc_oversampled_Mergers.csv")
oversampl35 = pd.read_csv(f"{core_path}/GZ1_Oversampled_Datasets/35_pc_oversampled_Mergers.csv")
oversampl50 = pd.read_csv(f"{core_path}/GZ1_Oversampled_Datasets/50_pc_oversampled_Mergers.csv")

final_20 = expert.append(oversampl20).sample(frac=1)
final_35 = expert.append(oversampl35).sample(frac=1)
final_50 = expert.append(oversampl50).sample(frac=1)
