import pandas as pd
import random as rnd


merger = pd.read_csv("../Data/GZ1_Trunc_Merger.csv")


def oversampler(dataset, size, seed = 42):
    rnd.seed(seed)
    updated_dataset = dataset
    while updated_dataset.index.size < size:
        for row_num in range(dataset.index.size):
            inserts = rnd.randint(0,6)
            for i in range(inserts):
                updated_dataset = updated_dataset.append(dataset.iloc[row_num])
                if updated_dataset.index.size >= size:
                    break
    return updated_dataset


oversampled_merger = oversampler(merger, 10500)
oversampled_merger.to_csv("../Data/GZ1_Oversampled_Datasets/20_pc_oversampled_Mergers.csv")

oversampled_merger = oversampler(merger, 22500)
oversampled_merger.to_csv("../Data/GZ1_Oversampled_Datasets/35_pc_oversampled_Mergers.csv")

oversampled_merger = oversampler(merger, 40928)
oversampled_merger.to_csv("../Data/GZ1_Oversampled_Datasets/50_pc_oversampled_Mergers.csv")

