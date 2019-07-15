import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
seed = 42
np.random.RandomState(seed=seed)
image_ids = pd.read_csv("../../Data/__CSV__/GZ1_Full_Expert_Paths.csv")
image_ids = image_ids[:1000]

ind_start = len(image_ids[image_ids.EXPERT == "NM"])

#def oversampler()
ros = RandomOverSampler(random_state=seed)

resample, relabel = ros.fit_resample(image_ids,image_ids.EXPERT.to_numpy())

#np.random.shuffle(resample)
resample = pd.DataFrame(resample, columns=["OBJID", "Source_Lable", "EXPERT", "Path"])

print(resample.EXPERT[995:1010])

print(len(resample[resample.EXPERT == "M"]))
print(len(resample[resample.EXPERT == "NM"]))

#resample.to_csv("../../Data/__CSV__/GZ1_Full_Expert_Paths_oversampled.csv")

