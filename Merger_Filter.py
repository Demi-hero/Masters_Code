import pandas as pd

expert_data = pd.read_csv("../Data/GZ1_Expert_Master.csv")
merger_data = pd.read_csv("../Data/GZ1_MG-T2_cross-search.csv")

# Create a list of Droppable Object IDs

ex_set1 = merger_data[merger_data.IN_EXP_1 == 1]
ex_set2 = merger_data[merger_data.IN_EXP_2 == 1]

ids_to_drop = ex_set1.OBJID_1.append(ex_set2.OBJID_2)

# drop out matching pairs from the master table
# ~ returns the compliment of the operation
new_expert = expert_data[~expert_data.OBJID.isin(ids_to_drop)]

# check the output has been moded
print(len(expert_data.index))
print(len(new_expert.index))

# write out this modded expert list

new_expert.to_csv("../Data/GZ1_Trunc_Expert.csv", header=False)
