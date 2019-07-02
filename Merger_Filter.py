import pandas as pd

data = pd.read_csv("../Data/GZ1_T2_Full.csv")


def mergers (row):
    if row["MG"] >= 0.5:
        return 1
    else:
        return 0


data["Merger"] = data.apply(lambda row:mergers(row), axis=1)

print(data[data.Merger == 1])
# print(data.head(10))

# mergers = data[data.MG >= 0.5]
# non_mergers = data[data.MG < 0.5]
# non_mergers.head(10)