from matplotlib import pyplot as plt
import pandas as pd 

corona = pd.read_csv("Data/alteredData.csv")

#Print dataset basic information (column names, dtype etc.)
corona.info()
#Print basic stats of the dataset
description = corona.groupby("Entity").describe()

a = []
for i in description.columns:
    if i[0] not in a:
        a.append(i[0])
print(a)

for i in a :
    print(f"\n{i} TABLE \n")
    print(description[i])