from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns

corona = pd.read_csv("Data/data.csv")

#To check NaN values use:
#print(corona.isnull().sum().sort_values(ascending=False))

#Rename
corona.rename(columns={"Daily tests": "Tests"}, inplace=True)

#Fill NaN values
#print(corona.isnull().sum().sort_values(ascending=False))
corona.Deaths = corona.groupby("Entity").Deaths.transform(lambda x: x.fillna(method="bfill"))
corona.Cases  = corona.groupby("Entity").Cases.transform(lambda x: x.fillna(method="bfill"))
corona.Tests  = corona.groupby("Entity").Tests.transform(lambda x: x.fillna(method="bfill"))
#bfill = next valid, doesnt fill cases where there is no next valid so we also do the opposite
corona.Deaths = corona.groupby("Entity").Deaths.transform(lambda x: x.fillna(method="ffill"))
corona.Cases  = corona.groupby("Entity").Cases.transform(lambda x: x.fillna(method="ffill"))
corona.Tests  = corona.groupby("Entity").Tests.transform(lambda x: x.fillna(method="ffill"))

#Drop longitude and latitude
corona.drop(columns=["Longitude","Latitude"],inplace=True)

#export csv

corona.to_csv("Data/alteredData.csv",index=False)

#Calculate pairwise correlation of data
corr = corona.corr().round(2)

useful_pairs = []
threshfold = 0.7

for r in range(len(corr.values)):
    for c in range(r):
        if corr.values[r][c] > threshfold: 
            useful_pairs.append([corr.columns[r],corr.columns[c]])

print("\nPrint usefull pairs:\n")
for pair in useful_pairs:
    print(pair[0],"and",pair[1])       

#Plot correlation heatmap
plt.figure("Pairwise Correlation",figsize=(10,10))
sns.heatmap(corr, annot= True)
plt.tight_layout()
plt.show()

#Necessary data for the plots
y_val = []
result = {x: [corona.groupby(['Continent'])['Cases'].max()[x].round(1), corona.groupby(['Continent'])['Tests'].sum()[x].round(1), corona.groupby(['Continent'])['Deaths'].max()[x].round(1)] for x in corona['Continent'].unique()}
result = dict(sorted(result.items(), key=lambda item: item[1]))

#Data calculation for the plots
for i in result.keys():
    y_val.append([i, result[i][0].round(1) / result[i][1].round(1) , result[i][2].round(1) / result[i][1].round(1), result[i][2].round(1) / result[i][0].round(1)])

df = pd.DataFrame(y_val, columns=['Continent', 'Cases/Tests', 'Deaths/Tests', 'Deaths/Cases'])

#Plots
fig, axs = plt.subplots(ncols=3, figsize=(20,8))
fig.canvas.manager.set_window_title("Essential plots")
sns.barplot(x='Continent', y='Cases/Tests', data=df, ax=axs[0])
sns.barplot(x='Continent', y='Deaths/Tests', data=df, ax=axs[1])
sns.barplot(x='Continent', y='Deaths/Cases', data=df, ax=axs[2])
plt.tight_layout()
plt.show()