import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

#Drop unnecessary columns
to_drop = ['Latitude','Longitude', 'Average temperature per year', 'GDP/Capita']
data.drop(to_drop, inplace=True, axis=1)


#Check if any data is missing from the DataFrame
#print(data.isnull().sum().sort_values(ascending=False))

#Fill data that was missing based on the Country
data.Deaths = data.groupby('Entity').Deaths.transform(lambda x: x.fillna(x.mean()))
data.Cases = data.groupby('Entity').Cases.transform(lambda x: x.fillna(x.mean()))
data['Daily tests'] = data['Daily tests'].fillna(0)

result = {x: data.groupby(['Continent'])['Deaths'].mean()[x] for x in data['Continent'].unique()}
result = dict(sorted(result.items(), key=lambda item: item[1]))

plt.figure('Deaths over Continents')
sns.lineplot(x=list(result.keys()), y=list(result.values()))
sns.barplot(x=list(result.keys()), y=list(result.values()))
plt.title('Average Deaths over Continents')
plt.show()
#plt.legend()
#plt.title(label='Deaths over Continents')
