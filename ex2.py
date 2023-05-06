from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans as kmeans
import seaborn as sns


corona = pd.read_csv("Data/alteredData.csv")

#Clustering 
data = corona
data = data.groupby('Entity').max()
data['Deaths/Cases'] = data['Deaths']/data['Cases']
data['Cases/Population'] = data['Cases']/data['Population']

result1 = data
result2 = data
result3 = data

#Deaths/Cases based Clustering
train1 = data[['Deaths/Cases']].to_numpy()
k1 = kmeans(n_clusters=5)
k1.fit(train1)
predict = lambda X: k1.predict(X)
clusters = predict(train1)
result1['Cluster'] = clusters
print("Deaths/Cases Clustering")
print(result1)

#Cases/Population based Clustering
train2 = data[['Cases/Population']].to_numpy()
k2 = kmeans(n_clusters=5)
k2.fit(train2)
predict = lambda X: k2.predict(X)
clusters = predict(train2)
result2['Cluster'] = clusters
print("Cases/Population Clustering")
print(result2)

#Deaths/Cases and Cases/Population based Clustering
train3 = data[['Deaths/Cases','Cases/Population']].to_numpy()
k3 = kmeans(n_clusters=5)
k3.fit(train3)
predict = lambda X: k3.predict(X)
clusters = predict(train3)
result3['Cluster'] = clusters
print("Deaths/Cases and Cases/Population Clustering")
print(result3)


corr = result3.corr().round(2)
plt.figure("Pairwise Correlation",figsize=(10,10))
sns.heatmap(corr, annot= True)
plt.tight_layout()
plt.show()