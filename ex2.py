from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans as kmeans
import seaborn as sns

#Function for kmeans clustering
def clustering(cluster_data, clusters, result):
    train = data[cluster_data].to_numpy()
    k = kmeans(n_clusters = clusters)
    k.fit(train)
    predict = lambda X: k.predict(X)
    clusters = predict(train)
    result['Cluster'] = clusters
    print(result)

corona = pd.read_csv("Data/alteredData.csv")

#Clustering 
data = corona
#!! maybe scale the data? !!
data = data.groupby('Entity').max()
data['Deaths/Cases'] = data['Deaths']/data['Cases']
data['Cases/Population'] = data['Cases']/data['Population']

result1 = result2 = result3 =  data

#Deaths/Cases based Clustering
print('Deaths/Cases Clustering')
clustering(['Deaths/Cases'], 5, result1)

#Cases/Population based Clustering
print('Cases/Population Clustering')
clustering(['Cases/Population'], 5, result2)

#Deaths/Cases and Cases/Population based Clustering
print('Deaths/Cases', 'Cases/Population Clustering')
clustering(['Deaths/Cases', 'Cases/Population'], 5, result3)

#Heatmap for Pairwise Correlation
corr = result3.corr().round(2)
plt.figure("Pairwise Correlation",figsize=(10,10))
sns.heatmap(corr, annot= True)
plt.tight_layout()
plt.show()

#Plots for each clustering
fig, axs = plt.subplots(ncols=3, figsize=(12,7))
fig.canvas.manager.set_window_title("Clustering Plots")
sns.scatterplot(x='Deaths', y='Cases', hue='Cluster', data=result1, ax=axs[0])
sns.scatterplot(x='Cases', y='Population', hue='Cluster', data=result2, ax=axs[1])
sns.scatterplot(x='Deaths/Cases', y='Cases/Population', hue='Cluster', data=result3, ax=axs[2])
plt.tight_layout()
plt.show()