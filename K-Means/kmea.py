import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Customers.csv')
x =dataset.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
#with in clusters sum of squares
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
kmeans=KMeans(n_clusters=5,init= 'k-means++',random_state=0)
ykmeans=kmeans.fit_predict(x)

plt.scatter(x[ykmeans==0,0],x[ykmeans==0,1],s = 100, color = 'red',label = 'Cluster1')
plt.scatter(x[ykmeans==1,0],x[ykmeans==1,1],s = 100, color = 'green',label = 'Cluster2')
plt.scatter(x[ykmeans==2,0],x[ykmeans==2,1],s = 100, color = 'cyan',label = 'Cluster3')
plt.scatter(x[ykmeans==3,0],x[ykmeans==3,1],s = 100, color = 'blue',label = 'Cluster4')
plt.scatter(x[ykmeans==4,0],x[ykmeans==4,1],s = 100, color = 'magenta',label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c ='yellow',label='Centroids')
plt.title('Clusters OF Customers')
plt.xlabel('Annual Income($)')
plt.ylabel('Spending Score(1-100)')
plt.show()