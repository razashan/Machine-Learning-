import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Iris.csv')
x = dataset.iloc[:, [1, 2, 3, 4]].values
kmeans = KMeans(n_clusters= 3, init = 'k-means++', max_iter = 300, n_init=10,
                random_state=0)
kmeans.fit(x)
var=kmeans.labels_
print(var)
dataset = pd.read_csv('Iris_test.csv')
x = dataset.iloc[:, [1, 2, 3, 4]].values
y_kmeans = kmeans.predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans== 0,1] , s=100, c = 'red' ,label='Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans== 1,1] , s=100, c = 'blue' ,label='Iris-versicolur')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans== 2,1] , s=100, c = 'green' ,label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c= 'yellow' ,label ='Centroids')
plt.legend()
plt.show()
