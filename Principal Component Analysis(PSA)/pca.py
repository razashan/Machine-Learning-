import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head(5)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scalar = StandardScaler()
scalar.fit(df)
scalad_data =scalar.transform(df)
print(scalad_data)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scalad_data)
x_pca = pca.transform(scalad_data)
print("\n",scalad_data.shape)
print(" ------------------")
print(x_pca.shape)
print(scalad_data)
print(x_pca)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Principle Component Analysis')
plt.ylabel('Second Principle Component')
plt.show()