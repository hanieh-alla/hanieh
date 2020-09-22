
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
customer_data = pd.read_csv('/Users/haniehalla/Desktop/ex_07_02/shopping-data.csv')
customer_data.shape
customer_data.head()

data = customer_data.iloc[:, 3:5].values
plt.figure(figsize=(10, 7))
plt.title("nemoodar derakhti moshtariyan foroshgah")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title("khoshe bandi nahayi")
plt.show()








