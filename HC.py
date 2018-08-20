import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
data = pd.read_csv('data/Mall_Customers.csv')
X = data.iloc[:, [2, 3]].values

# find optimal number of cluster
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distance')
plt.show()

# fit hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X)
# visualising
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='cluster_1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='cluster_2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='cluster_3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='cluster_4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='cluster_5')

# plt.scatter(hc.cluster_centers_[:, 0], hc.cluster_centers_[:, 1], s=300, c='yellow', label='cluster center')
plt.title('cluster of client')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
