import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv('data/Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values

# elbow method for finding the optimal number

wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('elbow method for optimal number of cluster')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()


# apply k-means on data

k_means = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300)
y_k_means = k_means.fit_predict(X)

# visualising
plt.scatter(X[y_k_means == 0, 0], X[y_k_means == 0, 1], s=100, c='red', label='cluster_1')
plt.scatter(X[y_k_means == 1, 0], X[y_k_means == 1, 1], s=100, c='blue', label='cluster_2')
plt.scatter(X[y_k_means == 2, 0], X[y_k_means == 2, 1], s=100, c='green', label='cluster_3')
plt.scatter(X[y_k_means == 3, 0], X[y_k_means == 3, 1], s=100, c='cyan', label='cluster_4')
plt.scatter(X[y_k_means == 4, 0], X[y_k_means == 4, 1], s=100, c='magenta', label='cluster_5')

plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, c='yellow', label='cluster center')
plt.title('cluster of client')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()






