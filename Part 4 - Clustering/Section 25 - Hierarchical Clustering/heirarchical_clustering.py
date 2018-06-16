# Heirarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

# Using dendogram to find optimal N clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# We have decided to use 5 clusters based on analysing Dendrogram
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_hc = hc.fit_predict(X)

np.unique(y_hc)

def visualise():
    colors = ['tomato', 'orange', 'skyblue', 'green', 'hotpink']
    cluster_labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']
    for i in np.unique(y_hc): # for each cluster
        plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=10, c=colors[i], label=cluster_labels[i])
    plt.title('Cluster of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    return plt.show()

visualise()
