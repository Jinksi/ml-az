# K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
dataset

X = dataset.iloc[:, [3,4]].values

# find the optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    # creating a kmeans model for i
    # use k-means++ to avoid random initialisation trap
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    # inertia == wcss
    wcss.append(kmeans.inertia_)

# plot the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# We've decided to use 5 clusters based on the Elbow Method graph
# Apply the k-means to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
y_kmeans
X

def visualise():
    # Visualise the clusters
    colors = ['tomato', 'orange', 'skyblue', 'hotpink', 'limegreen']
    cluster_titles = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']
    for i in range(0, 5):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans==i, 1], s=10, c=colors[i], label=cluster_titles[i])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='yellow', label='Centroids')
    plt.title('Clusters of Clients')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    return plt.show()

visualise()
