
from .core import centroids_methods

import numpy as np

class KMeans:

    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, data):

        data = np.asarray(data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.centroids = centroids_methods.initialize_centroids(data, self.n_clusters)

        for _ in range(self.max_iter):
            distances = centroids_methods.calculate_distances(data, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = data[labels == i]
                if len(cluster_points) == 0:
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            new_centroids = np.asarray(new_centroids)

            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        self.centroids_ = self.centroids
        self.labels_ = np.argmin(centroids_methods.calculate_distances(data, self.centroids), axis=1)

        
        self.clusters_ = [ data[self.labels_ == i] for i in range(self.n_clusters) ]
        
        return self