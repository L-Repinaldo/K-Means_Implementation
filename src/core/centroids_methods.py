
import numpy as np

def initialize_centroids(data, k):

    random_idx = np.random.choice(data.shape[0], size=k, replace=False)
    centroids = data[random_idx]

    return centroids



def calculate_distances(data, centroids):

    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(data - centroid, axis=1)

    return distances
