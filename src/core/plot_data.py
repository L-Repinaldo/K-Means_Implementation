import matplotlib.pyplot as plt
import numpy as np


def plot(data_array, centroids=None, labels=None, title="Data Plot"):

    data_array = np.asarray(data_array)

    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(
            data_array[:, 0],
            data_array[:, 1],
            label="Data Points",
            alpha=0.5
        )

    else:
        labels = np.asarray(labels)

        for cluster_id in np.unique(labels):
            cluster_points = data_array[labels == cluster_id]

            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.5
            )

    if centroids is not None:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='X',
            s=200,
            label='Centroids'
        )

    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()