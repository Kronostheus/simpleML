import numpy as np
import matplotlib.pyplot as plt
from utils.calculations import minkowski_distance
from sklearn.datasets import make_blobs


class KMeans:
    def __init__(self, k, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def _get_clusters(self, centroids, samples):
        """
        Based on current centroids, assign a cluster to each data sample and recalculate centroids based on new clusters
        :param centroids: current centroids
        :param samples: data samples to cluster
        :return: new centroids and cluster labels
        """

        clusters = [[] for _ in range(self.k)]
        labels = []

        for sample in samples:
            # Pairwise distance between sample and all current centroids
            closest = np.argmin([minkowski_distance(centroid, sample, p=2) for centroid in centroids])

            # Place sample into respective cluster
            clusters[int(closest)].append(sample)

            # Assign cluster label
            labels.append(int(closest))

        return self._get_centroids(clusters), labels

    def _get_centroids(self, clusters):
        """
        Get centroids of each cluster by computing their respective mean
        :param clusters: current clusters
        :return: current centroids
        """
        return [np.array(c).mean(axis=0) for c in clusters]

    def fit(self, samples):
        """
        Run K-Means Algorithm on data samples in order to partition them into K clusters
        :param samples: data samples to partition
        :return: cluster labels assigned to data samples
        """

        labels = None

        # Initialize clusters using Random Partition method; data split into k clusters and centroids are computed
        centroids = self._get_centroids(np.array_split(samples, self.k))

        for _ in range(self.max_iter):
            prev_centroids = centroids

            # Update step
            centroids, labels = self._get_clusters(centroids, samples)

            # Check if centroids changed, stop if they are equal
            if np.allclose(prev_centroids, centroids):
                break

        self.centroids = np.array(centroids)

        return labels

    def predict(self, samples):
        """
        Predicts unseen data samples by simply assigning the cluster label of the closest centroid.
        Does not alter clusters.
        :param samples: unseen data samples
        :return: cluster labels associated with data samples
        """
        assert self.centroids
        return self._get_clusters(self.centroids, samples)[1]


if __name__ == "__main__":
    # Generate dataset
    X, y = make_blobs(centers=3, n_samples=500, random_state=1)

    # Visualize
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.show()

    # Initialize KMeans with 3 clusters
    kmeans = KMeans(3)
    classes = kmeans.fit(X)

    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in classes]

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    ax2.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
    ax2.set_xlabel('$x_0$')
    ax2.set_ylabel('$x_1$')
    plt.show()
