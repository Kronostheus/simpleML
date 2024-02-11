import matplotlib.pyplot as plt
from utils.calculations import minkowski_distance
from sklearn.datasets import make_moons, make_circles


class DBSCAN:
    def __init__(self, eps=0.2, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.cluster_counter = 0
        self.samples = None

    def fit(self, samples):
        """
        Fits data samples to object
        :param samples: data samples
        :return: None
        """
        self.samples = samples

    def predict(self):
        """
        Run DBSCAN algorithm
        :return: cluster labels
        """

        # Dictionary with corresponding cluster labels of fitted data samples
        self.labels = {i: None for i, _ in enumerate(self.samples)}

        for idx, sample in enumerate(self.samples):

            # Sample already visited
            if self.labels[idx]:
                continue

            # Find neighbors
            neighbors = self.range_query(idx)

            # A point without sufficient neighbors is an outlier and marked as Noise (-1)
            if len(neighbors) >= self.min_samples:
                # Found cluster
                self.expand(idx, neighbors)
                self.cluster_counter += 1
            else:
                self.labels[idx] = -1

        return list(self.labels.values())

    def expand(self, sample_index, sample_neighbors):
        """
        Expand a cluster by iterating through a point's neighbors
        :param sample_index: focus point
        :param sample_neighbors: focus point's neighbors
        :return: None
        """
        # This point was found within the neighborhood of another, thus belonging to the same cluster
        if not self.labels[sample_index]:
            self.labels[sample_index] = self.cluster_counter

        # Iterate through each neighbor
        for n_index in sample_neighbors:

            # Get neighbors of neighbor
            new_neighbors = self.range_query(n_index)

            # Check if neighbor has enough neighbors (expanded neighbors)
            if len(new_neighbors) >= self.min_samples:

                for nb in new_neighbors:
                    # If expanded neighbor is not a part of a cluster
                    if not self.labels[nb] or self.labels[nb] == -1:

                        # If expanded neighbor is not a direct neighbor of focus point then add it to list
                        if not self.labels[nb] and nb not in sample_neighbors:
                            sample_neighbors.append(nb)

                        # Attribute cluster
                        self.labels[nb] = self.cluster_counter

    def range_query(self, query_index):
        """
        Use euclidean distance to return all neighbors of a point (query_index) that are within a certain proximity.
        :param query_index: focus point
        :return: surrounding neighbors
        """
        return [i for i, _ in enumerate(self.samples)
                if i != query_index and minkowski_distance(self.samples[query_index], self.samples[i]) <= self.eps]


X, _ = make_moons(300, noise=0.05)

dbscan = DBSCAN()
dbscan.fit(X)
labels = dbscan.predict()

group_colors = ['lightgreen', 'coral', 'skyblue']
colors = [group_colors[j] for j in labels]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.4))
ax1.scatter(X[:, 0], X[:, 1])
ax2.scatter(X[:, 0], X[:, 1], color=colors)
plt.show()

Z, _ = make_circles(300, noise=0.05, factor=0.5)

dbscan2 = DBSCAN()
dbscan2.fit(Z)
_labels = dbscan2.predict()

colors = [group_colors[j] for j in _labels]

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(6.4, 6.4))
ax3.scatter(Z[:, 0], Z[:, 1])
ax4.scatter(Z[:, 0], Z[:, 1], color=colors)
plt.show()
