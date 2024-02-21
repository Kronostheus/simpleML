import numpy as np
from utils.calculations import minkowski_distance


class DBSCAN:
    def __init__(self: "DBSCAN", eps: float = 0.2, min_samples: int = 5) -> None:
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.labels: dict[int, int] = None
        self.cluster_counter: int = 0
        self.samples: int = None

    def fit(self: "DBSCAN", samples: np.ndarray) -> None:
        """
        Fits data samples to object
        :param samples: data samples
        :return: None
        """
        self.samples = samples

    def predict(self: "DBSCAN") -> list[int]:
        """
        Run DBSCAN algorithm
        :return: cluster labels
        """

        # Dictionary with corresponding cluster labels of fitted data samples
        self.labels = {i: None for i, _ in enumerate(self.samples)}

        idx: int
        for idx, _ in enumerate(self.samples):

            # Sample already visited
            if self.labels[idx]:
                continue

            # Find neighbors
            neighbors: list[int] = self.range_query(idx)

            # A point without sufficient neighbors is an outlier and marked as Noise (-1)
            if len(neighbors) >= self.min_samples:
                # Found cluster
                self.expand(idx, neighbors)
                self.cluster_counter += 1
            else:
                self.labels[idx] = -1

        return list(self.labels.values())

    def expand(self: "DBSCAN", sample_index: int, sample_neighbors: list[int]) -> None:
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
        n_index: int
        nb: int
        for n_index in sample_neighbors:

            # Get neighbors of neighbor
            new_neighbors: list[int] = self.range_query(n_index)

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

    def range_query(self: "DBSCAN", query_index: int) -> list[int]:
        """
        Use euclidean distance to return all neighbors of a point (query_index) that are within a certain proximity.
        :param query_index: focus point
        :return: surrounding neighbors
        """
        return [
            i
            for i, _ in enumerate(self.samples)
            if i != query_index and minkowski_distance(self.samples[query_index], self.samples[i]) <= self.eps
        ]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles, make_moons

    X, _ = make_moons(300, noise=0.05)

    dbscan = DBSCAN()
    dbscan.fit(X)
    labels = dbscan.predict()

    group_colors = ["lightgreen", "coral", "skyblue"]
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
