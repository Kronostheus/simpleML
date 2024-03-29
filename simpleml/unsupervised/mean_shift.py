import numpy as np
from tqdm import tqdm
from utils.calculations import minkowski_distance


class MeanShift:
    def __init__(self: "MeanShift", bandwidth: int, max_iter: int = 300) -> None:
        self.bandwidth: int = bandwidth
        self.max_iter: int = max_iter
        self.centroids: np.ndarray = None
        self.labels: np.ndarray = None

    def fit(self: "MeanShift", samples: np.ndarray) -> np.ndarray:
        """
        Mean Shift clustering algorithm using a flat kernel
        :param samples: data samples to cluster
        :return: labels associated with cluster centers found for input data samples
        """

        # Initialize every sample as a cluster center
        centroids: np.ndarray = samples

        for _ in tqdm(range(self.max_iter)):

            new_centroids: np.ndarray = centroids.copy()

            centroid: np.ndarray
            for centroid in centroids:

                # Check all samples within a certain radius (flat kernel)
                in_bandwidth: dict[int, np.ndarray] = {
                    idx: sample
                    for idx, sample in enumerate(samples)
                    if minkowski_distance(sample, centroid, p=2) < self.bandwidth
                }

                # Compute the mean of the samples found as their cluster center
                new_centroid: np.ndarray = np.mean(list(in_bandwidth.values()), axis=0)

                # Modify centroid array
                idx: int
                for idx in in_bandwidth:
                    new_centroids[idx] = new_centroid

            # If cluster centers (centroids) have not shifted, they have converged
            if np.array_equal(centroids, new_centroids):
                break

            centroids = new_centroids

        self.centroids, self.labels = np.unique(centroids, return_inverse=True, axis=0)

        return self.labels

    def predict(self: "MeanShift", samples: np.ndarray) -> list[int]:
        """
        Computes Euclidean distance towards every cluster center and returns label associated with the nearest center
        :param samples: data samples to test
        :return: List of cluster labels
        """
        return [
            np.argmin([minkowski_distance(centroid, sample, p=2) for centroid in self.centroids]) for sample in samples
        ]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(centers=3, n_samples=100, random_state=1)

    meanshift = MeanShift(bandwidth=3)
    classes = meanshift.fit(X)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.scatter(X[:, 0], X[:, 1], c=classes, cmap="Set3", alpha=0.5)
    ax2.scatter(meanshift.centroids[:, 0], meanshift.centroids[:, 1], color="black", marker=".")

    ax2.set_xlabel("$x_0$")
    ax2.set_ylabel("$x_1$")
    plt.show()
