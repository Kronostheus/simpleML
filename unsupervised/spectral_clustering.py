import numpy as np
import scipy as sp

from unsupervised.kmeans import KMeans
from utils.calculations import minkowski_distance


class Spectral:
    def __init__(self: "Spectral", n_clusters: int, neighbor: int = 7) -> None:
        self.k: int = n_clusters
        self.kth_neighbor: int = neighbor - 1
        self.samples: np.ndarray = None
        self.samples_len: int = None

    def _precompute_neighbors(self: "Spectral") -> list[float]:
        """
        For all samples compute the mean of the distances between it and its K nearest neighbors,
         using Euclidean distance. This is equivalent to the sigma for that sample, to be used in the affinity function.
        :return: List of sigmas
        """
        return [
            np.mean(
                np.sort([
                    minkowski_distance(sample, other, p=2)
                    for other in self.samples if not np.array_equal(sample, other)
                ])[:self.kth_neighbor]
            )
            for sample in self.samples
        ]

    def _affinity_matrix(self: "Spectral") -> np.ndarray:
        """
        Compute affinity matrix, using Local Scaling self-tuning of the scaling parameters
        :return: Affinity Matrix
        """
        # Scaling parameters. Sigma of sample i is the mean of the distances to the K-nearest neighbors.
        scaling_dict: list[float] = self._precompute_neighbors()

        # Diagonal of affinity matrix is 0
        affinity_matrix: np.ndarray = np.zeros(shape=(self.samples_len, self.samples_len))

        # Distance function used is the Euclidean Distance (Minkowski with p=2)
        i: int
        j: int
        sample_i: np.ndarray
        sample_j: np.ndarray
        for i, sample_i in enumerate(self.samples):
            for j, sample_j in enumerate(self.samples):
                if i != j:
                    # A = exp(-d(i, j)^2 / (sigma_i * sigma_j)) where sigma_x = mean(d(x, n) for n closest neighbors)
                    affinity_matrix[i][j] = np.exp(
                        -(minkowski_distance(sample_i, sample_j, p=2) ** 2) / (scaling_dict[i] * scaling_dict[j])
                    )

        return affinity_matrix

    def fit(self: "Spectral", samples: np.ndarray) -> list[int]:
        """
        Apply Spectral Clustering algorithm as described in Ng et al. 2002
        Affinity matrix calculated with scaling parameter as described in Zelnik-Manor et al. 2005
        Clustering algorithm used: custom simple KMeans
        :param samples: data samples to cluster
        :return: labels associated with cluster
        """
        self.samples = samples
        self.samples_len = len(samples)

        # Compute affinity matrix (A)
        affinity: np.ndarray = self._affinity_matrix()

        # Square root of diagonal matrix (D) composed of the sum of each of A's rows => D^1/2
        d: np.ndarray = np.diag(np.power(np.sum(affinity, axis=0), -1/2))

        # Compute laplacian matrix (L) using formula L = D^1/2 . A . D^1/2
        laplacian: np.ndarray = d @ affinity @ d

        # Eigenvectors of L stacked as a matrix (X) -> pretty sure I had to look this up
        eig_vecs: np.ndarray
        _, eig_vecs = sp.sparse.linalg.eigs(laplacian, k=self.k)

        # Normalize X using formula X / sum(X^2)^1/2 which gives us a data sample representation (Y)
        normalized_eig_vecs: np.ndarray = eig_vecs / np.linalg.norm(eig_vecs, axis=1, keepdims=True)

        # Fit a KMeans algorithm to Y and receive cluster labels
        kmeans: KMeans = KMeans(k=self.k)

        return kmeans.fit(normalized_eig_vecs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    # Generate dataset
    X, _ = make_moons(300, noise=0.05)

    spectral = Spectral(n_clusters=2)
    clusters = spectral.fit(X)

    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in clusters]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    ax.set_title('Spectral Clustering')
    plt.show()
