import numpy as np


class PCA:
    """
    Principle Component Analysis (PCA)
    """

    def __init__(self: "PCA", n_components: float) -> None:
        self.num_components: float = n_components
        self.cov_matrix: np.ndarray = None
        self.explained_variance: np.ndarray = None
        self.components = None
        self.explained_variance_ratio = None

    @staticmethod
    def _mean_data(x: np.ndarray) -> np.ndarray:
        """
        Center data by subtracting each column with its mean
        :param x: raw data
        :return: centered data
        """

        return x - np.mean(x, axis=0)

    def fit(self: "PCA", x: np.ndarray) -> None:
        """
        Fit the model with data
        :param x: data points
        :return: None
        """

        # Center data as PCA is vulnerable to variance.
        x: np.ndarray = self._mean_data(x)

        # Build covariance matrix
        self.cov_matrix = np.cov(x.T)

        # There are multiple ways get components, such as np.linalg.svd
        eig_values: np.ndarray
        eig_vectors: np.ndarray
        eig_values, eig_vectors = np.linalg.eigh(self.cov_matrix)

        # np.linalg.eigh returns everything in ascending
        eig_values, eig_vectors = eig_values[::-1], eig_vectors[:, ::-1]

        # Explained variance == eigenvalues, the ratio [0, 1] can be obtained by dividing them by their total sum
        explained_variance_ratio: np.ndarray = eig_values / np.sum(eig_values)

        threshold: int
        if 0 < self.num_components < 1:
            # Cumulative sum (np.cumsum) throughout the list, although we could also use a while loop to stop earlier
            threshold = np.flatnonzero(np.cumsum(explained_variance_ratio) >= self.num_components)[0] + 1
        elif self.num_components >= 1 and isinstance(self.num_components, int):
            # Hard-code how many components we want returned
            threshold = self.num_components
        else:
            raise ValueError("num_components must be a positive int >= 1 or float between 0 and 1.")

        # Only save desired number of elements
        self.explained_variance = eig_values[:threshold]
        self.components = eig_vectors[:, :threshold].T
        self.explained_variance_ratio = explained_variance_ratio[:threshold]

    def fit_transform(self: "PCA", x: np.ndarray) -> np.ndarray:
        """
        Fit the model with data and apply the dimensionality reduction on it.

        :param x: raw data
        :return:
        """

        # Fit model
        self.fit(x)

        # Dimensionality reduction
        return self.transform(x)

    def transform(self: "PCA", x: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction on data.
        Assumes input needs to be centered, but that model has already been fitted.
        :param x:
        :return:
        """

        # Center data
        x: np.ndarray = self._mean_data(x)

        # Dimensionality reduction using principal components
        return np.dot(self.components, x.T).transpose()


def sklearn_svd_flip(matrix: np.ndarray, components: int) -> np.ndarray:
    """
    Sklearn pulled a sneaky and returns different components than my implementation.
    After some digging, I found that while both should be mathematically correct, sklearn flips eigenvectors' sign
    in order to enforce deterministic output.
    To ensure that this implementation was correct during testing, I apply the same flip as sklearn.

    :param matrix:
    :param components:
    :return:
    """

    # This is how the eigenvectors and eigenvalues are computed in my implementation
    eig_vectors: np.ndarray
    _, eig_vectors = np.linalg.eigh(matrix)
    eig_vectors = eig_vectors[:, ::-1]

    # The eigenvector flip is done in these next lines
    max_abs_cols: np.ndarray = np.argmax(np.abs(eig_vectors), axis=0)
    signs: np.ndarray = np.sign(eig_vectors[max_abs_cols, range(eig_vectors.shape[1])])
    eig_vectors *= signs

    # Return same number of vectors/components
    return eig_vectors[:, :components].T


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA as SklearnPCA  # noqa: N811 -> PCA not constant

    X = load_iris().data
    n_comp = 2

    sklearn_pca = SklearnPCA(n_components=n_comp)
    Xt_sklearn = sklearn_pca.fit_transform(X)

    pca = PCA(n_components=n_comp)
    Xt_pca = pca.fit_transform(X)

    if not (
        np.allclose(pca.explained_variance_ratio, sklearn_pca.explained_variance_ratio_)
        and np.allclose(pca.explained_variance, sklearn_pca.explained_variance_)
        # Components are not directly comparable due to sklearn's eigenvector flip
        and np.allclose(sklearn_svd_flip(pca.cov_matrix, n_comp), sklearn_pca.components_)
    ):
        raise Exception("Algorithms do not match")
