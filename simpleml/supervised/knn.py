import numpy as np
from utils.calculations import minkowski_distance


class KNN:
    def __init__(self: "KNN", k: int, p: int = 2) -> None:
        self.k: int = k
        self.p: int = p
        self.features: np.ndarray = None
        self.classes: np.ndarray = None

    def fit(self: "KNN", features: np.ndarray, classes: np.ndarray) -> None:
        """
        Fits training data to classifier
        :param X_train:
        :param classes:
        :return:
        """
        self.features: np.ndarray = features
        self.classes: np.ndarray = classes

    def predict(self: "KNN", features: np.ndarray) -> list[int]:
        """
        Predicts the labels of the test dataset according to their K-Nearest Neighbors
        :param X_test: test dataset
        :return: List of predicted labels
        """

        if not (self.classes and self.features):
            raise ValueError("Call fit() first.")

        y_pred: list[int] = []

        for sample in features:
            # Compute distances from each test sample to every training example
            distances: list[float] = [minkowski_distance(sample, train_i, p=self.p) for train_i in self.X_train]

            # Compute the k-nearest neighbors according to previously calculated distances
            closest: list[int] = [self.y_train[idx] for idx in np.argsort(distances)[: self.k]]

            # Attribute the label of the test sample according to the most common label of its neighbors
            y_pred.append(np.argmax(np.bincount(closest)))

        return y_pred
