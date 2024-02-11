import numpy as np

from utils.calculations import minkowski_distance


class KNN:
    def __init__(self, k, p=2):
        self.k = k
        self.p = p
        self.features, self.classes = None, None

    def fit(self, features, classes):
        """
        Fits training data to classifier
        :param X_train:
        :param classes:
        :return:
        """
        self.features = features
        self.classes = classes

    def predict(self, features):
        """
        Predicts the labels of the test dataset according to their K-Nearest Neighbors
        :param X_test: test dataset
        :return: List of predicted labels
        """

        if not (self.classes and self.features):
            raise ValueError("Call fit() first.")

        y_pred = []

        for sample in features:
            # Compute distances from each test sample to every training example
            distances = [minkowski_distance(sample, train_i, p=self.p) for train_i in self.X_train]

            # Compute the k-nearest neighbors according to previously calculated distances
            closest = [self.y_train[idx] for idx in np.argsort(distances)[:self.k]]

            # Attribute the label of the test sample according to the most common label of its neighbors
            y_pred.append(np.argmax(np.bincount(closest)))

        return y_pred
