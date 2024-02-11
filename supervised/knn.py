import numpy as np

from utils.calculations import minkowski_distance


class KNN:
    def __init__(self, k, p=2):
        self.k = k
        self.p = p
        self.X_train, self.y_train = None, None

    def fit(self, X_train, y_train):
        """
        Fits training data to classifier
        :param X_train:
        :param y_train:
        :return:
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predicts the labels of the test dataset according to their K-Nearest Neighbors
        :param X_test: test dataset
        :return: List of predicted labels
        """

        assert self.X_train and self.y_train

        y_pred = []

        for sample in X_test:
            # Compute distances from each test sample to every training example
            distances = [minkowski_distance(sample, train_i, p=self.p) for train_i in self.X_train]

            # Compute the k-nearest neighbors according to previously calculated distances
            closest = [self.y_train[idx] for idx in np.argsort(distances)[:self.k]]

            # Attribute the label of the test sample according to the most common label of its neighbors
            y_pred.append(np.argmax(np.bincount(closest)))

        return y_pred
