import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class LogisticRegression:
    def __init__(self, epochs=1, learning_rate=0.1):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        """
        Train Logistic Regression
        :param X: features
        :param y: targets
        :return: None
        """

        # Add a bias column to X
        Xb = self.add_bias(X)

        # Initialize weights according to a uniform distribution
        w = self.init_weights(Xb)

        for _ in range(self.epochs):

            for i, x in enumerate(Xb):

                # Compute dot product between weight vector and features
                y_pred = self.activation(np.dot(w, x))

                # If we made a mistake, update weights
                if y_pred != y[i]:
                    w += self.lr * (y[i] - y_pred) * x

        self.weights = w

    @staticmethod
    def activation(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def add_bias(X):
        """
        Add a bias column (value = 1) to X
        :param X: original data
        :return: new data
        """
        bias = np.ones((X.shape[0], 1))
        return np.hstack((X, bias))

    @staticmethod
    def init_weights(X):
        """
        Initialize weights according to a uniform distribution
        :param X: feature set (w/ bias)
        :return: weight vector
        """
        bounds = 1.0 / np.sqrt(X.shape[0])
        return np.random.default_rng().uniform(-bounds, bounds, size=X.shape[1])

    def predict(self, X):
        Xb = self.add_bias(X)
        return np.round(self.activation(Xb.dot(self.weights)))


if __name__ == "__main__":
    X = np.linspace(-10, 10)
    t = 1.0 / (1.0 + np.exp(-X))

    X = np.column_stack((X, t))
    y = np.where(X[:, 0] > 0, 1, 0)


    lr = LogisticRegression(epochs=1000)
    lr.fit(X, y)
    preds = lr.predict(X)

