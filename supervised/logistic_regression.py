import numpy as np


class LogisticRegression:
    def __init__(self, epochs=1, learning_rate=0.1):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, features, y):
        """
        Train Logistic Regression
        :param features: features
        :param y: targets
        :return: None
        """

        # Add a bias column to features
        features = self.add_bias(features)

        # Initialize weights according to a uniform distribution
        w = self.init_weights(features)

        for _ in range(self.epochs):

            for i, x in enumerate(features):

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
    def add_bias(features):
        """
        Add a bias column (value = 1) to features
        :param features: original data
        :return: new data
        """
        bias = np.ones((features.shape[0], 1))
        return np.hstack((features, bias))

    @staticmethod
    def init_weights(features):
        """
        Initialize weights according to a uniform distribution
        :param features: feature set (w/ bias)
        :return: weight vector
        """
        bounds = 1.0 / np.sqrt(features.shape[0])
        return np.random.default_rng().uniform(-bounds, bounds, size=features.shape[1])

    def predict(self, features):
        features = self.add_bias(features)
        return np.round(self.activation(features.dot(self.weights)))


if __name__ == "__main__":
    features = np.linspace(-10, 10)
    t = 1.0 / (1.0 + np.exp(-features))

    features = np.column_stack((features, t))
    y = np.where(features[:, 0] > 0, 1, 0)


    lr = LogisticRegression(epochs=1000)
    lr.fit(features, y)
    preds = lr.predict(features)

