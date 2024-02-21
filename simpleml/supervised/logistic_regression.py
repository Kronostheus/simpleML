import numpy as np


class LogisticRegression:
    def __init__(self: "LogisticRegression", epochs: int = 1, learning_rate: float = 0.1) -> None:
        self.lr: float = learning_rate
        self.epochs: int = epochs
        self.weights: np.ndarray = None

    def fit(self: "LogisticRegression", features: np.ndarray, y: np.ndarray) -> None:
        """
        Train Logistic Regression
        :param features: features
        :param y: targets
        :return: None
        """

        # Add a bias column to features
        features: np.ndarray = self.add_bias(features)

        # Initialize weights according to a uniform distribution
        w: np.ndarray = self.init_weights(features)

        i: int
        x: np.ndarray

        for _ in range(self.epochs):

            for i, x in enumerate(features):

                # Compute dot product between weight vector and features
                y_pred: float = self.activation(np.dot(w, x))

                # If we made a mistake, update weights
                if y_pred != y[i]:
                    w += self.lr * (y[i] - y_pred) * x

        self.weights = w

    @staticmethod
    def activation(x: float) -> float:
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def add_bias(features: np.ndarray) -> np.ndarray:
        """
        Add a bias column (value = 1) to features
        :param features: original data
        :return: new data
        """
        bias: np.ndarray = np.ones((features.shape[0], 1))
        return np.hstack((features, bias))

    @staticmethod
    def init_weights(features: np.ndarray) -> np.ndarray:
        """
        Initialize weights according to a uniform distribution
        :param features: feature set (w/ bias)
        :return: weight vector
        """
        bounds: float = 1.0 / np.sqrt(features.shape[0])
        return np.random.default_rng().uniform(-bounds, bounds, size=features.shape[1])

    def predict(self: "LogisticRegression", features: np.ndarray) -> np.ndarray:
        features: np.ndarray = self.add_bias(features)
        return np.round(self.activation(features.dot(self.weights)))


if __name__ == "__main__":
    features: np.ndarray = np.linspace(-10, 10)
    t: float = 1.0 / (1.0 + np.exp(-features))

    features: np.ndarray = np.column_stack((features, t))
    y: np.ndarray = np.where(features[:, 0] > 0, 1, 0)

    lr: LogisticRegression = LogisticRegression(epochs=1000)
    lr.fit(features, y)
    preds: np.ndarray = lr.predict(features)
