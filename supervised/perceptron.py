import numpy as np


class Perceptron:

    def __init__(self, epochs=1, learning_rate=0.1):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, features, y):
        """
        Train perceptron
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

    def predict(self, features):
        """
        Predicts the labels associated with features, equivalent to forward pass
        :param features: data samples
        :return: labels
        """
        features = self.add_bias(features)
        return np.array([self.activation(np.dot(self.weights, x)) for x in features])

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

    @staticmethod
    def activation(x):
        """
        Activation function
        :param x: equivalent to the dot product of weight vector and a data sample
        :return: -1 or 1 depending if x is positive
        """
        return -1 if x < 0 else 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # Positive examples
    P = np.array([[-2, 0], [-1, 0.5], [0, -0.1], [1, -1.5]])

    # Negative examples
    N = np.array([[2, 0], [1, 0.5], [0, 0.4], [1, 1.5]])

    features = np.vstack((P, N))
    y = np.hstack((np.ones(P.shape[0]), -np.ones(N.shape[0])))

    clf = Perceptron(epochs=1000, learning_rate=0.01)
    clf.fit(features, y)
    preds = clf.predict(features)

    p_pred = features[np.nonzero(preds == 1)]
    n_pred = features[np.nonzero(preds == -1)]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(P[:, 0], P[:, 1], 'go', label='True Positive Examples')
    ax1.plot(N[:, 0], N[:, 1], 'ro', label='True Negative Examples')
    ax1.legend(loc='best')

    ax2.plot(p_pred[:, 0], p_pred[:, 1], 'go', label='Predicted Positive Examples')
    ax2.plot(n_pred[:, 0], n_pred[:, 1], 'ro', label='Predicted Positive Examples')
    ax2.legend(loc='best')
    plt.show()
