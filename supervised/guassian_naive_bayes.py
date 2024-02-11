import math
from collections import namedtuple

import numpy as np

param = namedtuple("param", "mean variance")


class GaussianNB:
    def __init__(self):
        self.parameters = None
        self.classes = None
        self.class_priors = None

    def fit(self, features, labels):
        """
        Initializes parameters:
        Class labels;
        Calculate class priors (frequency);
        For every class -> calculate mean and variance for each column (feature);
        :param features: features
        :param labels: labels
        :return: None
        """

        # Save class label array
        self.classes = np.unique(labels)

        # Save P(c) for every class c
        self.class_priors = {c: np.mean(labels == c) for c in self.classes}

        # Save mean and variance of features (columns) attending to their label/class
        self.parameters = {c: [param(np.mean(Xc), np.var(Xc))
                               for Xc in features[np.where(labels == c)].T]
                           for c in self.classes}

    @staticmethod
    def conditional_probability(x, parameter):
        """
        Computes likelihood assuming a Gaussian (normal) distribution of each class.
        :param x: feature value
        :param parameter: parameter tuple associated with feature and class
        :return: probability distribution of x given class c -> P(x|c)
        """
        assert parameter.variance != 0
        numerator = math.exp(-((x - parameter.mean) ** 2) / (2 * parameter.variance))
        denominator = math.sqrt(2 * math.pi * parameter.variance)
        return numerator / denominator

    def predict(self, features):
        """
        Computes the labels for unseen data using a joint probability model classifier, leveraging the chain rule:
            P(C, x1, ..., xn) = P(C) * P(x1|C) * ... * P(xn|C)
        The predicted label will be the one who maximizes this probability.
        Assumes that features x are mutually independent, conditional on the class.
        :param features: data samples to predict
        :return: predicted labels
        """
        return [
            # Get the class
            self.classes[
                # That maximizes
                np.argmax([
                    # The product of the features probabilities conditional on each class, along with said class' prior
                    np.prod([self.conditional_probability(feature, p)
                             for feature, p in zip(sample, params)] + [self.class_priors[c]])
                    # Do this for every class and respective parameters
                    for c, params in self.parameters.items()])
            ]
            # For every data sample
            for sample in features
        ]



if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(np.array([[-0.8, -1]]))
