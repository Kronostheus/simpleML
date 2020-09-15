import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


class LinearRegression:
    def __init__(self):
        self.b1 = None
        self.b0 = None

    def fit(self, x, y):
        """
        Calculate parameters for linear function y = mx + b
        b1 -> m
        b0 -> b
        :param x: xaxis
        :param y: yaxis
        :return: None
        """

        # Coefficient is the division between cov(x,y) and var(x)
        self.b1 = np.cov(x.T, y, axis=0)[0][1] / np.var(x, axis=0)

        # Intercept is mean(y) - coefficient * mean(x)
        self.b0 = np.mean(y, axis=0) - self.b1 * np.mean(x, axis=0)

    def predict(self, x):
        """
        Predicts y based on y = mx + b
        m -> coefficient/slope
        b -> intercept
        :param x: xaxis of points to predict
        :return: y associated with x
        """
        # y = mx + b for all X
        return [self.b1 * xi + self.b0 for xi in x]


X, y, coef = make_regression(n_samples=300, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)

lr = LinearRegression()
lr.fit(X, y)

x_line = np.arange(X.min(), X.max())
y_line = lr.predict(x_line)

fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(X, y, alpha=0.5)
ax.plot(x_line, y_line, color='navy')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
fig.show()
