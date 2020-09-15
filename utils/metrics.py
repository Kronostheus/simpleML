import numpy as np


def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    return np.sum(y_true == y_pred, axis=0) / len(y_true)


def r2_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    true_mean = np.mean(y_true, axis=0)

    # Total sum of squares
    tss = sum([(true - true_mean) ** 2 for true in y_true])

    # Residual sum of squares
    rss = sum([(true - pred) ** 2 for true, pred in zip(y_true, y_pred)])

    return 1 - (rss / tss)
