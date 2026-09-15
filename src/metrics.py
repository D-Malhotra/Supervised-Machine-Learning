"""Regression metrics."""

import numpy as np


def pred(X, beta):
    return np.matmul(X, beta)


def residuals(y, y_pred):
    return y - y_pred


def rss(y, y_pred):
    return np.dot(residuals(y, y_pred), residuals(y, y_pred))


def mse(y, y_pred):
    N = len(y)
    return 1 / N * rss(y, y_pred)


def rsquare(y, y_pred):
    ybar = np.mean(y)
    RSS0 = rss(y, ybar)

    return 1 - rss(y, y_pred) / RSS0
