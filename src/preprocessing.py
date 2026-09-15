"""Feature scaling and design-matrix helpers."""

import numpy as np


def standardise(X):
    """Centre and scale `X`, returning the scaled matrix plus the statistics.

    The returned `mu` and `sigma` must be reused on the test set so that both
    sets sit on the same scale.
    """
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_std = (X - mu) / sigma

    return X_std, mu, sigma


def apply_standardisation(X, mu, sigma):
    """Scale `X` with statistics already learned from the training set."""
    return (X - mu) / sigma


def add_intercept(X):
    """Prepend a column of ones so the intercept is fitted as a weight."""
    return np.c_[np.ones(len(X)), X]
