"""Ordinary least squares."""

import numpy as np


def least_squares(y, X):
    """Closed-form OLS estimate of beta.

    The squared-error loss is a constant multiple of the RSS, so minimising it
    reduces to solving the normal equations.
    """
    XtX = np.matmul(np.transpose(X), X)
    XtY = np.dot(np.transpose(X), y)
    betahat = np.linalg.solve(XtX, XtY)

    return betahat
