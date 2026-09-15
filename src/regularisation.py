"""Lasso and elastic-net regression, fitted by gradient descent.

Both penalties leave the intercept (`beta[0]`) unshrunk, so every gradient
below zeroes its first entry before adding the penalty term.
"""

import numpy as np

from src.metrics import pred, rss

MAX_ITERATIONS = 1000
TOLERANCE = 1e-8


def _penalised_beta(beta):
    """Return a copy of `beta` with the intercept zeroed out."""
    without_intercept = beta.copy()
    without_intercept[0] = 0

    return without_intercept


def lasso_loss(y, X, beta, l=0.01):
    N = len(y)
    y_pred = pred(X, beta)

    return rss(y, y_pred) / (2 * N) + l * np.sum(np.abs(beta[1:]))


def lasso_grad(y, X, beta, l):
    N = len(y)
    XtY = np.matmul(np.transpose(X), y)
    XtX = np.matmul(np.transpose(X), X)

    return -1 * (XtY - np.dot(XtX, beta)) / N + l * np.sign(_penalised_beta(beta))


def elastic_net_loss(y, X, beta, l=0.01, alpha=0.01):
    """Elastic-net objective: RSS/(2N) + l * (alpha*||b||_1 + (1-alpha)*||b||_2^2).

    The squared L2 norm is deliberate -- it is what `elastic_net_grad` below
    differentiates, so loss and gradient describe the same objective.
    """
    N = len(y)
    y_pred = pred(X, beta)

    l1 = np.sum(np.abs(beta[1:]))
    l2_squared = np.dot(beta[1:], beta[1:])

    return rss(y, y_pred) / (2 * N) + l * (alpha * l1 + (1 - alpha) * l2_squared)


def elastic_net_grad(y, X, beta, l, alpha=0.1):
    N = len(y)
    XtY = np.matmul(np.transpose(X), y)
    XtX = np.matmul(np.transpose(X), X)

    penalised = _penalised_beta(beta)
    penalty_grad = alpha * np.sign(penalised) + 2 * (1 - alpha) * penalised

    return -1 * (XtY - np.dot(XtX, beta)) / N + l * penalty_grad


def _relative_improvement(loss_fn, y, X, beta, step, d, *args):
    """Relative drop in loss from taking one step -- the stopping criterion."""
    current = loss_fn(y, X, beta, *args)
    updated = loss_fn(y, X, beta + step * d, *args)

    return abs((current - updated) / current)


def _descend(loss_fn, grad_fn, y, X, beta, *args):
    """Shared gradient-descent loop with a decaying step size."""
    step = 1
    d = -1 * grad_fn(y, X, beta, *args)
    counter = 1

    while (counter < MAX_ITERATIONS) and _relative_improvement(
        loss_fn, y, X, beta, step, d, *args
    ) > TOLERANCE:
        beta = beta + step * d
        d = -1 * grad_fn(y, X, beta, *args)
        counter += 1
        # Step size shrinks with the iteration count.
        step = 10 / counter

    return beta


def fit_lasso(y, X, beta, l):
    return _descend(lasso_loss, lasso_grad, y, X, beta, l)


def fit_elastic_net(y, X, beta, l, alpha):
    return _descend(elastic_net_loss, elastic_net_grad, y, X, beta, l, alpha)


def coefficient_path(y, X, lambda_values, fit_fn=fit_lasso, *args):
    """Absolute coefficient values across a range of lambda, for path plots."""
    betas = [
        fit_fn(y, X, np.ones(X.shape[1]), l, *args) for l in lambda_values
    ]

    return np.abs(np.array(betas))
