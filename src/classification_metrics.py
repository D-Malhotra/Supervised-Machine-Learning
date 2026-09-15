"""Classification metrics, for labels encoded as 0/1."""

import numpy as np


def true_positives(y_test, y_pred):
    return np.dot(y_pred, y_test)


def false_positives(y_test, y_pred):
    y_flip = np.ones(len(y_test)) - y_test

    return np.dot(y_pred, y_flip)


def accuracy(y_test, y_pred):
    return np.mean(y_test == y_pred)


def precision(y_test, y_pred):
    """Share of positive predictions that are correct.

    Undefined when the model predicts no positives at all -- reachable at the
    extremes of a threshold sweep, and for kernel widths that collapse onto the
    majority class. Reports 0.0 in that case rather than nan.
    """
    TP = true_positives(y_test, y_pred)
    FP = false_positives(y_test, y_pred)

    if TP + FP == 0:
        return 0.0

    return TP / (TP + FP)


def recall(y_test, y_pred):
    """Share of actual positives that are found."""
    n_positive = np.sum(y_test)

    if n_positive == 0:
        return 0.0

    return true_positives(y_test, y_pred) / n_positive


def f_score(y_test, y_pred):
    """Harmonic mean of precision and recall, 0.0 when both are zero."""
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)

    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)
