"""Random forest: bootstrap-aggregated decision trees with feature subsampling."""

from collections import defaultdict

import numpy as np

from src.classification_metrics import false_positives, precision, true_positives
from src.decision_tree import build_tree, classify
from src.random_state import get_rng


def resolve_n_features(n_features, n_columns):
    """Turn the `n_features` setting into a column count.

    'sqrt' is the usual choice for classification; None searches every column,
    which reduces the forest to plain bagging.
    """
    if n_features is None:
        return None
    if n_features == "sqrt":
        return max(1, int(np.sqrt(n_columns)))

    return int(n_features)


def train_forest(
    X, y, feature_names, n_trees, max_depth, sample_weights=None,
    n_features="sqrt", rng=None,
):
    """Fit `n_trees` trees, each on its own bootstrap resample of the rows."""
    rng = get_rng(rng)

    if sample_weights is None:
        # Uniform weights when none are supplied.
        sample_weights = np.ones(X.shape[0]) / X.shape[0]
    else:
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)

    n_sampled = resolve_n_features(n_features, X.shape[1])
    N = X.shape[0]

    trees = []
    for _ in range(n_trees):
        sample = rng.choice(N, size=N, replace=True)
        trees.append(
            build_tree(
                X[sample, :],
                y[sample],
                sample_weights[sample],
                feature_names,
                max_depth,
                depth=1,
                n_features=n_sampled,
                rng=rng,
            )
        )

    return trees


def predict_forest(forest, X):
    """Majority vote across the trees."""

    def aggregate(decisions):
        count = defaultdict(int)
        for decision in decisions:
            count[decision] += 1
        return max(count, key=count.get)

    if len(X.shape) == 1:
        return aggregate([classify(tree, X) for tree in forest])

    return np.array([aggregate([classify(tree, x) for tree in forest]) for x in X])


def predict_proba_forest(forest, X):
    """Fraction of trees voting for the positive class, per sample."""
    votes = np.array([[classify(tree, x) for tree in forest] for x in X])

    return votes.mean(axis=1)


def forest_accuracy(forest, X_test, y_test):
    return np.mean(predict_forest(forest, X_test) == y_test)


def balanced_sample_weights(y_train):
    """Inverse-frequency weights, so the minority class is not ignored."""
    N = len(y_train)
    positives = np.sum(y_train == 1)
    negatives = N - positives

    weights = (negatives / N) * y_train
    weights += (positives / N) * (np.ones(N) - y_train)

    return weights


def roc_curve(forest, X_test, y_test, thresholds):
    """True-positive rate, false-positive rate and precision at each threshold.

    Thresholding the fraction of trees voting positive traces out the curve.
    """
    vote_fraction = predict_proba_forest(forest, X_test)

    n_positive = np.sum(y_test)
    n_negative = len(y_test) - n_positive

    tpr, fpr, prec = [], [], []
    for threshold in thresholds:
        y_pred = (vote_fraction >= threshold).astype(float)

        tpr.append(true_positives(y_test, y_pred) / n_positive)
        fpr.append(false_positives(y_test, y_pred) / n_negative)
        prec.append(precision(y_test, y_pred))

    return np.array(tpr), np.array(fpr), np.array(prec)
