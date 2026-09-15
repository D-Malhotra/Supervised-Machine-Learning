"""Cross-validation and hyperparameter search.

The original notebook defined `cross_val` twice with incompatible signatures --
once for the regression models and once for the forest -- so the second
silently overwrote the first. They live here under distinct names.

Every routine takes an `rng`, so results are reproducible.
"""

import numpy as np

from src.knn import knn_score
from src.metrics import pred
from src.random_forest import forest_accuracy, train_forest
from src.regularisation import fit_elastic_net, fit_lasso
from src.random_state import DEFAULT_SEED, get_rng
from src.svm import sgd, svm_score

# DEFAULT_SEED and get_rng are re-exported here so a notebook can take its
# folds and its seed from the same module.
__all__ = [
    "DEFAULT_SEED",
    "get_rng",
    "make_folds",
    "cross_val",
    "sweep_lambda",
    "grid_search_elastic_net",
    "choose_best_k",
    "cross_val_forest",
    "grid_search_forest",
    "cross_val_svm",
]


def make_folds(n_samples, k, rng=None, shuffle=True):
    """Split `n_samples` indices into `k` folds.

    Uses `array_split`, so it does not require `n_samples` to divide evenly by
    `k`. Shuffling is on by default -- contiguous folds are only meaningful if
    the rows are already in random order.
    """
    indices = np.arange(n_samples)
    if shuffle:
        indices = get_rng(rng).permutation(indices)

    return np.array_split(indices, k)


def _train_indices(folds, held_out):
    return np.concatenate([f for i, f in enumerate(folds) if i != held_out])


def cross_val(y, X, folds, fit_fn, score_fn, *fit_args):
    """Mean validation score for a linear model fitted by `fit_fn`."""
    scores = []

    for i in range(len(folds)):
        val_idx = folds[i]
        train_idx = _train_indices(folds, i)

        beta = fit_fn(
            y[train_idx], X[train_idx, :], np.ones(X.shape[1]), *fit_args
        )
        scores.append(score_fn(y[val_idx], pred(X[val_idx, :], beta)))

    return np.mean(scores)


def sweep_lambda(y, X, folds, lambda_values, score_fn, fit_fn=fit_lasso):
    """Cross-validated score at each value of lambda."""
    return np.array(
        [cross_val(y, X, folds, fit_fn, score_fn, l) for l in lambda_values]
    )


def grid_search_elastic_net(y, X, folds, lambda_values, alpha_values, score_fn):
    """Best lambda for each alpha, minimising the cross-validated score.

    Returns a list of (alpha, best_lambda, best_score), one row per alpha.
    """
    results = []

    for alpha in alpha_values:
        best_score, best_lambda = np.inf, lambda_values[0]

        for l in lambda_values:
            score = cross_val(y, X, folds, fit_elastic_net, score_fn, l, alpha)
            if score < best_score:
                best_score, best_lambda = score, l

        results.append((alpha, best_lambda, best_score))

    return results


def choose_best_k(X, y, folds, k_range):
    """Cross-validated R^2 for each k, plus the k that maximises it."""
    k_scores = np.zeros(len(k_range))

    for i, k in enumerate(k_range):
        scores = []
        for j in range(len(folds)):
            val_idx = folds[j]
            train_idx = _train_indices(folds, j)
            scores.append(
                knn_score(X[train_idx, :], y[train_idx], X[val_idx, :], y[val_idx], k=k)
            )
        k_scores[i] = np.mean(scores)

    return k_range[np.argmax(k_scores)], k_scores


def cross_val_forest(
    y, X, feature_names, folds, n_trees, max_depth, sample_weights=None,
    n_features="sqrt", rng=None,
):
    """Mean validation accuracy for a forest at the given settings."""
    rng = get_rng(rng)
    scores = []

    for i in range(len(folds)):
        val_idx = folds[i]
        train_idx = _train_indices(folds, i)

        weights = None if sample_weights is None else sample_weights[train_idx]
        forest = train_forest(
            X[train_idx, :], y[train_idx], feature_names, n_trees, max_depth,
            weights, n_features, rng,
        )
        scores.append(forest_accuracy(forest, X[val_idx, :], y[val_idx]))

    return np.mean(scores)


def grid_search_forest(
    y, X, feature_names, folds, n_trees_values, depth_values,
    sample_weights=None, n_features="sqrt", rng=None,
):
    """Search n_trees x max_depth, maximising cross-validated accuracy.

    Returns (best_n_trees, best_max_depth, best_score, score_grid).

    Every configuration is evaluated against the same bootstrap draws, by
    reseeding from a single seed rather than letting one generator advance
    across the grid. Scores here differ by a few thousandths between
    configurations, so an unpaired comparison would be selecting on resampling
    noise rather than on the hyperparameters.
    """
    seed = int(get_rng(rng).integers(0, 2 ** 32))
    grid = np.zeros((len(n_trees_values), len(depth_values)))

    for i, n in enumerate(n_trees_values):
        for j, d in enumerate(depth_values):
            grid[i, j] = cross_val_forest(
                y, X, feature_names, folds, n, d, sample_weights, n_features,
                np.random.default_rng(seed),
            )

    best_i, best_j = np.unravel_index(np.argmax(grid), grid.shape)

    return n_trees_values[best_i], depth_values[best_j], grid[best_i, best_j], grid


def cross_val_svm(y, X, folds, l, rng=None, **sgd_kwargs):
    """Mean validation accuracy for the linear SVM at regularisation `l`."""
    rng = get_rng(rng)
    scores = []

    for i in range(len(folds)):
        val_idx = folds[i]
        train_idx = _train_indices(folds, i)

        w = sgd(X[train_idx, :], y[train_idx], l, rng=rng, **sgd_kwargs)
        scores.append(svm_score(w, X[val_idx, :], y[val_idx]))

    return np.mean(scores)
