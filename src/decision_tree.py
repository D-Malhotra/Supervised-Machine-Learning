"""A weighted, cross-entropy decision tree.

Trees are plain nested dicts. A leaf has only a 'majority_label'; an internal
node also carries the feature and threshold it splits on, plus 'left' and
'right' subtrees.
"""

import numpy as np

from src.random_state import unseeded_rng

MIN_SAMPLES_LEAF = 2


def cross_entropy(y, sample_weights):
    """Weighted cross entropy of the labels in `y`."""
    label_weights = {yi: 0 for yi in set(y)}
    for yi, wi in zip(y, sample_weights):
        label_weights[yi] += wi

    total_weight = sum(label_weights.values())

    entropy = 0
    for weight in label_weights.values():
        pi = weight / total_weight
        if pi > 0:
            entropy -= pi * np.log(pi)

    return entropy


def split_samples(X, y, sample_weights, column, value):
    """Partition the samples on `X[:, column] < value`."""
    left_mask = X[:, column] < value

    X_left, X_right = X[left_mask, :], X[~left_mask, :]
    y_left, y_right = y[left_mask], y[~left_mask]
    w_left, w_right = sample_weights[left_mask], sample_weights[~left_mask]

    return (X_left, X_right), (y_left, y_right), (w_left, w_right)


def best_split_value(X, y, sample_weights, column):
    """Threshold within `column` that minimises the weighted child entropy."""
    unique_vals = np.unique(X[:, column])

    CE_val, threshold = np.inf, None

    for value in unique_vals:
        _, (y_l, y_r), (w_l, w_r) = split_samples(X, y, sample_weights, column, value)

        # A split that puts everything on one side carries no information.
        if len(y_l) == 0 or len(y_r) == 0:
            continue

        p_left = sum(w_l) / (sum(w_l) + sum(w_r))
        p_right = 1 - p_left
        new_cost = p_left * cross_entropy(y_l, w_l) + p_right * cross_entropy(y_r, w_r)

        if new_cost < CE_val:
            CE_val, threshold = new_cost, value

    return CE_val, threshold


def best_split(X, y, sample_weights, n_features=None, rng=None):
    """Best (entropy, column, threshold) over a random subset of features.

    Sampling `n_features` of the columns at each node is what separates a
    random forest from plain bagging. Passing `n_features=None` searches every
    column, which gives an ordinary decision tree.
    """
    n_columns = np.shape(X)[1]

    if n_features is None or n_features >= n_columns:
        candidate_columns = range(n_columns)
    else:
        rng = unseeded_rng(rng)
        candidate_columns = rng.choice(n_columns, size=n_features, replace=False)

    min_CE, split_column, split_val = np.inf, 0, 0

    for column in candidate_columns:
        # Skip columns whose samples cannot be separated.
        if len(np.unique(X[:, column])) < 2:
            continue

        ce, val = best_split_value(X, y, sample_weights, column)
        if ce < min_CE:
            min_CE, split_column, split_val = ce, column, val

    return min_CE, split_column, split_val


def majority_vote(y, sample_weights):
    """The label carrying the most weight in `y`."""
    majority_label = {yi: 0 for yi in set(y)}
    for yi, wi in zip(y, sample_weights):
        majority_label[yi] += wi

    return max(majority_label, key=majority_label.get)


def build_tree(
    X,
    y,
    sample_weights,
    feature_names,
    max_depth,
    depth=1,
    n_features=None,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    rng=None,
):
    """Grow a tree until the labels are pure, the depth caps out, or the node
    is too small to split."""
    if len(np.unique(y)) == 1 or depth >= max_depth or len(X) <= min_samples_leaf:
        return {"majority_label": majority_vote(y, sample_weights)}

    CE, split_index, split_val = best_split(X, y, sample_weights, n_features, rng)

    # Infinite entropy means the sampled features cannot separate the samples.
    if CE == np.inf:
        return {"majority_label": majority_vote(y, sample_weights)}

    (X_l, X_r), (y_l, y_r), (w_l, w_r) = split_samples(
        X, y, sample_weights, split_index, split_val
    )

    child_args = (feature_names, max_depth, depth + 1, n_features, min_samples_leaf, rng)

    return {
        "feature_name": feature_names[split_index],
        "feature_index": split_index,
        "value": split_val,
        "majority_label": None,
        "left": build_tree(X_l, y_l, w_l, *child_args),
        "right": build_tree(X_r, y_r, w_r, *child_args),
    }


def classify(tree, x):
    """Classify a single sample by walking the tree to a leaf."""
    if tree["majority_label"] is not None:
        return tree["majority_label"]

    if x[tree["feature_index"]] < tree["value"]:
        return classify(tree["left"], x)

    return classify(tree["right"], x)
