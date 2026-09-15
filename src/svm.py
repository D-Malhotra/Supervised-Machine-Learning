"""Linear soft-margin SVM, fitted by stochastic gradient descent.

Labels must be encoded as {-1, +1}. The intercept is carried as the *last*
column of the design matrix (see `append_intercept`), because the gradient
excludes that final weight from the regularisation term.
"""

import numpy as np

from src.random_state import get_rng

DEFAULT_REGULARISATION = 1e5


def append_intercept(X):
    """Append a column of ones as the final column.

    Note this appends rather than prepends -- the opposite of
    `preprocessing.add_intercept`, which the linear-regression models use. The
    SVM gradient identifies the bias by position, as `w[-1]`.
    """
    return np.hstack((X, np.ones((len(X), 1))))


def hinge_loss(w, X, y, regul_strength=DEFAULT_REGULARISATION):
    distances = 1 - y * (X @ w)
    distances[distances < 0] = 0
    hinge = regul_strength * np.sum(distances)

    return 0.5 * np.dot(w, w) + hinge


def cost_gradient(w, X_batch, y_batch, l):
    """Subgradient of the hinge loss, averaged over the batch."""
    # Handle a single sample being passed in.
    if type(y_batch) == np.float64:
        y_batch = np.asarray([y_batch])
        X_batch = np.asarray([X_batch])

    distance = 1 - (y_batch * (X_batch @ w))
    dw = np.zeros(len(w))

    regularised = w.copy()
    regularised[-1] = 0  # The bias is not regularised.

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = regularised
        else:
            di = regularised - (l * y_batch[ind] * X_batch[ind])
        dw += di

    return dw / len(y_batch)


def sgd(
    X, y, l, max_iterations=2000, stop_criterion=0.01, learning_rate=1e-5,
    rng=None, print_outcome=False,
):
    """Fit the weight vector by SGD, checking convergence on 2^n-th passes."""
    rng = get_rng(rng)

    weights = np.zeros(X.shape[1])
    nth = 0
    prev_cost = np.inf

    for iteration in range(1, max_iterations):
        # Shuffle each pass to avoid repeating update cycles.
        order = rng.permutation(len(y))
        X, y = X[order], y[order]

        for xi, yi in zip(X, y):
            descent = cost_gradient(weights, xi, yi, l)
            weights = weights - (learning_rate * descent)

        if iteration == 2 ** nth or iteration == max_iterations - 1:
            cost = hinge_loss(weights, X, y, l)
            if print_outcome:
                print("Iteration is: {}, Cost is: {}".format(iteration, cost))

            if abs(prev_cost - cost) < stop_criterion * prev_cost:
                return weights

            prev_cost = cost
            nth += 1

    return weights


def svm_score(w, X, y):
    """Accuracy, with `y` in {-1, +1}."""
    y_preds = np.sign(X @ w)

    return np.mean(y_preds == y)


def to_binary_labels(y):
    """Map {-1, +1} labels to {0, 1} for the shared classification metrics."""
    binary = np.array(y, dtype=float).copy()
    binary[binary < 0] = 0

    return binary


def cosine_similarity(x, y):
    """Cosine of the angle between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cosine_matrix(vectors):
    """Pairwise cosine similarity matrix -- symmetric by construction."""
    return np.array(
        [[cosine_similarity(a, b) for a in vectors] for b in vectors]
    )
