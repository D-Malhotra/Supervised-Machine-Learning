"""k-nearest-neighbours regression.

Distances are unweighted Euclidean, so the features must be standardised
before any of this is called.
"""

import numpy as np

from src.metrics import rsquare


def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2, axis=1))


def k_neighbours(X_train, X_test, k=5, return_distance=False):
    """Indices of the k training points closest to each row of `X_test`."""
    dist = []
    neigh_ind = []

    point_dist = [euclidean_distance(x_test, X_train) for x_test in X_test]

    for row in point_dist:
        sorted_neigh = sorted(enumerate(row), key=lambda x: x[1])[:k]

        neigh_ind.append([tup[0] for tup in sorted_neigh])
        dist.append([tup[1] for tup in sorted_neigh])

    if return_distance:
        return np.array(dist), np.array(neigh_ind)

    return np.array(neigh_ind)


def knn_predict(X_train, y_train, X_test, k=5):
    """Predict each test point as the mean target of its k neighbours."""
    neighbours = k_neighbours(X_train, X_test, k=k)

    return np.array([np.mean(y_train[neighbour]) for neighbour in neighbours])


def knn_score(X_train, y_train, X_test, y_test, k=5):
    y_pred = knn_predict(X_train, y_train, X_test, k=k)

    return rsquare(y_test, y_pred)
