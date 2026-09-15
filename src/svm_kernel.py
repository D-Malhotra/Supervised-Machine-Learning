"""Kernelised SVM, fitted by stochastic gradient descent in the dual.

The model learns coefficients `u` over the training points plus a separate
intercept `b`, so predictions are sign(K(X_test, X_train) @ u + b).
"""

import numpy as np

from src.classification_metrics import accuracy, f_score, precision
from src.random_state import get_rng
from src.svm import to_binary_labels


def sigmoid_kernel(X1, X2, sigma):
    """Sigmoid (tanh) kernel matrix between the rows of `X1` and `X2`."""
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            kernel[i][j] = np.tanh(sigma * np.dot(X1[i], X2[j]) + 1)

    return kernel


def kernel_cost(u, K, y, l, b=0):
    """Hinge cost under the kernel trick."""
    distances = 1 - y * (K @ u + b)
    distances[distances < 0] = 0
    hinge = l * distances.mean()

    return 0.5 * np.dot(u, K @ u) + hinge


def kernel_cost_gradient(u, K_batch, y_batch, l=1e3, b=0):
    """Subgradient with respect to `u` and `b`, returned as one stacked array."""
    # Handle a single sample being passed in.
    if type(y_batch) == np.float64 or type(y_batch) == np.int32:
        y_batch = np.asarray([y_batch])
        K_batch = np.asarray([K_batch])

    distance = 1 - (y_batch * (K_batch @ u + b))
    dw = np.zeros(len(u) + 1)

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = K_batch @ u
            db = 0
        else:
            di = K_batch @ u - (l * y_batch[ind] * K_batch[ind])
            db = -l * y_batch[ind]
        dw[:-1] += di
        dw[-1] += db

    return dw / len(y_batch)


def sgd_kernel(
    K, y, batch_size=32, max_iterations=4000, stop_criterion=0.001,
    learning_rate=1e-4, l=1e3, rng=None, print_outcome=False,
):
    """Fit the dual coefficients and intercept by minibatch SGD."""
    rng = get_rng(rng)

    u = np.zeros(K.shape[1])
    b = 0

    nth = 0
    prev_cost = np.inf

    for iteration in range(1, max_iterations):
        batch_idx = rng.permutation(len(y))[:batch_size]
        K_b, y_b = K[batch_idx], y[batch_idx]

        for ki, yi in zip(K_b, y_b):
            cost_grad = kernel_cost_gradient(u, ki, yi, l, b)
            ascent, ascent_intercept = cost_grad[:-1], cost_grad[-1]
            u = u - (learning_rate * ascent)
            b = b - (learning_rate * ascent_intercept)

        if iteration == 2 ** nth or iteration == max_iterations - 1:
            cost = kernel_cost(u, K, y, l, b)
            if print_outcome:
                print("Iteration is: {}, Cost is: {}".format(iteration, cost))

            if abs(prev_cost - cost) < stop_criterion * prev_cost:
                return u, b

            prev_cost = cost
            nth += 1

    return u, b


def kernel_predict(u, b, X_train, X_eval, sigma):
    """Predicted labels in {0, 1} for the rows of `X_eval`."""
    K_eval = sigmoid_kernel(X_eval, X_train, sigma)
    y_preds = np.sign(K_eval @ u + b)

    return to_binary_labels(y_preds)


def kernel_score(u, b, X_train, X_eval, y_eval, sigma):
    """Accuracy, precision and F-score, with `y_eval` in {0, 1}.

    `X_train` is passed explicitly rather than read from module state, so the
    same fitted model can be scored against any evaluation set.
    """
    y_preds = kernel_predict(u, b, X_train, X_eval, sigma)

    return accuracy(y_eval, y_preds), precision(y_eval, y_preds), f_score(y_eval, y_preds)
