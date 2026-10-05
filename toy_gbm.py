"""Gradient boosting in its simplest form: squared error, depth-2 trees.

Readable rewrite of a 16-line toy (verbatim copy in original/toy_gbm_16_lines.py).
Same algorithm, same output: the base rate, the first tree's leaf values, and
the number of correct predictions on the 100 test customers.
"""

import numpy as np

from bank import X, y

N_TRAIN = 400  # rows 0-399 train, 400-499 test
FEATURES = (0, 1)  # income, missed payments
MAX_DEPTH = 2
N_TREES = 200
LEARNING_RATE = 0.01


def sse(e):
    """Sum of squared errors around the mean: how badly one constant fits e."""
    return ((e - e.mean()) ** 2).sum()


def grow(p, rows, depth=MAX_DEPTH):
    """Fit one regression tree to the residuals y - p, using only `rows`.

    Returns the tree's prediction for every customer, train and test, so the
    boosting loop can add it to p directly.
    """
    residual = (y - p)[rows]

    best_sse, best_split = np.inf, None
    for f in FEATURES:
        # every observed value except the largest, so both sides are non-empty
        for threshold in np.unique(X[rows, f])[:-1]:
            yes = X[rows, f] > threshold
            split_sse = sse(residual[yes]) + sse(residual[~yes])
            if split_sse < best_sse:
                best_sse, best_split = split_sse, (f, threshold, yes)

    if depth < 1 or best_split is None:
        return residual.mean()  # leaf: predict the mean residual

    f, threshold, yes = best_split
    yes_branch = grow(p, rows[yes], depth - 1)
    no_branch = grow(p, rows[~yes], depth - 1)
    return np.where(X[:, f] > threshold, yes_branch, no_branch)


if __name__ == "__main__":
    train = np.arange(N_TRAIN)
    p = y[train].mean()  # start every customer at the base rate
    print(p, np.unique(grow(p, train)))  # leaf values of the first tree

    for _ in range(N_TREES):
        p = p + LEARNING_RATE * grow(p, train)

    print(((p > 0.5) == y)[N_TRAIN:].sum())
