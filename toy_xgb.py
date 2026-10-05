"""toy_gbm.py rewritten with the XGBoost paper's math (Chen & Guestrin, KDD 2016).

Same skeleton as toy_gbm.py: a recursive grow() that returns the tree's
prediction for all 500 rows, and a boosting loop that adds eta * tree.
What changes, with the paper's equation numbers:

    loss            logistic (log-loss) instead of squared error
    g_i, h_i        first and second derivative of the loss       Sec. 2.2
    leaf weight     w* = -G / (H + lambda)                        Eq. (5)
    split gain      1/2 [GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l)]   Eq. (7)
    gamma           a split must gain more than gamma             Eq. (7)
                    BUT the library drops the 1/2: its gain (the number get_dump
                    prints) and its gamma are both 2x the paper's. Verified in
                    compare_xgboost.py; the toy uses the library's scale.
    split search    sort once per feature, cumulative sums        Alg. 1
    min_child       each child needs sum(h) >= min_child_weight   (library param)
    eta             shrinkage of every new tree                   Sec. 2.3

The model works in margin (log-odds) space; sigmoid(margin) is the probability.
"""

import numpy as np

from bank import X, y

X = X.astype(np.float32)  # the library stores features and thresholds as float32
N_TRAIN = 400


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def leaf_score(G, H, lam):
    """One leaf's term in Eq. (6), without the -1/2."""
    return G * G / (H + lam)


def best_split(rows, g, h, lam, min_child):
    """Exact greedy split search (Alg. 1). Returns (gain, feature, threshold) or None.

    Gain is on the library's scale: 2x Eq. (7), before subtracting gamma.
    """
    if len(rows) < 2:  # one row cannot be split (reachable with min_child=0)
        return None
    G, H = g[rows].sum(), h[rows].sum()
    best = None
    best_gain = 0.0  # a split must improve the objective at all
    for f in range(X.shape[1]):
        order = rows[np.argsort(X[rows, f], kind="stable")]  # sorted(I, by x_jk)
        x = X[order, f]
        # Alg. 1's inner loop, vectorised: candidate k puts rows 0..k on the left
        GL, HL = np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1]
        GR, HR = G - GL, H - HL
        gain = leaf_score(GL, HL, lam) + leaf_score(GR, HR, lam) - leaf_score(G, H, lam)
        valid = (x[:-1] < x[1:]) & (min_child <= HL) & (min_child <= HR)
        gain[~valid] = -np.inf
        k = gain.argmax()
        if gain[k] > best_gain:
            best_gain = gain[k]
            best = (gain[k], f, (x[k] + x[k + 1]) / 2)  # threshold at the midpoint
    return best


def grow(rows, g, h, depth, lam, gamma, min_child):
    """Grow one tree on `rows`.

    Returns (margin contribution for every customer, gain of the root split),
    with gain None when the node ends up a leaf.
    """
    leaf = np.full(len(X), -g[rows].sum() / (h[rows].sum() + lam))  # Eq. (5)
    split = best_split(rows, g, h, lam, min_child) if depth > 0 else None
    if split is None:
        return leaf, None

    gain, f, threshold = split
    left = X[rows, f] < threshold  # the library sends x < threshold to the left
    left_pred, left_gain = grow(rows[left], g, h, depth - 1, lam, gamma, min_child)
    right_pred, right_gain = grow(rows[~left], g, h, depth - 1, lam, gamma, min_child)

    # prune bottom-up: a split whose children are both leaves must gain more than gamma
    if left_gain is None and right_gain is None and gain <= gamma:
        return leaf, None
    return np.where(X[:, f] < threshold, left_pred, right_pred), gain


def fit(lam=1.0, gamma=0.0, eta=0.3, min_child=1.0, depth=2, n_trees=50, train=None):
    """Boost n_trees trees on the train rows; return the margin for every customer."""
    if train is None:
        train = np.arange(N_TRAIN)
    base = y[train].mean()
    margin = np.full(len(X), np.log(base / (1 - base)))  # start from the training log-odds
    for _ in range(n_trees):
        q = sigmoid(margin)
        g, h = q - y, q * (1 - q)  # log-loss derivatives w.r.t. the margin
        tree, _ = grow(train, g, h, depth, lam, gamma, min_child)
        margin = margin + eta * tree
    return margin


if __name__ == "__main__":
    margin = fit()
    correct = ((sigmoid(margin) > 0.5) == y)[N_TRAIN:].sum()
    print("test accuracy", correct, "/ 100")
