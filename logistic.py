"""Logistic regression on the same bank data: the simple model to beat.

One weight per feature plus an intercept, and the prediction is
    p = sigmoid(b + w_income * income + w_missed * missed)
so the whole model fits on one line and every weight can be read directly:
it is the change in log-odds of default for one more unit of that feature.

The script shows, in order:
  1. the fitted weights and what they mean (odds ratios);
  2. that predict_proba is nothing more than the sigmoid of that line;
  3. the same fit by hand with Newton's method, using the g and h that
     XGBoost uses (g = p - y, h = p(1 - p)): same loss, same derivatives,
     only the model differs (one line here, a sum of trees there);
  4. test accuracy next to toy_gbm.py (a tie: the data is nearly linear in
     log-odds), and how a line handles an interaction (only as a new feature);
  5. L2 regularisation, the counterpart of XGBoost's lambda.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from bank import X, y

N_TRAIN = 400  # rows 0-399 train, 400-499 test, as in toy_gbm.py
NAMES = ("income", "missed")

train, test = np.arange(N_TRAIN), np.arange(N_TRAIN, len(y))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def newton(Z, y, n_iter=25):
    """Maximum likelihood by Newton's method. Z has a leading column of ones."""
    w = np.zeros(Z.shape[1])
    for _ in range(n_iter):
        p = sigmoid(Z @ w)
        g = p - y  # first derivative of log-loss w.r.t. the margin, per row
        h = p * (1 - p)  # second derivative, per row
        w -= np.linalg.solve(Z.T @ (Z * h[:, None]), Z.T @ g)
    return w


if __name__ == "__main__":
    # C=np.inf: no regularisation, plain maximum likelihood (scikit-learn >= 1.8 spelling).
    # newton-cholesky is Newton's method, the same algorithm as newton() below.
    model = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10)
    model.fit(X[train], y[train])
    b, w = model.intercept_[0], model.coef_[0]

    print("1. Weights (log-odds per unit, odds ratio per unit)")
    print(f"   intercept  {b:+.4f}")
    for name, wi in zip(NAMES, w, strict=True):
        print(f"   {name:<9}  {wi:+.4f}   x{np.exp(wi):.3f}")
    print(
        "   e.g. one more missed payment multiplies the odds of default by "
        f"{np.exp(w[1]):.2f}; 10k EUR more income multiplies them by {np.exp(10 * w[0]):.2f}"
    )

    print("\n2. predict_proba is sigmoid(b + X @ w)")
    by_hand = sigmoid(b + X @ w)
    print(f"   max |difference| {np.abs(by_hand - model.predict_proba(X)[:, 1]).max():.1e}")

    print("\n3. Same fit by Newton's method with XGBoost's g and h")
    Z = np.column_stack([np.ones(len(y)), X])
    w_newton = newton(Z[train], y[train])
    print(f"   newton  {np.round(w_newton, 4)}")
    print(f"   sklearn {np.round(np.r_[b, w], 4)}")
    print(f"   max |difference| {np.abs(w_newton - np.r_[b, w]).max():.1e}")

    print("\n4. Test set (100 customers)")
    p_test = model.predict_proba(X[test])[:, 1]
    correct = int(((p_test > 0.5) == y[test]).sum())
    print(f"   correct {correct}/100   log-loss {log_loss(y[test], p_test):.4f}")
    print(f"   always 'no default': {int((y[test] == 0).sum())}/100   toy_gbm.py: 78/100")
    # bank.py's true logit, rewritten as intercept + slopes, is
    #   0.4 - 0.05 income + 0.9 missed  (+ an income x (missed > 2) interaction).
    # A line can only learn the first part; trees can find an interaction by
    # themselves, a line needs it handed over as a feature. Here the interaction
    # is weak, so it changes almost nothing: on this data the line already ties
    # boosting. Boosting earns its keep when the signal is not close to linear.
    inter = np.column_stack([X, (X[:, 0] - 40) * (X[:, 1] > 2)])
    model_i = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10)
    model_i.fit(inter[train], y[train])
    p_i = model_i.predict_proba(inter[test])[:, 1]
    print(
        f"   + hand-made interaction feature: correct {int(((p_i > 0.5) == y[test]).sum())}/100"
        f"   log-loss {log_loss(y[test], p_i):.4f}"
    )
    print(
        f"   true weights 0.4, -0.05, 0.9, -0.02   fitted "
        f"{np.round(np.r_[model_i.intercept_, model_i.coef_[0]], 3)}"
    )

    print("\n5. L2 regularisation: minimise sum(log-loss) + |w|^2 / (2C)")
    print("   1/C plays the role of XGBoost's lambda (there on leaf weights, here on")
    print("   the slopes; the intercept is not penalised). Smaller C, smaller weights:")
    for C in (np.inf, 1.0, 0.01, 0.001):
        m = LogisticRegression(C=C, solver="newton-cholesky", tol=1e-10).fit(X[train], y[train])
        acc = int((m.predict(X[test]) == y[test]).sum())
        print(
            f"   C={C:<6}  income {m.coef_[0, 0]:+.4f}  missed {m.coef_[0, 1]:+.4f}"
            f"   correct {acc}/100"
        )
    print("   The penalty depends on feature units (income in k EUR, missed as a count);")
    print("   in real use, standardise features first so all weights are penalised alike.")
