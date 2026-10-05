"""Synthetic bank data for the toy boosting scripts.

500 customers, two features, one binary label:
    X[:, 0]  income          annual income, k EUR
    X[:, 1]  missed          missed payments in the last 12 months
    y        defaulted       1 if the customer defaulted on the loan

The first 400 rows are the training set, the last 100 the test set
(toy_gbm.py hard-codes that split). The ground truth is a logistic model
with an interaction, so a depth-2 tree has something to find.

The data is frozen in bank.csv next to this file. If the CSV exists it is
read as is; otherwise the data is generated from a fixed seed and written
there. The seed alone is not enough: NumPy does not guarantee the same
random stream across versions. Delete bank.csv to regenerate.
"""

from pathlib import Path

import numpy as np

CSV = Path(__file__).with_name("bank.csv")
SEED = 0
N = 500


def generate(seed=SEED, n=N):
    rng = np.random.default_rng(seed)
    income = np.round(rng.lognormal(mean=np.log(40), sigma=0.45, size=n), 1)
    missed = rng.poisson(lam=1.2, size=n).astype(float)

    logit = -1.6 - 0.05 * (income - 40) + 0.9 * missed - 0.02 * (income - 40) * (missed > 2)
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(float)

    return np.column_stack([income, missed]), y


def load():
    if not CSV.exists():
        X, y = generate()
        with open(CSV, "w", newline="\n") as f:  # LF, as .gitattributes wants
            np.savetxt(
                f,
                np.column_stack([X, y]),
                fmt=["%.1f", "%d", "%d"],
                delimiter=",",
                header="income,missed,defaulted",
                comments="",
            )
    data = np.loadtxt(CSV, delimiter=",", skiprows=1)
    assert data.shape == (N, 3), f"{CSV} has shape {data.shape}, expected ({N}, 3)"
    return data[:, :2], data[:, 2]


X, y = load()
income, missed = X[:, 0], X[:, 1]

if __name__ == "__main__":
    print("X shape", X.shape, " default rate", y.mean().round(3))
    print("income  min/median/max", income.min(), np.median(income), income.max())
    print("missed  value counts", dict(zip(*np.unique(missed, return_counts=True), strict=True)))
