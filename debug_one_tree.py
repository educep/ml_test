"""Debug helper: one tree, toy vs library, with the library's tree dump."""

import sys

import numpy as np
import xgboost as xgb

from bank import X, y
from toy_xgb import fit

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
tr = np.arange(400)
p = {
    "objective": "binary:logistic",
    "tree_method": "exact",
    "max_depth": 2,
    "eta": 0.3,
    "lambda": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "base_score": y[tr].mean(),
    "nthread": 1,
}
bst = xgb.train(p, xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=n)
print(bst.get_dump(with_stats=True)[-1])
lib = bst.predict(xgb.DMatrix(X), output_margin=True)
toy = fit(lam=1.0, gamma=0.0, eta=0.3, min_child=1.0, depth=2, n_trees=n)
print("unique lib", np.unique(lib.round(5)))
print("unique toy", np.unique(toy.round(5)))
print("max diff", np.abs(toy - lib).max())
