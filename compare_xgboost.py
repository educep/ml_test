"""Check toy_xgb.py against the real library, tree by tree.

If the toy implements the paper the way the library does, both produce the
same margins (log-odds) for every customer, train and test, up to float noise.
Each config changes one knob so a mismatch points at the piece of math that
differs.
"""

import numpy as np
import xgboost as xgb

from bank import X, y
from toy_xgb import fit

tr = np.arange(400)
dtrain, dall = xgb.DMatrix(X[tr], label=y[tr]), xgb.DMatrix(X)

configs = [
    # library defaults, depth 2
    dict(lam=1.0, gamma=0.0, eta=0.3, min_child=1.0, depth=2, n_trees=50),
    # No regularisation: trees split nodes of weight 0.2, where two features can separate the
    # training rows identically (equal gain). Toy and library break the tie differently, so
    # train margins match but a few test customers land in different leaves.
    dict(lam=0.0, gamma=0.0, eta=0.3, min_child=0.0, depth=2, n_trees=50),
    # stronger lambda, deeper trees
    dict(lam=5.0, gamma=0.0, eta=0.1, min_child=1.0, depth=3, n_trees=100),
    # gamma active, then strong
    dict(lam=1.0, gamma=0.5, eta=0.3, min_child=1.0, depth=3, n_trees=50),
    dict(lam=1.0, gamma=2.0, eta=0.3, min_child=1.0, depth=3, n_trees=50),
    # min_child_weight active
    dict(lam=1.0, gamma=0.0, eta=0.3, min_child=5.0, depth=3, n_trees=50),
]

for c in configs:
    params = {
        "objective": "binary:logistic",
        "tree_method": "exact",  # Alg. 1; the library default is "hist"
        "max_depth": c["depth"],
        "eta": c["eta"],
        "lambda": c["lam"],
        "gamma": c["gamma"],
        "min_child_weight": c["min_child"],
        "base_score": y[tr].mean(),  # otherwise the library estimates its own intercept
        "nthread": 1,
    }
    bst = xgb.train(params, dtrain, num_boost_round=c["n_trees"])
    lib = bst.predict(dall, output_margin=True)
    toy = fit(**c)
    d = np.abs(toy - lib)
    acc_toy, acc_lib = (int(((m > 0) == y)[400:].sum()) for m in (toy, lib))
    print(
        f"{c}\n   max |toy - lib|  train {d[tr].max():.1e}  test {d[400:].max():.1e}"
        f"   test acc toy {acc_toy} lib {acc_lib}"
    )
