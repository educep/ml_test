# ml_test: XGBoost from 16 lines up

Learning repo. Start from a 16-line gradient-boosting toy, then rebuild it with
the math of the XGBoost paper (Chen & Guestrin, KDD 2016) until it reproduces
the real `xgboost` library to about 1e-6.

```bash
uv run python bank.py             # look at the data
uv run python toy_gbm.py          # the 16-line toy, readable version
uv run python toy_xgb.py          # same skeleton, XGBoost math
uv run python compare_xgboost.py  # toy_xgb vs the library, 6 configs
uv run python debug_one_tree.py 3 # library tree dump after N rounds + diff
```

Setup after cloning: `uv sync && uv run pre-commit install`. Every commit then runs
`ruff check --fix` and `ruff format` (config in `pyproject.toml`). Run them by hand with
`uv run pre-commit run --all-files`.

## Files

| File | What it is |
|---|---|
| `bank.py` | Synthetic data: 500 customers, `income` (k EUR) and `missed` (missed payments), `y` = defaulted. Rows 0-399 train, 400-499 test. |
| `toy_gbm.py` | Readable rewrite of the 16-line toy, same output. Squared-error gradient boosting, depth-2 trees, 200 trees, learning rate 0.01. |
| `original/toy_gbm_16_lines.py` | The 16 lines exactly as transcribed, excluded from ruff. Run with `PYTHONPATH=. uv run python original/toy_gbm_16_lines.py`. |
| `toy_xgb.py` | The same `grow()` + loop, with log-loss, gradients/hessians, lambda, gamma, min_child_weight, float32 thresholds. |
| `compare_xgboost.py` | Trains the library with `tree_method="exact"` and the same knobs, compares margins on all 500 rows. |
| `debug_one_tree.py` | Prints the library's last tree (`get_dump(with_stats=True)`) and the toy vs library diff. |

## Reading the 16-line original

```
p              current prediction for every customer (starts at the base rate 0.385)
e = (y - p)[i] residuals of the customers in this node: what the model still gets wrong
grow()         fits ONE small tree to those residuals
  for f, t     try every feature and every threshold ("is income > t?")
  sse(L)+sse(R) how well two constant predictions explain the residuals; keep the best
  d < 1        depth limit reached: the leaf predicts the mean residual
  np.where     the tree's output for ALL 500 rows (train and test), so the
               loop can add it to p directly
p += .01*grow() add a small step of the new tree: shrinkage / learning rate
```

`toy_gbm.py` is the same code with names (`residual`, `yes_branch`, `LEARNING_RATE`) and
`p` passed as an argument instead of read from a global.

Result: 78/100 correct on the test rows, vs 57 for "always predict no default".

## From toy_gbm to XGBoost

The toy already *is* XGBoost for one special case. With squared loss,
`g = p - y = -e` and `h = 1`, so:

- `sse(L) + sse(R) = sum(e^2) - G_L^2/n_L - G_R^2/n_R`, so minimising it is
  maximising `G_L^2/H_L + G_R^2/H_R`: the paper's Eq. (7) with `lambda = 0`.
- the leaf `e.mean()` is `-G/H`: the paper's Eq. (5) with `lambda = 0`.

So `toy_gbm.py` = XGBoost with squared error, `lambda=0`, `gamma=0`,
`min_child_weight=0`, `eta=0.01`, `max_depth=2`. What the paper adds:

| Knob | Paper | Effect | In toy_xgb.py |
|---|---|---|---|
| any convex loss | Sec. 2.2, second-order Taylor | needs only `g_i`, `h_i` per row | `g, h = q - y, q*(1-q)` (log-loss) |
| `lambda` (L2 on leaf weights) | Eq. (2), (5) | shrinks each leaf toward 0; small nodes shrink most | `-G / (H + lam)` |
| `gamma` (min split loss) | Eq. (2), (7) | a split must gain more than gamma, pruned bottom-up | `best <= gamma` |
| `min_child_weight` | not in the paper | a child needs `sum(h) >= value`; with log-loss h = p(1-p), so confident nodes weigh little | `HL >= min_child` |
| `eta` | Sec. 2.3 | shrinkage of each tree | `m + eta * grow(...)` |
| exact greedy split | Alg. 1 | sort once per feature, cumulative sums of g and h | `np.cumsum` |
| `colsample_*`, `subsample` | Sec. 2.3 | random features / rows per tree | not implemented |
| missing values | Alg. 3 | learns a default direction per split | not implemented |
| `hist` / `approx` | Alg. 2, Sec. 3.3 | candidate thresholds from (weighted) quantiles | not implemented |

## What the comparison taught (verified, not from docs)

1. **The library's `gamma` is 2x the paper's gamma.** Eq. (7) has a factor 1/2;
   the library's gain does not. The `gain=` numbers in `get_dump(with_stats=True)`
   are on the library's scale, so to set `gamma`, read the gains of the splits you
   want to kill in the dump and pick a value just above them.
2. **Thresholds are float32 midpoints.** The split is `x < (a + b)/2` computed in
   float32. A test customer whose value sits exactly on a midpoint can go the
   other way in a float64 reimplementation.
3. **Equal-gain ties are broken arbitrarily.** With `lambda=0, min_child_weight=0`
   trees split nodes weighing 0.2, where two different rules separate the
   training rows identically. Same training fit, different test predictions.
   `min_child_weight` and `lambda` remove these degenerate splits.
4. **The library default is not the paper's exact algorithm:** `tree_method`
   defaults to `hist`. `compare_xgboost.py` forces `exact`, and sets
   `base_score` so both start from the same intercept.

## Next steps

- Turn on `subsample` / `colsample_bytree` in the toy, compare again (needs the same RNG, so compare distributions, not exact margins).
- Add missing values to `bank.py` and implement Alg. 3's default direction.
- Swap `exact` for `hist` in the library and watch how far it drifts from the toy.
- Plot test log-loss vs number of trees for several `eta`: shrinkage trades trees for generalisation.
