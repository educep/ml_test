# How to learn XGBoost with this repo

**Don't read the paper cover to cover, and don't just run the code either. Alternate between them.** The math you need is about 1.5 pages (Section 2, pp. 2-3). Most of the rest of the paper is about speed: cache access, disk-based training, distributed computing. Start with the code, then read Section 2 with the code open beside it.

## Step 1: Feel boosting with the simple toy (30 min, code only)

Run `toy_gbm.py`, then break it on purpose:
- Set `LEARNING_RATE` to 0.01, then 0.1, then 1.0, and watch the test accuracy.
- Set `N_TREES` to 10, 200, 2000.
- Add one `print` in the loop to see how much the residuals shrink after each tree.

**Goal:** be able to say in one sentence "each tree fits what the model still gets wrong, and we add only a small step of it." Everything else is a refinement of that.

## Step 2: Read Section 2 of the paper with `toy_xgb.py` beside it (1-1.5 h)

Read pp. 2-3, Eq. (1) to (7). Every equation has a matching line in the code; the docstring at the top lists them. **Do the derivation by hand with a pen:**
1. Write the loss as a second-order Taylor expansion (a quadratic approximation). For each row it's just g·w + ½h·w².
2. Group the rows by leaf. For each leaf you get a quadratic in w, and setting its derivative to zero gives w* = −G/(H+λ), Eq. (5).
3. Put w* back in. That gives the tree's score, Eq. (6), and the split gain, Eq. (7).
4. Plug in squared loss (g = −residual, h = 1) and λ = 0. You get back exactly the `sse` rule from the toy.

If you can redo those four steps without looking, you understand XGBoost's core. The key idea: **the loss only reaches a tree through two numbers per row, g and h.** That's why any loss function works.

## Step 3: Play with the settings (1-2 h, code)

This is where "using it correctly" comes from:
- **Read a tree dump.** Run `uv run python debug_one_tree.py 5` and read `gain` and `cover` on each node. `cover` is the sum of h in that node, not the number of rows.
- **`min_child_weight` with log-loss.** For log-loss, h = p(1−p). Rows the model is already confident about (p near 0 or 1) weigh almost nothing. So `min_child_weight=5` does **not** mean "at least 5 rows". It means "at least 5 rows' worth of uncertainty". That changes how you set it on imbalanced data.
- **`gamma`.** Read the gains in the dump, pick a `gamma` just above the gains of the splits you want to remove, and check that they disappear. Remember the library's gamma is 2× the paper's γ.
- **`lambda`.** Raise it and watch the leaf values shrink. Small leaves shrink the most, because λ is large compared with their H.

## Step 4: Section 3 of the paper (45 min, reading)

pp. 3-5 cover two things you'll meet in real work:
- **Approximate split finding.** The candidate thresholds are quantiles weighted by h (p. 4 explains why h acts as a weight). This is the family the library's default `hist` method belongs to.
- **Missing values (Alg. 3).** Each split learns which way missing values go by default. Exercise: put NaNs into `bank.py` (then delete `bank.csv` so the data is regenerated) and look for `missing=` in the dump.

Skip Section 4 (systems design) and the appendix unless you're curious.

## Step 5: Apply it to a real model

Take an XGBoost model you already use or maintain at work. Once you've done steps 1-4, read its tree dump and its parameters with fresh eyes. Check what `cover` looks like given how imbalanced the classes are, whether `min_child_weight` makes sense, and whether `tree_method` was chosen on purpose.

**Time budget:** about 4-5 hours over 2-3 sessions. Steps 2 and 3 are where the real understanding happens.
