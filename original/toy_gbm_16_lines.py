import numpy as np
from bank import X, y              # 500 customers
sse = lambda e: ((e - e.mean()) ** 2).sum()
def grow(i=np.arange(400), d=2):   # small tree
    best, F, e = np.inf, None, (y - p)[i]
    for f in (0, 1):               # income, missed
        for t in np.unique(X[i, f])[:-1]:
            L = X[i, f] > t        # answer is yes
            g = sse(e[L]) + sse(e[~L])
            if g < best: best, F, T, S = g, f, t, L
    if d < 1 or F is None: return e.mean()
    a, b = grow(i[S], d - 1), grow(i[~S], d - 1)
    return np.where(X[:, F] > T, a, b)
print(p := y[:400].mean(), np.unique(grow()))
for k in range(200): p += .01 * grow()  # 200 trees
print(((p > .5) == y)[400:].sum())
