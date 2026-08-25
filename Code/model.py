import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLITS = [0.5, 0.65, 0.8]
SEEDS = [0, 1]
DEPTH = 10
LEAF = 200
NTREE = 80
ALPHA = 1000


def build(name, seed):
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    if name == "rforest":
        return RandomForestRegressor(n_estimators=NTREE, max_depth=DEPTH,
                                     min_samples_leaf=LEAF, n_jobs=-1,
                                     random_state=seed)
    if name == "xtrees":
        return ExtraTreesRegressor(n_estimators=NTREE, max_depth=DEPTH,
                                   min_samples_leaf=LEAF, n_jobs=-1,
                                   random_state=seed)
    raise ValueError(name)


def rank_ic(te, col):
    s = te.groupby("date").apply(
        lambda x: x[col].corr(x["y"], method="spearman"), include_groups=False)
    return float(s.mean())


def evaluate(d, f, name):
    out = []
    for q in SPLITS:
        cut = d["date"].quantile(q)
        tr = d[d["date"] <= cut]
        te = d[d["date"] > cut].copy()
        for seed in SEEDS:
            mod = build(name, seed).fit(tr[f], tr["y"])
            te["p"] = mod.predict(te[f])
            out.append(rank_ic(te, "p"))
    return np.array(out)


def decile_spread(d, f, name, q=0.7, seed=0, n=10):
    import pandas as pd
    cut = d["date"].quantile(q)
    tr = d[d["date"] <= cut]
    te = d[d["date"] > cut].copy()
    mod = build(name, seed).fit(tr[f], tr["y"])
    te["p"] = mod.predict(te[f])
    dec = te.groupby("date")["p"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n, labels=False))
    return (te[dec == n - 1]["y"].mean() - te[dec == 0]["y"].mean()) * 1e4