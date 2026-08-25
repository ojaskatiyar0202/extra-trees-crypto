import numpy as np
import pandas as pd

RANK_N = 45
EPS = 1e-9


def load(px_path, fd_path):
    px = pd.read_csv(px_path, parse_dates=["date"])
    fd = pd.read_csv(fd_path, parse_dates=["date"])[["symbol", "date", "pay", "n"]]
    d = px.merge(fd, on=["symbol", "date"])
    return d.sort_values(["symbol", "date"]).reset_index(drop=True)


def small_set(d):
    g = d.groupby("symbol")
    d["ret1"] = g["close"].pct_change()
    d["ret5"] = g["close"].pct_change(5)
    d["ret20"] = g["close"].pct_change(20)
    d["pay1"] = d["pay"]
    d["pay5"] = g["pay"].transform(lambda s: s.rolling(5).mean())
    d["dv"] = d["close"] * d["volume"]
    d["logdv"] = np.log(d["dv"].clip(lower=1))

    g = d.groupby("symbol")
    d["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
    d["dvchg20"] = d["dv"] / g["dv"].transform(lambda s: s.rolling(20).mean())

    f = ["ret1", "ret5", "ret20", "pay1", "pay5", "logdv", "vol20", "dvchg20"]
    return d, f


def large_set(d):
    d["dv"] = d["close"] * d["volume"]
    d["logdv"] = np.log(d["dv"].clip(lower=1))
    g = d.groupby("symbol")
    d["ret"] = g["close"].pct_change()
    f = ["logdv", "pay"]

    for h in [1, 2, 3, 5, 10, 15, 20, 30, 60, 90]:
        c = f"ret{h}"
        d[c] = g["close"].pct_change(h)
        f.append(c)

    d["age"] = g.cumcount()
    f.append("age")

    for w in [10, 20, 60]:
        hi = g["close"].transform(lambda s, w=w: s.rolling(w).max())
        lo = g["close"].transform(lambda s, w=w: s.rolling(w).min())
        d[f"pos{w}"] = (d["close"] - lo) / (hi - lo + EPS)
        d[f"dd{w}"] = d["close"] / hi - 1
        f += [f"pos{w}", f"dd{w}"]

    for w in [3, 5, 10, 20, 60]:
        d[f"pay{w}"] = g["pay"].transform(lambda s, w=w: s.rolling(w).mean())
        d[f"payv{w}"] = g["pay"].transform(lambda s, w=w: s.rolling(w).std())
        f += [f"pay{w}", f"payv{w}"]

    for k in [1, 2, 3, 5]:
        c = f"paylag{k}"
        d[c] = g["pay"].shift(k)
        f.append(c)

    d["paysign"] = np.sign(d["pay"])
    d["payflip"] = g["pay"].transform(lambda s: np.sign(s).diff().abs() / 2)
    d["n_pay"] = d["n"]
    f += ["paysign", "payflip", "n_pay"]

    for w in [5, 10, 20, 60]:
        d[f"dvchg{w}"] = d["dv"] / g["dv"].transform(lambda s, w=w: s.rolling(w).mean())
        d[f"dvvol{w}"] = g["dv"].transform(
            lambda s, w=w: s.rolling(w).std() / s.rolling(w).mean())
        f += [f"dvchg{w}", f"dvvol{w}"]

    # regroup so the rolling stats below can see ret and amihud
    g = d.groupby("symbol")

    for w in [5, 10, 20, 30, 60]:
        for stat in ["std", "skew", "kurt"]:
            c = f"{stat}{w}"
            d[c] = g["ret"].transform(
                lambda s, w=w, stat=stat: getattr(s.rolling(w), stat)())
            f.append(c)

    d["amihud"] = d["ret"].abs() / d["dv"].clip(lower=1) * 1e9
    f.append("amihud")
    g = d.groupby("symbol")
    for w in [10, 30]:
        c = f"amihud{w}"
        d[c] = g["amihud"].transform(lambda s, w=w: s.rolling(w).mean())
        f.append(c)

    mkt = d.groupby("date")["ret"].mean().rename("mkt")
    d = d.merge(mkt, on="date")
    d["mktvol"] = d["date"].map(mkt.rolling(20).std())
    d["excess"] = d["ret"] - d["mkt"]
    f += ["mkt", "mktvol", "excess"]

    # cross-sectional percentile rank of the first RANK_N features, per day
    base = [c for c in f if c not in ("mkt", "mktvol")]
    for c in base[:RANK_N]:
        r = f"r_{c}"
        d[r] = d.groupby("date")[c].rank(pct=True)
        f.append(r)

    d["pay_x_dv"] = d["pay"] * d["logdv"]
    d["pay_x_vol"] = d["pay"] * d["std20"]
    d["mom_x_vol"] = d["ret20"] / (d["std20"] + EPS)
    d["rev_x_liq"] = d["ret"] * d["amihud"]
    f += ["pay_x_dv", "pay_x_vol", "mom_x_vol", "rev_x_liq"]

    return d, f


def target(d, min_dv=1e6, min_names=100, clip=0.005):
    d["nxt"] = d.groupby("symbol")["close"].pct_change().shift(-1)
    d["y"] = d["nxt"] - d.groupby("date")["nxt"].transform("mean")
    d["y"] = d["y"].clip(d["y"].quantile(clip), d["y"].quantile(1 - clip))
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[d["dv"] > min_dv]
    return d[d.groupby("date")["symbol"].transform("size") > min_names]