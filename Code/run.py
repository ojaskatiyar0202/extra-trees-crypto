import warnings

import numpy as np

import features
import model

warnings.filterwarnings("ignore")

PRICES = "data/prices.csv"
FUNDING = "data/panel.csv"
MODELS = ["linear", "ridge", "rforest", "xtrees"]

d = features.load(PRICES, FUNDING)
d, big = features.large_set(d)
d, small = features.small_set(d)
d = features.target(d)

big = [c for c in big if d[c].isna().mean() < 0.5]
d = d.dropna(subset=big + small + ["y"]).reset_index(drop=True)

print(f"{len(d)} rows, {d['symbol'].nunique()} symbols, {d['date'].nunique()} days")
print(f"{len(small)} small features, {len(big)} large features")
print(f"{d['date'].min().date()} to {d['date'].max().date()}\n")

print(f"{'model':9s} {'IC small':>10} {'IC large':>10}")
print("-" * 31)
ic = {}
for name in MODELS:
    s = model.evaluate(d, small, name)
    b = model.evaluate(d, big, name)
    ic[name] = (s, b)
    print(f"{name:9s} {s.mean():>+10.4f} {b.mean():>+10.4f}", flush=True)

lin_s, lin_b = ic["linear"]
xt_s, xt_b = ic["xtrees"]
gap = xt_b - lin_b
print("\ncrossover, same rows and splits, only the feature count changes")
print(f"  small: linear {lin_s.mean():+.4f}  xtrees {xt_s.mean():+.4f}  "
      f"xtrees wins {int((xt_s > lin_s).sum())}/{len(xt_s)}")
print(f"  large: linear {lin_b.mean():+.4f}  xtrees {xt_b.mean():+.4f}  "
      f"xtrees wins {int((xt_b > lin_b).sum())}/{len(xt_b)}")
print(f"  gap on large: {gap.mean():+.4f}, sd {gap.std():.4f}, "
      f"t = {gap.mean() / (gap.std() / np.sqrt(len(gap))):+.2f}")

rf_b = ic["rforest"][1]
print("\nsplit rule, both on the large set")
print(f"  rforest (best threshold)   {rf_b.mean():+.4f}")
print(f"  xtrees  (random threshold) {xt_b.mean():+.4f}")

print("\ndecile spread, bp/day")
print(f"  xtrees on large  {model.decile_spread(d, big, 'xtrees'):.1f}")
print(f"  linear on small  {model.decile_spread(d, small, 'linear'):.1f}")