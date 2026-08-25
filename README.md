# The value of non-parametric ML methods in cross-sectional crypto returns

We rank cryptocurrency perpetual futures by expected next-day relative return and
ask a narrower question than whether the ranking makes money: does the number of
features decide whether flexible models beat linear ones? We fit the same four
models twice, once on eight features and once on 114, holding the rows, the splits
and the random seeds fixed, and find that the winner changes. The features are the standard cross-sectional families: recent returns, volatility, funding, liquidity and where the price sits in its recent range, measured at one horizon each in the small set and at several in the large.

# Setup

Each day we observe around 500 perpetual futures on Binance. We predict, for each
one, its next-day return minus the average next-day return across the whole
cross-section. Subtracting the average is what makes this a relative bet. Crypto is
dominated by a single common move, and when Bitcoin falls almost everything falls
with it; that move is far larger than anything specific to an individual contract.
Removing it leaves the part that might be forecastable, and it makes the resulting
portfolio market neutral by construction rather than by hedging.

The trade implied by a ranking is to buy the top decile, short the bottom decile and hold for a day. Perpetuals suit this, since a short position is a contract rather than a borrowed asset, so both legs are equally available on every contract in the panel. 

Return rank persistence from one day to the next is −0.049, essentially zero, so today's ordering tells you nothing about tomorrow's. Any forecast we produce is therefore genuinely new information rather than persistence in disguise.

## Parametric and non-parametric models

A linear model assumes each feature contributes independently and always in the same
direction. If momentum has a coefficient of 0.3 then it adds 0.3 per unit of
momentum whether the contract is liquid or illiquid, calm or volatile. That is a
strong assumption, and it is also efficient, since only one number per feature has
to be estimated.

A tree assumes nothing about the functional form. It partitions the sample on one
feature at a time, recursively, so each condition is evaluated only on the subsample
its parent produced. A leaf reached by three successive conditions therefore holds
the rows satisfying all three conditions jointly, and its mean outcome doesn't rely on any feature marginally. The cost is
variance: refitting on a resample of the same data relocates the splits, which is
why eighty trees are averaged rather than one being used alone.
We use four models arranged so that each pair differs in exactly one design choice.

|                 | parametric      | non-parametric               |
|-----------------|-----------------|------------------------------|
| plain           | linear          | random forest, best split    |
| regularised     | ridge           | extra trees, random split    |

Left to right isolates whether flexibility helps. Top to bottom isolates
regularisation, in the sense that ridge penalises large coefficients and extra trees
randomises where its splits fall.

## How a tree chooses a split

At each node the algorithm chooses one feature and one threshold on that feature.
The cost of a candidate split is the total squared deviation of the outcome within
each of the two resulting groups,

```
cost(t) = sum over left  (y - ybar_left)^2
        + sum over right (y - ybar_right)^2
```

and the split with the lowest cost is kept. Each leaf then predicts the mean of the
rows that reached it, which follows from the loss rather than being a convention;
differentiating the same expression with respect to a single constant gives the mean
as the minimiser.

The two tree models differ in how the threshold is arrived at. Random forest sorts
the rows by the feature and evaluates every midpoint between adjacent values. Our
training block has 79,433 rows and a typical feature takes around 61,000 distinct
values, so with 114 features that is roughly seven million evaluations at the root
node alone, repeated at every node below. Extra trees draws one threshold per
feature uniformly between that feature's minimum and maximum at the node, giving 114
candidates, and keeps the best of those. It still chooses the feature by comparison;
it doesn't asks whether the cut on that feature was the best available one.

Choosing a worse split deliberately sounds counter-intuitive but is done with the intention to minimise variance.  An optimised threshold is fitted to the sample it was chosen on,
including the noise in that sample, and when the signal is as weak as next-day
returns most of what the search is minimising against is noise. A randomly drawn
threshold cannot overfit in that way. Each individual tree is worse, which is to say
more biased, but the eighty trees end up far less similar to one another, so
averaging cancels more of their error. Mentch and Zhou (2020) show that split
randomisation reduces the effective degrees of freedom of a forest and that forests
beat plain bagging in low and medium signal-to-noise settings while bagging wins
when the signal is strong; their conclusion is that forests win on real data because
real data is noisy.

## Parameter tuning

The **learned parameters** are what fitting determines. For the linear models that
is 114 coefficients; for the trees it is the feature and threshold at every node
together with the mean at every leaf. All of them are computed from the training
rows by minimising squared error.

The **free hyperparameters** are the ones that control how much flexibility the fit
is allowed. For the trees these are `max_depth` and `min_samples_leaf`; for ridge it
is `alpha`. They cannot be chosen on training error, because a deeper tree always
fits the training rows better and a smaller penalty always fits them better, so
training error would drive depth upwards without limit. Choosing them properly needs
a third block of data that the fit has not seen.

The **fixed hyperparameters** are set by convention or by compute budget rather than
by search. We use 80 trees, since more trees is weakly better and cannot overfit,
and we leave the number of features considered at each node at all 114, which keeps
the comparison between the two tree models clean.

We did not tune. Depth is fixed at 10, minimum leaf size at 200 and alpha at 1000,
all chosen by judgement before any modelling. This means the results below describe
one reasonable configuration rather than each model's best, and someone could
reasonably ask whether a different alpha would rescue the linear model on the large
feature set. We cannot answer that. It also means no validation block was required,
so the data is split two ways rather than three, and the test block is a genuine
holdout in the sense that no decision anywhere in the project was made by looking at
it.

## Splits and seeds

The split is by date and never at random. A random split would place a contract's
15 June row in training and its 14 June row in test, and since adjacent days are
highly correlated that is close to handing the model the answer. Splitting by date
guarantees that every training row precedes every test row.

We cut at the 50th, 65th and 80th percentile of dates, giving three train and test
pairs from the same panel, and run each at two random seeds. Both tree models are
stochastic, so a single seed tells us nothing about stability. This is not a
hypothetical concern: an earlier version of this work reported a gradient boosting
result from seed zero that looked strong, and across five seeds the same
configuration ranged from 0.0062 to 0.0315, with seed zero being the best of the
five. Linear and ridge are deterministic and give identical numbers at both seeds.

Six fits per model per feature set gives a mean, a spread, and a paired comparison
against another model on matched splits and seeds.

## Metrics

Three metrics do three different jobs and it is easy to conflate them.

Inside the fit, both model families minimise squared error on the demeaned return.
That is the loss function, and it is not what we report.

For reporting we use **rank information coefficient**. On each test day we rank the
contracts by prediction, rank them by realised outcome, take the Spearman
correlation of the two orderings, and average across days. Ranks rather than values,
because the trade only uses the ordering and never the magnitude of a prediction;
saying plus 0.4% instead of plus 4% changes nothing as long as the ordering holds.
Per day rather than pooled, because pooling would let a day with 500 contracts
outweigh a day with 150, and we trade every day regardless of how good that day
happens to be. An IC above 0.02 is usually treated as usable in cross-sectional
equity work, and nobody reports 0.3, since returns are almost entirely noise.

We also report the **decile spread**, which is the realised return of the actual
trade in basis points per day. Each test day we sort by prediction into ten buckets,
and take the mean realised outcome of the top bucket minus the bottom. Rank IC uses
the whole cross-section; the decile spread uses only the hundred contracts that
would actually be held. The two can disagree, and here they do.

## Features

The small set takes one feature from each family. The large set widens every family
and adds cross-sectional ranks and a few products.

| family | small set | large set |
|---|---|---|
| returns | `ret1`, `ret5`, `ret20` | `pct_change(h)` for h in 1, 2, 3, 5, 10, 15, 20, 30, 60, 90 |
| risk shape | `vol20` | rolling std, skew and kurtosis of `ret1` at 5, 10, 20, 30, 60 |
| funding | `pay1`, `pay5` | rolling mean and std of `pay` at 3, 5, 10, 20, 60; lags 1, 2, 3, 5; sign; sign flips; payments per day |
| liquidity | `logdv`, `dvchg20` | `log(close x volume)`; `dv` over its own rolling mean at 5, 10, 20, 60; rolling std of `dv` over its mean; Amihud `abs(ret)/dv`, raw and smoothed at 10 and 30 |
| price location | — | position within rolling high-low range at 10, 20, 60; drawdown from rolling high at each; days since listing |
| market state | — | cross-sectional mean return; its 20-day rolling std; return minus that mean |
| ranks | — | cross-sectional percentile rank, within each day, of the features above |
| interactions | — | `pay x logdv`, `pay x std20`, `ret20 / std20`, `ret x amihud` |

Every feature is deliberately scale free. Bitcoin trades near 65,000 and small
contracts near 0.02, so a raw price tells the model which contract it is looking at
rather than what is happening to it. Returns, log volume, ratios to a contract's own
history and cross-sectional ranks are all comparable across the panel, which is what
allows one model fitted across 543 contracts to learn a relationship that holds for
all of them.

The target is winsorised at the 0.5% and 99.5% tails. Crypto produces days where a
contract returns several hundred percent, and one such row dominates a squared-error
fit. Adding this roughly doubled every model's IC, which is itself evidence that
outliers rather than non-linearity were the binding problem before.

We keep contract-days with more than one million dollars of volume and days with
more than 100 surviving contracts. The first threshold matters a great deal and is
revisited below. The second is because ranking into deciles needs breadth, and with
twenty contracts a decile is two names and the spread is noise.

## Data

Funding rates come from the Binance USD-margined futures endpoint,

```
fapi.binance.com/fapi/v1/fundingRate
```

and daily prices and volumes from the monthly kline archive,

```
data.binance.vision/?prefix=data/futures/um/monthly/klines/
```

Merging the two and dropping rows where the rolling windows have not filled leaves
112,905 contract-days across 543 perpetuals, from 30 October 2025 to 30 July 2026.
Funding is available to 24 August 2026 but prices stop at the end of July, because
the monthly kline file for August is not published until the month closes.

The universe was taken from contracts listed at the time of download, so perpetuals
delisted before October 2025 never enter the panel. Within the window no contract's
price series terminates early, so there is no truncation bias of the kind that
hides a final collapse, but the universe selection remains.

## Results

| model | IC, 8 features | IC, 114 features |
|---|---:|---:|
| linear | +0.0605 | +0.0303 |
| ridge | +0.0610 | +0.0313 |
| random forest | +0.0269 | +0.0399 |
| extra trees | +0.0388 | +0.0489 |

**Discussion.** The two parametric models get worse as features are added and the
two non-parametric models get better. Linear falls from 0.0605 to 0.0303 and ridge
from 0.0610 to 0.0313, while random forest rises from 0.0269 to 0.0399 and extra
trees from 0.0388 to 0.0489. The ordering of the four models is reversed between the
two columns.

Ridge barely improves on linear in either column, 0.0610 against 0.0605 and 0.0313
against 0.0303. This is informative. If the linear degradation were driven by
collinearity among near-duplicate features, which the large set has in quantity
given returns at ten horizons and volatility at five windows, then shrinking the
coefficients should have recovered a good deal of it. It does not, which points at
ordinary overfitting from parameter count rather than at coefficient instability.

| comparison | linear | extra trees | extra trees wins |
|---|---:|---:|---:|
| 8 features | +0.0605 | +0.0388 | 0 of 6 |
| 114 features | +0.0303 | +0.0489 | 6 of 6 |

**Discussion.** The paired differences on the large set have a mean of +0.0186 and a
standard deviation of 0.0030 across the six fits, giving a t-statistic of 15.29. The
six fits are not independent, since the three splits share overlapping training data
and the two seeds share everything else, so the effective sample is smaller than six
and this figure should be read as a summary of consistency rather than as a p-value.
What it does establish is that the reversal is not an artefact of one split or one
seed, which was the failure mode we were guarding against.

| split rule | IC, 114 features |
|---|---:|
| random forest, best threshold | +0.0399 |
| extra trees, random threshold | +0.0489 |

**Discussion.** Both models bag eighty trees to the same depth and the same minimum
leaf size, and the visible difference between them is whether the threshold at each
node is searched for or drawn at random. Random splitting is 23% better here, which
is the direction Mentch and Zhou predict for low signal-to-noise data.

There is a confound. Scikit-learn's `RandomForestRegressor` bootstraps rows by
default and its `ExtraTreesRegressor` does not, so our two models differ in two
respects rather than one, and the gap cannot be attributed to the split rule alone.
Setting `bootstrap=True` on the extra trees would isolate it. We report the number
as suggestive rather than as a clean mechanism test.

| configuration | decile spread, bp/day |
|---|---:|
| extra trees, 114 features | 125.9 |
| linear, 8 features | 76.8 |

**Discussion.** The two metrics disagree. Linear on eight features has the higher
rank IC at 0.0605 against 0.0489, and the lower decile spread at 76.8 against 125.9.
They measure different regions of the cross-section. Rank IC uses all 500 contracts,
so a model that orders the middle of the distribution well scores highly. The decile
spread uses only the extremes. A tree isolates extreme regions with hard splits,
which costs it accuracy in the middle and gains it accuracy in the tails, and the
tails are what the trade holds. We saw the same divergence between R-squared and IC
in the funding work, and the lesson is the same: the metric has to match what the
strategy consumes.

### Liquidity

A decile spread of 126 basis points a day is around 360% annualised, which is not a
plausible return and is worth investigating rather than reporting. We refit extra
trees on the large feature set at progressively higher volume floors.

| minimum daily volume | contracts per day | mean, bp/day | std, bp/day | Sharpe |
|---|---:|---:|---:|---:|
| $1m | 435 | 112.6 | 192.0 | 11.21 |
| $10m | 133 | 112.5 | 388.8 | 5.53 |
| $50m | 59 | 17.2 | 651.7 | 0.50 |

**Discussion.** The edge is an illiquidity artefact. Restricting to contracts
trading over fifty million dollars a day cuts the mean return from 113 basis points
to 17 and the Sharpe ratio from 11.2 to 0.5. Nothing tradeable at size survives.

This is what the model should be expected to do. Amihud illiquidity is among the
features precisely because illiquidity is a known return predictor, and assets that
are hard to trade earn higher expected returns as compensation for that difficulty.
The model loads on illiquidity because that is where the unexploited variation sits,
but the return is payment for being unable to trade, so it cannot be harvested. A
Sharpe ratio of 11 was the signal that this had happened.

The fifty million dollar row should be read with care. Fifty-nine contracts a day
means deciles of about six names each, which is why the standard deviation is 652
basis points, and the Sharpe ratio of 0.50 is correspondingly imprecise. The
direction across the three rows is unambiguous; the level of the last one is not.

The comparison between models is unaffected by any of this, since it is a relative
statement on identical data.

## Files

```
code/features.py    loading, both feature sets, target construction and filters
code/model.py       the four estimators, rank IC, evaluation across splits and seeds
code/run.py         builds the panel, runs the comparison, prints the tables
```

## Running it

```bash
pip install -r requirements.txt
python code/run.py
```

Run from the repository root, since the data paths are relative to it. Expect around
ten minutes, almost all of it in the random forest fits, which search every
candidate threshold across 114 features at every node.

## Limitations

Hyperparameters were fixed before modelling rather than tuned. Depth 10, minimum
leaf 200 and alpha 1000 were chosen by judgement, so the comparison is between four
models at one configuration each rather than between four models at their best. The
gap on the large feature set is wide relative to its variation across fits, which
makes a reversal under tuning unlikely, but this was not tested.

The split rule comparison between random forest and extra trees is confounded by the
bootstrap default, as described above.

The t-statistic of 15.29 is computed across six fits that share training data and
therefore are not independent observations.

The panel is fitted once on the earliest 70% of dates and applied to the remainder,
rather than refitted on a rolling basis and walked forward. A rolling refit is closer
to how a strategy would run and is what Gu, Kelly and Xiu (2020) use. The fixed
split is adequate for comparing model classes, which is the question here, but the
reported ICs are not a simulation of deployment.

Bootstrap resampling assumes rows are independent. Contract-days are not: contracts
on the same date share a common factor and each contract's own rows are serially
correlated, so the panel structure is broken by resampling. This is standard
practice and works adequately, but the assumption is violated.

The feature set was specified once, before any modelling, from standard families in
the cross-sectional literature. It was not searched over. Reporting a feature set
selected because it produced the crossover would be selection on the outcome, and we
note explicitly that this did not happen.

Costs are not modelled beyond the liquidity sweep. Turnover on a 114-feature tree
model is high and the decile spread figures are gross.

The sample covers nine months of one exchange. Crypto perpetuals over this period
are not a general market, and the funding mechanism has no equity analogue.

The universe was taken from currently listed contracts, so perpetuals that delisted
before the window opened never appear.

## References

Geurts, P., Ernst, D. and Wehenkel, L. (2006). Extremely randomized trees.
*Machine Learning*, 63(1), 3-42.

Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine learning.
*Review of Financial Studies*, 33(5), 2223-2273.

Mentch, L. and Zhou, S. (2020). Randomization as regularization: a degrees of
freedom explanation for random forest success. *Journal of Machine Learning
Research*, 21(171), 1-36.
