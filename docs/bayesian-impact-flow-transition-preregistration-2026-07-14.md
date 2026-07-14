# BIFT v1 preregistration — 2026-07-14

## Status and claim boundary

- **BIFT returns have not been opened.** This stage uses causal feature values,
  timestamps, contemporaneously completed prices, and executable support only.
- support artifact:
  `results/bayesian_impact_flow_transition_support_2026-07-14.json`
- verified source range: 2020-01-01 through 2023-12-31 UTC
- sealed policy windows: full 2024, full 2025, and 2026 YTD
- no profitability claim is made by this document

BIFT is an independent successor to the rejected NETF and CARTA experiments.
It does not repair their thresholds, token mappings, or holds. It tests a
completed mechanism: **a change in the relation among aggressive flow, price
impact, and event intensity, followed by either propagation or absorption**.

## Economic hypothesis

Public aggressive trades can cluster, but their price consequence is not
stationary. A large flow shock that continues and moves price in its direction
is interpreted as propagating impact. The same persistent flow without price
alignment is interpreted as absorption and is faded. Bayesian online
change-point detection (BOCPD) supplies the event clock; it is not asked to
predict a return directly.

Primary sources support the ingredients, not BIFT profitability:

- Adams and MacKay, *Bayesian Online Changepoint Detection*:
  <https://arxiv.org/abs/0710.3742>
- Bacry and Muzy, *Hawkes model for price and trades high-frequency dynamics*:
  <https://arxiv.org/pdf/1301.1135>
- Jaisson, *Market impact as anticipation of the order flow imbalance*:
  <https://arxiv.org/pdf/1402.1288>
- Binance USD-M aggregate-trade API semantics:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Compressed-Aggregate-Trades-List>
- Binance USD-M kline semantics:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>

## Verified data and availability

BIFT uses only the already verified official Binance BTCUSDT USD-M archives.

- aggregate-trade 5m feature SHA256:
  `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- official 5m kline SHA256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`

An hour is usable only when all twelve 5-minute slots from `:00` through
`:55` exist, are outside source quarantine, have positive quote notional and
event count, and have valid positive open/close prices. Its timestamp is the
last included 5-minute bar (`:55`); it is never materialized at the hour start.

The full confirmed source-gap day, every missing 5-minute slot, and the next 24
bars are quarantined. BOCPD is run independently on each contiguous usable
hourly segment. An unavailable hour resets its run-length posterior and prior
expected run. Rolling robust scaling and change-score thresholds do not
compress time or reset: they remain wall-clock windows, ignore invalid values,
and use strictly prior clean observations only.

## Causal hourly observations

For a completed hour `t`, define:

- `Q_t`: total quote notional;
- `F_t`: signed aggressive quote notional;
- `I_t = F_t / Q_t`: aggressive flow imbalance;
- `R_t = log(close_t / open_t)`: completed-hour price return;
- `A_t = sign(I_t) × R_t / sqrt(max(abs(I_t), 0.01))`: concavity-normalized
  impact alignment;
- `N_t`: aggregate-trade event count;
- `C_t = log(1 + N_t)`: event-intensity scale.

The fixed BOCPD vector is `(I_t, A_t, C_t)`. Event-count imbalance is retained
for diagnostics but is not a fourth detector dimension; adding it later would
be a new experiment.

Each vector component is standardized from a strictly lagged 720-hour rolling
median and recursive rolling median absolute deviation, requiring 168 prior
clean hours. The current hour is excluded with `shift(1)`, scale uses `1.4826 ×
MAD`, zero scales become unavailable, and finite scores are clipped to
`[-12, 12]`.

## BOCPD event clock

BIFT uses a multivariate independent-dimension Student-t predictive recursion
with the following fixed parameters:

- constant hazard mean: 168 hours;
- maximum retained run length: 672 hours;
- short-run diagnostic horizon: 6 hours;
- Normal-Gamma prior: `kappa=0.1`, `alpha=2.0`, `beta=1.0` per dimension.

A constant hazard makes raw reset probability nearly constant. BIFT therefore
requires **both**:

1. normalized expected-run drop; and
2. predictive surprise

to exceed their own strictly lagged wall-clock 4,320-hour percentile, with at
least 720 prior valid detector hours. The setup hour must also satisfy
`abs(I_t) >= 0.02`. Neither threshold sees the current observation in its
baseline.

## Fixed transition and branch semantics

Let `t0` be a setup and `d = sign(I_t0)`. No position opens at `t0`. Exactly
three subsequent completed hours, `t0+1 .. t0+3`, are observed.

A candidate exists only if:

1. setup and all three confirmation hours are clean;
2. `d × sum(I_t0+1 .. I_t0+3) > 0`, so flow remains persistent;
3. `log(close_t0+3 / close_t0)` is finite and nonzero.

The branches are immutable:

- **propagation:** `d × log(close_t0+3 / close_t0) > 0`; trade side `d`;
- **absorption:** `d × log(close_t0+3 / close_t0) < 0`; trade side `-d`.

Zero or unavailable price confirmation fails closed; it is not forced into the
absorption branch. The signal becomes available after the third confirmation
hour's `:55` bar completes. Entry is the next 5-minute open. Exit is the open
144 bars later (12 hours). Positions do not overlap; a new position may enter
at the same open at which the prior one exits.

Both branches share the same event clock. A later evaluator may not remove a
losing branch, invert one branch, change confirmation or hold, or release
skipped events to create a different opportunity set.

## Split containment

For every reported split, setup origin, all three confirmation hours, signal,
entry, complete held path, and scheduled-open exit must lie inside the split
and outside quarantine. A crossing trade is skipped, never force-closed. The
generic scheduler also rechecks signal through exit quarantine even for a
hand-built signal frame.

## Support-only calibration

Only the joint run-drop/surprise percentile was varied. Every feature,
baseline, branch, confirmation, hold, and support floor was fixed. The stopping
rule selects the highest tested percentile passing all floors:

- at least 250 non-overlapping candidates total;
- at least 40 in each calendar year;
- at least 30 in each 2023 half;
- each side at least 25%;
- each branch at least 25%.

| percentile | total | 2020 | 2021 | 2022 | 2023 | H1 | H2 | long | propagation | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.900 | 323 | 75 | 60 | 90 | 98 | 58 | 39 | 53.56% | 68.42% | yes |
| **0.925** | **272** | **64** | **46** | **76** | **86** | **51** | **34** | **53.68%** | **69.85%** | **yes** |
| 0.950 | 216 | 48 | 33 | 64 | 71 | 40 | 30 | 56.94% | 70.37% | no |
| 0.975 | 143 | 33 | 25 | 44 | 41 | 24 | 16 | 55.94% | 72.73% | no |
| 0.990 | 62 | 15 | 10 | 22 | 15 | 8 | 6 | 56.45% | 75.81% | no |

The selected `0.925` clock has 190 propagation and 82 absorption trades; long
and short shares are 53.68% and 46.32%. The source produces 35,064 hourly rows,
34,921 clean hours, 34,586 detector-available hours, and 10 independently reset
posterior segments. No return, CAGR, MDD, win rate, or future path was used.

No further support repair is allowed after this artifact. Failure of the
return gate rejects **BIFT v1**.

## Frozen return-evaluation protocol

The evaluator must be committed separately and may not alter the signal.

### Windows

1. train: 2020-01-01 through 2022-12-31;
2. select: full 2023, also reported as H1 and H2;
3. sealed test: full 2024;
4. sealed eval: full 2025;
5. untouched forward report: 2026 YTD with explicit end time.

Support calibration inspected 2023 candidate incidence, not 2023 returns.
Pre-2024 history is not a market clean room because other strategies have used
it, but the BIFT rule and its evaluation gate are frozen before BIFT outcomes.

### Exact execution and strict risk

- leverage `L=0.5`;
- fee `5 bp` plus slippage `1 bp` per notional side;
- per-side account cost `c=L×0.0006=0.0003`;
- underlying signed open-to-open return
  `r=side×(exit_open/entry_open-1)`;
- exact multiplier `(1-c)×(1+Lr)×(1-c)`;
- full-clock CAGR includes every idle interval;
- strict MDD applies entry cost, then assumes the favorable held extreme forms
  the high-water mark before the adverse held extreme, then scheduled-open
  return and exit cost; the exit bar's later high/low is excluded;
- every table reports **absolute return, CAGR, strict MDD,
  CAGR/strict-MDD, and trade count**, plus side and branch counts.

### Frozen controls

The candidate schedule is reserved before control actions are applied. Report:

1. frozen BIFT branch mapping;
2. always follow the setup flow on every candidate;
3. always fade the setup flow on every candidate;
4. propagation-only and absorption-only, with the other branch abstaining but
   not releasing later candidates;
5. branch-label permutation using `numpy.random.default_rng(20260714)`.

Controls cannot replace the preregistered BIFT policy after outcomes open.

### Statistical test and qualification

For each split, assign each net account trade return to its UTC entry week
(Monday 00:00), sum returns within nonempty weeks, and run 100,000 independent
weekly Rademacher sign flips with seed `20260714`. The statistic is the
trade-weighted mean `sum(weekly sums)/N`; the one-sided p-value is
`(1 + count(permuted >= observed))/100001`. Empty results return `p=1.0`.

BIFT advances only if:

- train and full 2023 have positive absolute return;
- train and full 2023 each have CAGR/strict-MDD at least `3.0` and strict MDD
  at most `15%`;
- 2023 H1 and H2 each have positive absolute return and at least 30 trades;
- full 2023 has at least 80 trades;
- train and full 2023 weekly-cluster one-sided p-values are below `0.10`;
- the frozen branch mapping beats both always-follow and always-fade on
  `min(train ratio, 2023 ratio)` without deleting either branch.

If any condition fails, BIFT v1 is rejected without branch, threshold, hold,
stop, or direction repair. Sealed 2024+ outcomes remain closed.

## RLLM boundary

BIFT must first establish a deterministic edge. Only after passing train and
2023 may one compact Gemma policy receive causal state such as run-drop rank,
surprise rank, flow persistence, impact relation, branch, current position, and
time to exit. The RLLM may abstain or size. It may not alter the base event,
recover timestamps/raw identifiers, train on incomplete delayed rewards, or use
sealed outcomes to redesign BIFT.
