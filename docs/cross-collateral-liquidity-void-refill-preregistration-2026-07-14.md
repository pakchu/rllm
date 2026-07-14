# CLVR v1 preregistration — 2026-07-14

## Status and claim boundary

- **CLVR returns have not been opened.** This stage may inspect source
  integrity, causal feature values, event timestamps, branch balance, and
  executable support only.
- support artifact:
  `results/cross_collateral_liquidity_void_refill_support_2026-07-14.json`
- support decision: **rejected before returns**
- physically available experiment range: calendar 2023 UTC only
- sealed policy windows: full 2024, full 2025, and 2026 YTD
- no profitability claim is made by this document

CLVR is a new displayed-liquidity mechanism, not another threshold repair of
the rejected price, funding, open-interest, or aggregate-flow experiments. It
compares the *shape response* of Binance's USD-margined linear BTC perpetual
book with its coin-margined inverse BTC perpetual book after the same completed
directional shock.

## Economic hypothesis

The two contracts reference BTC but have different collateral and payoff
structures. Their raw depth units are therefore not comparable. CLVR first
reduces each venue to dimensionless bid/ask and near/far ratios and only then
takes a cross-contract difference.

After a flow-confirmed price shock, the liquidity response can have two
falsifiable interpretations:

1. **void** — coin-margined liquidity on the stress side depletes relative to
   USD-M at both the one-percent level and the one-to-five-percent shape. The
   depleted book is interpreted as incomplete price discovery, so CLVR follows
   the shock;
2. **refill** — coin-margined stress-side liquidity replenishes relative to
   USD-M on both measures. The refill is interpreted as shock absorption, so
   CLVR fades the shock.

Primary references define source and order-book semantics, not CLVR
profitability:

- Binance public data repository:
  <https://github.com/binance/binance-public-data>
- Binance USD-M `BTCUSDT` daily `bookDepth` archive:
  <https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/>
- Binance COIN-M `BTCUSD_PERP` daily `bookDepth` archive:
  <https://data.binance.vision/data/futures/cm/daily/bookDepth/BTCUSD_PERP/>
- Binance COIN-M local-order-book sequencing semantics:
  <https://developers.binance.com/legacy-docs/derivatives/coin-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- Cont, Kukanov, and Stoikov on order-flow imbalance and depth:
  <https://arxiv.org/abs/1011.6402>
- Gould and Bonart on queue imbalance and subsequent price movement:
  <https://arxiv.org/abs/1512.03492>

The papers support testing liquidity imbalance as state information. They do
not establish this cross-collateral transformation or guarantee an edge.

## Physically sealed data

The depth panel is a physically pre-2024 file. Its builder refuses to request
post-2023 rows, verifies every retained Binance archive against the official
checksum sidecar, and records every source hash. Raw snapshots are reduced
independently by contract before the two panels are joined.

| source | range | rows / completeness | SHA256 |
|---|---|---:|---|
| cross-collateral depth panel | 2023-01-01 through 2023-12-31 | 105,120 rows; 101,649 joint-complete | `53e16cf71581f03c7b1cc3da6a13222923ce68aa9e869d89f02078221bb4eee4` |
| Binance BTCUSDT 5m execution market | 2020-01-01 through 2023-12-31 | 420,768 rows | `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d` |

Each accepted contract bar is the independent median of at least eight
complete snapshots in `[bar open, bar open + 5m)`. Its first snapshot must be
no later than 60 seconds after the open and its last no earlier than 240
seconds after the open. Each snapshot must contain all cumulative
`-5..-1,+1..+5` percentage levels. No interpolation, forward fill, backward
fill, nearest-time join, or compressed event clock is permitted.

The official 2023 archives are absent on USD-M 2023-02-08 and 2023-02-09 and
COIN-M 2023-09-25. Published partial sessions also remain gaps. A signal at
bar `t` requires joint-complete depth at `t-6..t`. A depth gap after entry does
not cancel an already scheduled trade because using future source availability
to decide whether the trade existed would itself be lookahead.

The loader fails closed on timestamp, schema, manifest, or file-hash mismatch.
The execution market must be a duplicate-free exact five-minute grid.

## Causal feature clock

For contract `v` in `{um, cm}` and completed five-minute bar `t`, let
`B1`, `B5`, `A1`, and `A5` be cumulative native depth at minus one percent,
minus five percent, plus one percent, and plus five percent. Define within each
contract:

`level_v,t = log(B1_v,t / A1_v,t)`

`shape_v,t = log((B1_v,t / B5_v,t) / (A1_v,t / A5_v,t))`

Then define dimensionless cross-collateral geometry:

`cross_level_t = level_cm,t - level_um,t`

`cross_shape_t = shape_cm,t - shape_um,t`

The completed shock window is six five-minute bars:

`R_t = log(close_t / close_(t-6))`

`D_t = sign(R_t)`

`level_response_t = D_t × (cross_level_t - cross_level_(t-6))`

`shape_response_t = D_t × (cross_shape_t - cross_shape_(t-6))`

For an upward shock, stress-side ask depletion raises both signed responses;
for a downward shock, stress-side bid depletion lowers the raw geometry but
multiplication by `D=-1` again makes both responses positive. Negative signed
responses therefore represent relative refill for either shock direction.

The same six completed market bars must have taker flow aligned with the shock:

`flow_t = sum_(i=t-5..t)(2 × taker_buy_quote_i - quote_volume_i)`

and `D_t × flow_t > 0`.

`R`, `level_response`, and `shape_response` are independently standardized
without the current observation:

1. shift the raw series by one bar;
2. compute a rolling median over 8,640 calendar bars, requiring 2,016 prior
   clean observations;
3. compute the recursive rolling median of prior absolute deviations from
   their corresponding lagged centers;
4. divide the current deviation by `1.4826 × MAD`;
5. treat zero scale as unavailable and clip finite scores to `[-12, 12]`.

Call the results `ZR`, `ZL`, and `ZS`. A row is eligible only when the source
window is clean, flow is aligned, all scores are finite, and `ZL` and `ZS` are
nonzero with the same sign. Its unsigned extremity is

`score_t = cbrt(abs(ZR_t) × abs(ZL_t) × abs(ZS_t))`.

The score threshold is an eligible-only, strictly one-bar-lagged rolling
quantile over 17,280 calendar bars, requiring 4,032 prior eligible scores.
Every input at `t` is available only after bar `t` completes. Entry is the open
of bar `t+1`.

## Frozen action and holding rule

- `ZL>0` and `ZS>0`: branch `void`, side `D_t`;
- `ZL<0` and `ZS<0`: branch `refill`, side `-D_t`.

The two branches share one candidate clock. Positions do not overlap. The
frozen holding period is 12 completed five-minute bars (one hour), followed by
exit at the scheduled open. A new trade may enter at the same open at which a
prior trade exits. Signal, entry, full held path, and scheduled exit must all
lie inside the reported split. A boundary-crossing trade is skipped, never
force-closed.

The branch direction is the central hypothesis and cannot be repaired after
returns open. A failed result cannot be rescued under the CLVR v1 name by
reversing actions, changing the response or holding window, substituting raw
cross-contract depth, changing the flow condition, adding a stop, or filtering
by a return-derived regime.

## Support-only calibration

No return, future high/low, PnL, CAGR, MDD, win rate, or label may be inspected.
Only the score quantile may vary over the frozen grid
`[0.900, 0.925, 0.950, 0.975, 0.990, 0.995]`. The stopping rule selects the
highest quantile satisfying every floor:

- at least 400 non-overlapping candidates in calendar 2023;
- at least 180 in each half;
- at least 75 in each quarter;
- each side at least 25%;
- each branch at least 25%.

All other features, baselines, branch directions, holding period, scheduler,
and support floors are frozen. If no quantile passes, CLVR v1 is rejected
without opening returns. Once the highest passing quantile is materialized in
the support artifact, no further support repair is allowed.

Outcome-blind support was:

| quantile | raw | scheduled | Q1 | Q2 | Q3 | Q4 | H1 | H2 | long | void | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.900 | 6,689 | 1,936 | 325 | 475 | 574 | 562 | 800 | 1,136 | 49.74% | 77.43% | no |
| 0.925 | 5,059 | 1,535 | 264 | 383 | 444 | 444 | 647 | 888 | 50.88% | 79.22% | no |
| 0.950 | 3,397 | 1,089 | 196 | 265 | 309 | 319 | 461 | 628 | 50.05% | 80.44% | no |
| 0.975 | 1,669 | 618 | 114 | 152 | 172 | 180 | 266 | 352 | 50.81% | 79.61% | no |
| 0.990 | 674 | 282 | 60 | 63 | 76 | 83 | 123 | 159 | 48.58% | 74.47% | no |
| 0.995 | 323 | 156 | 34 | 39 | 40 | 43 | 73 | 83 | 47.44% | 71.15% | no |

At `0.900` through `0.975`, support counts and side balance pass but the refill
share is only 19.56% through 22.57%, below the frozen 25% branch floor. At
`0.990` and `0.995`, branch balance passes but the total, half, and quarter
count floors fail. No quantile satisfies every gate, so no quantile is selected
and CLVR v1 is rejected without opening a single return. No evaluator will be
run under the CLVR v1 name.

## Frozen pre-2024 return evaluation

Had support passed, the evaluator would have been committed separately before
it was run and could not have altered the frozen support clock. The support
rejection means this section remains an unexecuted contract.

### Windows

1. train: 2023-01-01 through 2023-06-30;
2. select: 2023-07-01 through 2023-12-31;
3. stability reports: Q1, Q2, Q3, and Q4 of 2023;
4. sealed test: full 2024;
5. sealed eval: full 2025;
6. untouched forward report: 2026 YTD with an explicit end time.

Support calibration may inspect 2023 event incidence but not any 2023 price
after a candidate's entry. Calendar 2023 is not a repository-wide clean room
because prior strategy research has used it; the protection here is a new
mechanism, frozen candidate clock, separately committed evaluator, and sealed
post-2023 progression.

### Exact execution and strict risk

- leverage `L=0.5`;
- fee `5 bp` plus slippage `1 bp` per notional side;
- per-side account cost `c=L×0.0006=0.0003`;
- underlying signed open-to-open return
  `r=side×(exit_open/entry_open-1)`;
- exact multiplier `(1-c)×(1+Lr)×(1-c)`;
- full-clock CAGR counts every idle interval in the split;
- strict MDD applies entry cost, assumes the favorable held extreme forms the
  high-water mark before the adverse held extreme, then applies the
  scheduled-open return and exit cost; the exit bar's later high/low is
  excluded;
- every result reports **absolute return, CAGR, strict MDD,
  CAGR/strict-MDD, and trade count**, plus side and branch counts.

### Frozen controls

The original CLVR schedule is reserved before a control changes or suppresses
an action. Abstention cannot release a later candidate. Diagnostics are:

1. frozen CLVR action;
2. exact direction reversal on every reserved candidate;
3. always follow the completed shock;
4. always fade the completed shock;
5. `void` only and `refill` only, with the other branch abstaining;
6. branch-label permutation using `numpy.random.default_rng(20260714)`, with
   permuted `void` following and permuted `refill` fading.

Controls cannot replace CLVR v1 after outcomes open.

### Statistical test and qualification

For each split, assign each net account trade return to its UTC entry week
(Monday 00:00), sum returns within nonempty weeks, and run 100,000 independent
weekly Rademacher sign flips with seed `20260714`. The statistic is the
trade-weighted mean `sum(weekly sums)/N`; the one-sided p-value is
`(1 + count(permuted >= observed))/100001`. Empty results return `p=1.0`.

CLVR advances to a post-2023 data build only if all conditions hold:

- train and select each have positive absolute return;
- train and select each have CAGR/strict-MDD at least `3.0` and strict MDD at
  most `15%`;
- Q1, Q2, Q3, and Q4 each have positive absolute return and at least 75 trades;
- train and select each have at least 180 trades;
- train and select weekly-cluster one-sided p-values are below `0.10`;
- frozen CLVR beats exact reversal, always-follow, and always-fade on
  `min(train CAGR/MDD, select CAGR/MDD)`.

If any condition fails, CLVR v1 is rejected without threshold, feature,
direction, hold, stop, source, or branch repair. The 2024+ returns remain
closed. If every condition passes, the exact source builder and signal are
extended to a physically frozen 2024 file before test returns are opened; 2025
and 2026 remain sealed until the corresponding earlier gate passes.

## RLLM boundary

CLVR must first establish a deterministic edge. Only after it passes the
pre-2024 gate may one compact Gemma policy receive a causal symbolic state:
shock magnitude bucket, shock direction, flow-alignment strength, signed
liquidity-response buckets, branch, source freshness, current position, and
time to scheduled exit. Raw timestamps and row identifiers are excluded.

The RLLM may abstain or choose bounded size. It may not move the event clock,
reverse the base branch mapping, train on incomplete delayed rewards, receive
future source availability, or use sealed outcomes to redesign CLVR.
