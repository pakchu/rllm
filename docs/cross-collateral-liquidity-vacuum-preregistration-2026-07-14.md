# CLV v1 preregistration — 2026-07-14

## Status and claim boundary

- **CLV returns have not been opened.** Only source integrity, causal feature
  values, event timestamps, side balance, and executable support may be used at
  this stage.
- source range: calendar 2023 UTC only
- sealed policy windows: full 2024, full 2025, and 2026 YTD
- support artifact:
  `results/cross_collateral_liquidity_vacuum_support_2026-07-14.json`

The outcome-blind CLVR support study established that its refill state was too
rare for the preregistered branch floor. It did not inspect a return. CLV is a
separate, narrower economic claim: trade only the sufficiently frequent
cross-collateral liquidity **vacuum** state and follow the completed shock.
The rejected refill/fade hypothesis is not silently repaired or evaluated.

## Economic hypothesis

Binance's USD-margined linear `BTCUSDT` perpetual and coin-margined inverse
`BTCUSD_PERP` perpetual have different collateral and payoff structures. Their
raw depth units cannot be compared. CLV instead asks whether a price shock has
left the coin-margined stress side unusually hollow relative to USD-M after
each book is reduced to dimensionless bid/ask and near/far geometry.

For an upward shock, relative depletion of coin-margined ask depth is a vacuum;
for a downward shock, relative depletion of coin-margined bid depth is a
vacuum. When both depth-level and depth-shape responses agree, CLV interprets
the vacuum as incomplete price discovery and follows the shock for one hour.

Primary sources define data and book semantics, not profitability:

- Binance public data repository:
  <https://github.com/binance/binance-public-data>
- USD-M `BTCUSDT` `bookDepth` archive:
  <https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/>
- COIN-M `BTCUSD_PERP` `bookDepth` archive:
  <https://data.binance.vision/data/futures/cm/daily/bookDepth/BTCUSD_PERP/>
- Binance COIN-M local-order-book sequencing semantics:
  <https://developers.binance.com/legacy-docs/derivatives/coin-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- Order-flow imbalance and depth evidence:
  <https://arxiv.org/abs/1011.6402>
- Queue-imbalance evidence:
  <https://arxiv.org/abs/1512.03492>

The cited papers motivate testing liquidity imbalance as state information;
they do not establish CLV or imply a trading edge.

## Physically sealed data and availability

| source | range | rows / completeness | SHA256 |
|---|---|---:|---|
| cross-collateral depth panel | 2023-01-01 through 2023-12-31 | 105,120 rows; 101,649 joint-complete | `53e16cf71581f03c7b1cc3da6a13222923ce68aa9e869d89f02078221bb4eee4` |
| Binance BTCUSDT 5m execution market | 2020-01-01 through 2023-12-31 | 420,768 rows | `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d` |

The depth builder is physically bounded to 2023 and verifies each available
official archive with its checksum sidecar. Each retained contract bar is the
independent median of at least eight complete snapshots in
`[bar open, bar open + 5m)`, with first snapshot no later than 60 seconds and
last snapshot no earlier than 240 seconds. Each snapshot must contain all
cumulative `-5..-1,+1..+5` percentage levels.

There is no interpolation, fill, nearest-time join, or compressed event clock.
A signal at `t` requires both contracts to be complete at every bar `t-6..t`.
A gap after entry does not cancel a scheduled trade because future source
availability cannot decide whether a past trade existed. Hash, schema,
timestamp, and exact five-minute-grid checks fail closed.

## Frozen causal state

For contract `v` in `{um, cm}` and completed bar `t`, let `B1`, `B5`, `A1`,
and `A5` denote cumulative depth at minus one, minus five, plus one, and plus
five percent. Define:

`level_v,t = log(B1_v,t / A1_v,t)`

`shape_v,t = log((B1_v,t / B5_v,t) / (A1_v,t / A5_v,t))`

`cross_level_t = level_cm,t - level_um,t`

`cross_shape_t = shape_cm,t - shape_um,t`

Over the completed six-bar response window:

`R_t = log(close_t / close_(t-6))`

`D_t = sign(R_t)`

`level_response_t = D_t × (cross_level_t - cross_level_(t-6))`

`shape_response_t = D_t × (cross_shape_t - cross_shape_(t-6))`

The same six completed market bars require shock-aligned taker flow:

`D_t × sum_(i=t-5..t)(2 × taker_buy_quote_i - quote_volume_i) > 0`.

`R`, `level_response`, and `shape_response` are independently transformed to
`ZR`, `ZL`, and `ZS` by a strictly one-bar-lagged rolling median and recursive
MAD over 8,640 calendar bars, requiring 2,016 prior clean observations. The
scale is `1.4826 × MAD`, zero scale is unavailable, and finite values are
clipped to `[-12,12]`.

A vacuum is eligible only when the source window is clean, flow is aligned,
all three scores are finite, and `ZL>0` and `ZS>0`. Its extremity is

`score_t = cbrt(abs(ZR_t) × abs(ZL_t) × abs(ZS_t))`.

The threshold is an eligible-event, strictly one-bar-lagged rolling quantile
over 17,280 calendar bars, requiring 4,032 prior eligible cross-collateral
response scores. The inherited feature implementation calculates the threshold
before the refill branch is suppressed; this exact shared-state rarity clock
is frozen and hash-pinned rather than rewritten after CLVR support was seen.

Every input at `t` is available only after the five-minute bar closes. CLV
enters `side=D_t` at the next five-minute open and exits at the open 12 bars
later. Positions do not overlap; a position may enter where the preceding one
exits. Signal, entry, held path, and scheduled exit must be inside the split.

## Support-only calibration

No return, future high/low, PnL, CAGR, MDD, win rate, or label may be inspected.
Only the shared-state score quantile varies over
`[0.900,0.925,0.950,0.975,0.990,0.995]`. The highest quantile passing every
floor is selected:

- at least 400 non-overlapping vacuum candidates in 2023;
- at least 180 in each half;
- at least 75 in each quarter;
- long and short each at least 25%.

Every feature, direction, response and holding window, scheduler, and support
floor is otherwise fixed. If no quantile passes, CLV v1 is rejected without
returns.

Outcome-blind support was:

| quantile | raw | scheduled | Q1 | Q2 | Q3 | Q4 | H1 | H2 | long | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.900 | 5,034 | 1,632 | 281 | 401 | 460 | 490 | 682 | 950 | 50.43% | yes |
| 0.925 | 3,838 | 1,305 | 225 | 319 | 369 | 392 | 544 | 761 | 51.26% | yes |
| 0.950 | 2,595 | 933 | 160 | 228 | 267 | 278 | 388 | 545 | 51.55% | yes |
| **0.975** | **1,221** | **521** | **91** | **129** | **148** | **153** | **220** | **301** | **51.82%** | **yes** |
| 0.990 | 463 | 223 | 42 | 52 | 65 | 64 | 94 | 129 | 50.22% | no |
| 0.995 | 205 | 116 | 20 | 31 | 33 | 32 | 51 | 65 | 47.41% | no |

The stopping rule selects `0.975`, the highest passing quantile. Its frozen
schedule has 270 long and 251 short candidates. No return, future price path,
or outcome statistic was opened. No further support repair is allowed.

## Frozen return evaluation contract

Only if support passes, a separate evaluator must be committed before any CLV
return is opened.

### Splits

1. train: 2023-01-01 through 2023-06-30;
2. select: 2023-07-01 through 2023-12-31;
3. stability: Q1, Q2, Q3, Q4;
4. sealed test: full 2024;
5. sealed eval: full 2025;
6. untouched forward report: 2026 YTD with explicit end time.

### Execution and strict risk

- leverage `L=0.5`;
- fee `5 bp` plus slippage `1 bp` per notional side;
- per-side account cost `c=0.0003`;
- signed underlying return `r=side×(exit_open/entry_open-1)`;
- exact multiplier `(1-c)×(1+Lr)×(1-c)`;
- full-clock CAGR includes every idle interval;
- strict MDD applies entry cost, favorable held extreme first, adverse held
  extreme second, then scheduled-open return and exit cost; the exit bar's
  later high/low is excluded;
- every table reports **absolute return, CAGR, strict MDD, CAGR/strict-MDD,
  and trades**, plus side counts.

### Frozen controls and statistics

The original candidate schedule is reserved before a control changes an
action, so abstention never releases a later candidate. Controls are frozen
CLV, exact reversal, always long, always short, and a
`numpy.random.default_rng(20260714)` sign permutation. They are diagnostics and
cannot replace CLV.

Net account trade returns are clustered by UTC entry week. Each split uses
100,000 independent weekly Rademacher sign flips with seed `20260714`; the
one-sided p-value is `(1 + count(permuted >= observed))/100001`. Empty results
return `p=1.0`.

CLV advances only if:

- train and select each have positive absolute return;
- train and select each have CAGR/strict-MDD at least `3.0` and strict MDD at
  most `15%`;
- every 2023 quarter has positive absolute return and at least 75 trades;
- train and select each have at least 180 trades and weekly p-value below
  `0.10`;
- frozen CLV beats exact reversal, always-long, and always-short on
  `min(train CAGR/MDD, select CAGR/MDD)`.

Any failure rejects CLV v1 without threshold, feature, direction, hold, stop,
or regime repair. The 2024+ outcomes remain closed. If all gates pass, the
unchanged builder and signal are extended to a physically frozen 2024 file
before test returns are opened.

## RLLM boundary

Only after deterministic qualification may one compact Gemma policy receive
causal symbolic buckets for shock magnitude and direction, flow strength,
liquidity-level response, liquidity-shape response, source freshness, current
position, and time to exit. It may abstain or choose bounded size. It may not
move the event clock, reverse the base action, receive identifiers or future
availability, train on incomplete delayed rewards, or redesign CLV from sealed
outcomes.
