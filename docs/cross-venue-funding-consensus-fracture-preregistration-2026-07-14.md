# CFCF v1 preregistration — 2026-07-14

## Status and claim boundary

- **CFCF returns have not been opened.** This stage uses source integrity,
  causal feature values, event timestamps, and executable support only.
- support artifact:
  `results/cross_venue_funding_consensus_fracture_support_2026-07-14.json`
- verified source range: 2021-01-01 through 2023-12-31 UTC
- sealed policy windows: full 2024, full 2025, and 2026 YTD
- no profitability claim is made by this document

CFCF is a new cross-venue mechanism rather than a threshold repair of the
rejected single-venue flow experiments. It asks whether an unusually large,
same-signed difference between Binance and Bybit premium indices and realized
funding rates identifies directional crowding that subsequently converges.

## Economic hypothesis

Perpetual-futures funding and premium indices are related measures of contract
crowding, but their exact construction and participant mix differ by venue.
When Bybit-minus-Binance premium and realized funding spreads are both unusually
positive, CFCF calls the Bybit side relatively rich and shorts Binance BTCUSDT.
When both are unusually negative, it calls the Bybit side relatively cheap and
goes long Binance BTCUSDT. The trade is a deliberately falsifiable directional
convergence hypothesis, not a claim that the two venue measures are directly
arbitrageable.

Primary sources define the data fields and timestamps, not CFCF profitability:

- Binance funding-rate history:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History>
- Binance premium-index klines:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data>
- Bybit funding-rate history:
  <https://bybit-exchange.github.io/docs/v5/market/history-fund-rate>
- Bybit premium-index klines:
  <https://bybit-exchange.github.io/docs/v5/market/premium-index-kline>
- Binance executable kline semantics:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>

## Physically sealed data

CFCF reads physically pre-2024 copies whose manifests state that outcomes were
not opened. It does not slice a 2026-containing frame at runtime.

| source | range | rows | SHA256 |
|---|---|---:|---|
| Binance funding | 2021-01-01 00:00 through 2023-12-31 16:00 | 3,285 | `654c668e3aea344d5906465cbbd090f2e4ff0c47e9d4bd8cf3856c24549cfc97` |
| Bybit funding | 2021-01-01 00:00 through 2023-12-31 16:00 | 3,285 | `d7e019f34120d84d7c574a361a670b104d8f0c17f9b155d2dd01f1dc74913204` |
| Binance premium index 1h | 2021-01-01 00:00 through 2023-12-31 23:00 | 26,280 | `ed2626c14591cf77f927f71559b81f3c2d0be1d1d5085af4abf7884578f4f972` |
| Bybit premium index 1h | 2021-01-01 00:00 through 2023-12-31 23:00 | 26,280 | `ebfed8281a9e9e9780bbe542c04d00bf52d2dcebe175caa3c8aa3a94f361482b` |
| Binance BTCUSDT 5m execution market | 2020-01-01 through 2023-12-31 | 420,768 | `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d` |

Both premium series must form an exact duplicate-free hourly grid, both funding
series an exact duplicate-free eight-hour grid, and the 2021-2023 execution
slice an exact five-minute grid. A file or manifest hash mismatch fails closed.
The Binance funding source's observed timestamp jitter is at most 47 ms and was
normalized to the nearest UTC eight-hour boundary before this experiment was
defined. Premium-index OHLC is a signal source only; no Bybit premium value is
used as an executable price.

## Causal event and feature clock

Let settlement `s` be one of 00:00, 08:00, or 16:00 UTC. The premium candle
whose open timestamp is `s` is the first complete premium hour following that
settlement. Its close becomes available only after the `:55` five-minute slot
has completed. CFCF therefore materializes the signal on that `:55` slot and
enters at the next five-minute open, one hour after settlement. The realized
funding values stamped at `s` are already known by then.

For every hourly premium row `t`, define

`P_t = BybitPremiumClose_t - BinancePremiumClose_t`.

For every synchronized settlement `s`, define

`F_s = BybitFundingRate_s - BinanceFundingRate_s`.

Each series is robustly standardized without the current observation:

1. shift the raw series by one observation;
2. compute a rolling median;
3. compute a recursive rolling median of prior absolute deviations from their
   corresponding lagged centers;
4. divide the current value minus the lagged center by `1.4826 × MAD`;
5. treat zero scale as unavailable and clip finite scores to `[-12, 12]`.

The fixed premium baseline is 2,160 hours with at least 720 prior hours. The
fixed funding baseline is 270 settlements with at least 90 prior settlements.
Call the resulting scores `ZP_s` and `ZF_s`.

A settlement is eligible only when both scores are finite, nonzero, and have
the same sign. Its signed crowding score is

`C_s = sign(ZP_s) × sqrt(abs(ZP_s) × abs(ZF_s))`.

The crowding threshold is a strictly lagged rolling quantile of `abs(C)` over
540 settlement events, requiring 180 prior available events. Unavailable or
opposite-signed observations do not become candidates.

## Frozen action and holding rule

- `C_s > 0`: branch `bybit_rich`, Binance BTCUSDT side `-1`;
- `C_s < 0`: branch `bybit_cheap`, Binance BTCUSDT side `+1`.

Entry is the next five-minute open after the signal slot. Exit is the open 84
five-minute bars later, which is the next synchronized funding boundary.
Positions do not overlap; a new position may enter at the same open at which a
prior position exits. Settlement origin, signal, entry, full held path, and
scheduled-open exit must all lie inside the reported split. A crossing trade is
skipped, never force-closed.

The action maps a relative-venue state to a directional Binance position. That
mapping is the central risk of the hypothesis and may not be repaired after
returns open. In particular, a failed result cannot be rescued by reversing the
direction, changing the holding period, trading Bybit, or converting the rule
to a two-leg spread under the CFCF v1 name.

## Support-only calibration

Only the crowding-score quantile was varied. Every feature, baseline, direction,
hold, and support floor was fixed. The stopping rule selects the highest tested
quantile satisfying all floors:

- at least 200 non-overlapping candidates total;
- at least 40 in each of 2021, 2022, and 2023;
- at least 30 in each 2023 half;
- each side at least 25%;
- each branch at least 25%.

| quantile | total | 2021 | 2022 | 2023 | H1 | H2 | long | Bybit rich | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.500 | 902 | 181 | 380 | 341 | 167 | 173 | 47.67% | 52.33% | yes |
| 0.600 | 740 | 155 | 309 | 276 | 136 | 139 | 49.05% | 50.95% | yes |
| 0.700 | 579 | 118 | 240 | 221 | 105 | 115 | 51.81% | 48.19% | yes |
| 0.800 | 402 | 78 | 161 | 163 | 75 | 88 | 50.75% | 49.25% | yes |
| **0.900** | **223** | **46** | **88** | **89** | **38** | **51** | **53.81%** | **46.19%** | **yes** |
| 0.925 | 171 | 33 | 68 | 70 | 27 | 43 | 53.80% | 46.20% | no |
| 0.950 | 118 | 22 | 47 | 49 | 17 | 32 | 55.93% | 44.07% | no |
| 0.975 | 68 | 12 | 31 | 25 | 9 | 16 | 55.88% | 44.12% | no |

The selected `0.900` clock has 120 `bybit_cheap` and 103 `bybit_rich`
scheduled candidates. No return, CAGR, MDD, win rate, future high/low, or future
price path was inspected. No further support repair is allowed. Failure of the
return gate rejects CFCF v1.

## Frozen pre-2024 return evaluation

The evaluator must be committed separately before it is run and may not alter
the signal or support clock.

### Windows

1. train: 2021-01-01 through 2022-12-31;
2. select: full 2023, also reported as H1 and H2;
3. sealed test: full 2024;
4. sealed eval: full 2025;
5. untouched forward report: 2026 YTD with explicit end time.

Support calibration inspected 2023 candidate incidence, not 2023 returns.
Pre-2024 history is not a market clean room because other strategy research has
used it, but CFCF's mechanism and gate are frozen before CFCF outcomes.

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
- every result reports **absolute return, CAGR, strict MDD,
  CAGR/strict-MDD, and trade count**, plus side and branch counts.

### Frozen controls

The original CFCF schedule is reserved before a control changes an action, so
an abstaining control cannot release a later candidate:

1. frozen convergence direction;
2. exact direction reversal on every reserved candidate;
3. always long on every reserved candidate;
4. always short on every reserved candidate;
5. `bybit_rich` only and `bybit_cheap` only, with the other branch abstaining;
6. branch-label permutation with `numpy.random.default_rng(20260714)`, mapping
   permuted rich labels to short and permuted cheap labels to long.

Controls are diagnostics and cannot replace CFCF v1 after outcomes open.

### Statistical test and qualification

For each split, assign each net account trade return to its UTC entry week
(Monday 00:00), sum returns within nonempty weeks, and run 100,000 independent
weekly Rademacher sign flips with seed `20260714`. The statistic is the
trade-weighted mean `sum(weekly sums)/N`; the one-sided p-value is
`(1 + count(permuted >= observed))/100001`. Empty results return `p=1.0`.

CFCF advances only if:

- train and full 2023 have positive absolute return;
- train and full 2023 each have CAGR/strict-MDD at least `3.0` and strict MDD
  at most `15%`;
- 2023 H1 and H2 each have positive absolute return and at least 30 trades;
- full 2023 has at least 80 trades;
- train and full 2023 weekly-cluster one-sided p-values are below `0.10`;
- frozen CFCF beats exact reversal, always-long, and always-short policies on
  `min(train CAGR/MDD, 2023 CAGR/MDD)`.

If any condition fails, CFCF v1 is rejected without threshold, feature,
direction, hold, stop, venue, or branch repair. Sealed 2024+ outcomes remain
closed.

## RLLM boundary

CFCF must first establish a deterministic edge. Only after passing train and
2023 may one compact Gemma policy receive causal symbolic state such as premium
spread rank, funding spread rank, their agreement and magnitude, branch,
current position, and time to exit. The RLLM may abstain or size. It may not
change the base event, receive raw timestamps or identifiers, train on
incomplete delayed rewards, or use sealed outcomes to redesign CFCF.
