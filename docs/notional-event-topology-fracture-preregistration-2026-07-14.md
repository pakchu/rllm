# NETF alpha preregistration — 2026-07-14

## Status and boundary

- **NETF-specific returns have not been opened.** This support stage uses only causal feature values, timestamps, and contemporaneously completed prices.
- Pre-2024 returns have been used by unrelated prior research, including the rejected MFIC v1. NETF therefore claims a newly frozen policy, not an untouched market-history clean room.
- 2024, 2025, and 2026 NETF policy outcomes remain sealed.
- support artifact: `results/notional_event_topology_fracture_support_2026-07-14.json`
- support artifact SHA256: `4062da454fbd83e10e04dc9b3b01d0884277023d68e209219bb1bcc76d38588e`

## Motivation

MFIC v1 found at most about `+0.5 bp` of account-level gross edge per trade against about `6 bp` of round-trip account cost. NETF is not a horizon or threshold repair. It changes the state abstraction from same-window impact curvature to a sequential disagreement-and-revelation event with a 4–8 hour consequence horizon.

The core inference is:

> aggressive-event **breadth** can point one way while aggressive **capital/notional** points the other way; if price initially follows breadth but later breadth, capital flow, and price all align with the capital side, the delayed revelation may persist long enough to cover costs.

This exact signal is novel inference, not a claim established by the cited literature.

## External basis

Primary evidence supports the ingredients but not NETF profitability:

- Binance USD-M aggTrades aggregate market fills by price/taking-side/time rules and expose maker-side semantics: <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data>
- Easley and O'Hara, trade size can contain information: <https://www.sciencedirect.com/science/article/pii/0304405X87900298>
- Easley, Kiefer, and O'Hara, the trading process carries information beyond price alone: <https://www.sciencedirect.com/science/article/pii/S0927539897000054>
- Dufour and Engle, inter-trade time and impact are jointly informative: <https://ideas.repec.org/a/bla/jfinan/v55y2000i6p2467-2498.html>
- Rambaldi, Bacry, and Lillo, event timing and volume interact in Hawkes order-book dynamics: <https://arxiv.org/abs/1602.07663>
- Bouchaud, Farmer, and Lillo, markets digest supply/demand changes slowly: <https://arxiv.org/abs/0809.0822>
- Easley, López de Prado, and O'Hara, order-flow toxicity is naturally measured in volume/event time: <https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929>

Because no L2 book is available, NETF does not claim to observe replenishment directly. It infers a completed transition from public aggressive events, notional, and price.

## Data and causality contract

NETF uses the verified official Binance daily archives and the same fail-closed source loader as the frozen MFIC data foundation.

- aggTrade feature SHA256: `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- official daily 5-minute kline SHA256: `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- support range: `2020-01-01 00:00:00` through `2023-12-31 23:55:00`, UTC

The full confirmed source-gap day, each missing aggregate-trade slot, and the following 24 bars are quarantined. Setup, confirmation path, signal bar, next-open entry, full hold path, and scheduled-open exit must all remain clean. Setup origin, signal, entry, and exit must belong to the evaluated split; causal carry-in from a prior split is forbidden. Trades crossing a split boundary are skipped, never force-closed into a split.

All rolling thresholds use only clean observations, are shifted by one complete bar, cover the prior 8,640 bars (30 days), and require at least 2,016 observations (7 days).

## Feature topology

For completed five-minute bar `t`:

- `Q_t = signed_quote_notional_t`; `d_t = sign(Q_t)` is the capital direction;
- `E_t = signed_event_imbalance_t`; `c_t = sign(E_t)` is aggressive-event breadth direction;
- `P_t = sign(micro_log_return_t)` is immediate price direction;
- `C_t = flow_coherence_t`;
- `S_t = buy_sell_event_size_log_ratio_t`.

Define topology tension:

`T_t = sqrt(C_t × |E_t|) × |S_t|`.

The expression measures the **severity of one topology fracture**, not three independent confirmations. If buy/sell event counts are `n_b,n_s` and mean event notionals are `μ_b,μ_s`, opposite signs of `n_b-n_s` and `n_bμ_b-n_sμ_s` are possible only when the mean-size ratio overcomes the count ratio. Consequently `|S_t|` is mechanically related to the required disagreement. NETF intentionally uses it as the margin by which size dominance overcomes breadth, while `C_t` and `|E_t|` measure notional and breadth strength. The threshold is the strictly lagged rolling **87.5th percentile**.

Three structure marks compare the current bar with each feature's strictly lagged rolling 80th percentile:

1. `interarrival_burstiness` — clustered aggressive-event timing;
2. `event_notional_hhi` — concentrated aggregate-event notional;
3. `underlying_trades_per_agg_event` — a large `(last_trade_id - first_trade_id + 1) / aggregate_event_count` span.

At least one structure mark must be active. The third feature is used only as a **trade-ID-span proxy**. NETF does not assume that every integer ID in the span proves one contiguous fill, nor does it infer resting-book liquidity from this field.

## Setup and completed transition

A setup at bar `t0` requires:

1. clean source and at least 64 aggregate-trade events;
2. nonzero capital and crowd directions;
3. `d_t0 = -c_t0` — notional and event breadth disagree;
4. `P_t0 = c_t0` — price initially follows the crowd/event side;
5. `T_t0` at or above its lagged 87.5th percentile;
6. at least one structure mark at or above its lagged 80th percentile.

No trade occurs at setup. After exactly `L` completed bars, capital revelation is confirmed only if, over bars `t0+1 ... t0+L`:

- signed quote notional summed in `d_t0` direction is positive;
- mean signed event imbalance in `d_t0` direction is positive;
- `d_t0 × log(close_(t0+L) / close_t0) > 0`;
- every bar from setup through confirmation is clean.

The signal becomes available after confirmation bar `t0+L` closes. Entry is the next 5-minute open in direction `d_t0`.

## Frozen candidates

| candidate | confirmation | scheduled hold | intended interpretation |
|---|---:|---:|---|
| `netf_fast` | 6 bars / 30m | 48 bars / 4h | rapid capital revelation |
| `netf_slow` | 12 bars / 60m | 96 bars / 8h | slower digestion of the topology fracture |

There is one fixed branch, `capital_revelation`. Direction inversion, branch addition, stop tuning, threshold repair, and post-result hold changes are forbidden.

## Support-only result

No future return, entry-to-exit return, CAGR, MDD, or outcome statistic was computed.

| candidate | setups | raw signals | non-overlap | 2020 | 2021 | 2022 | 2023 | 2023 H1 | 2023 H2 | long | short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `netf_fast` | 1,898 | 360 | 319 | 140 | 57 | 49 | 73 | 29 | 44 | 47.34% | 52.66% |
| `netf_slow` | 1,898 | 318 | 267 | 111 | 48 | 46 | 61 | 20 | 40 | 50.19% | 49.81% |

Both candidates pass the frozen support floors: at least 250 total, 40 per calendar year, 20 per 2023 half, and 25% per side.

Support calibration changed only the tension percentile; every feature, candidate, confirmation, hold, and support floor remained fixed. No NETF return was opened.

| tension quantile | fast total / min year / H1 / H2 | fast pass | slow total / min year / H1 / H2 | slow pass |
|---:|---:|:---:|---:|:---:|
| 0.850 | 391 / 65 / 40 / 51 | yes | 327 / 60 / 26 / 46 | yes |
| 0.875 | 319 / 49 / 29 / 44 | yes | 267 / 46 / 20 / 40 | yes |
| 0.900 | 245 / 37 / 25 / 34 | no | 205 / 35 / 16 / 31 | no |
| 0.950 | 109 / 11 / 12 / 21 | no | 88 / 8 / 10 / 18 | no |
| 0.975 | 62 / 6 / 9 / 11 | no | 48 / 3 / 8 / 11 | no |

The complete tried grid was `{0.850, 0.875, 0.900, 0.950, 0.975}`. The stopping rule is **the highest tried quantile at which both frozen candidates pass every support floor**, selecting `0.875`. No further support-only threshold, feature, branch, candidate, or floor repair is allowed.

## Return-evaluation protocol

### Windows

1. train: `2020-01-01` through `2022-12-31`
2. select: full 2023, with H1 and H2 reported separately
3. sealed test: full 2024
4. sealed eval: full 2025
5. untouched forward report: 2026 YTD with explicit end timestamp

### Execution and metrics

- leverage `L=0.5`;
- notional fee `5 bp` plus slippage `1 bp`, totaling `6 bp` **per side**;
- next-open entry and scheduled-open exit;
- for underlying signed raw return `r = side × (exit_open / entry_open - 1)`, per-side account cost is `c = L × (0.0005 + 0.0001) = 0.0003`, and the exact trade equity multiplier is `(1-c) × (1+Lr) × (1-c)`;
- a flat round trip therefore costs `1-(1-0.0003)^2 = 5.9991 bp` of account equity; trade multipliers compound multiplicatively in chronological order;
- strict MDD applies entry cost first, excludes the exit bar's later OHLC after its open, and assumes the most favorable held extreme establishes the account high-water before the most adverse held extreme; drawdown denominator is that running/hypothetical high-water equity, followed by scheduled-open return and exit cost;
- CAGR uses the full split clock including idle periods;
- exact CAGR is `(ending_equity ** (1 / wall_clock_years) - 1) × 100`, with `wall_clock_years = (split_end - split_start) / 365.25 days`;
- every table reports absolute return, CAGR, strict MDD, CAGR/strict-MDD, and trades.

The exact weekly-cluster sign-flip test is independently frozen here:

1. assign each net trade return to the UTC week of its entry, anchored Monday `00:00:00`;
2. omit empty weeks, retain zero-return trades, and sum returns within each nonempty week;
3. use the trade-weighted observed mean `Σr_i/N`, not an equal-week mean;
4. initialize `numpy.random.default_rng(20260714)` independently per split;
5. perform 100,000 permutations with one independent Rademacher sign per week and statistic `Σ(z_g × weekly_sum_g)/N`;
6. report `(1 + count(permuted >= observed)) / 100001` as the one-sided positive-mean p-value;
7. return `p=1.0` when no trade/nonempty cluster exists.

### Selection gate

A candidate advances only when:

- train and full 2023 absolute returns are positive;
- train and full 2023 CAGR/strict-MDD are at least `3.0`;
- train and full 2023 strict MDD are at most `15%`;
- 2023 H1 and H2 absolute returns are each positive with at least 20 trades;
- full 2023 has at least 60 trades;
- train and full 2023 one-sided weekly-cluster p-values are below `0.10`.

If both qualify, select the larger `min(train ratio, 2023 ratio)`, then lower 2023 strict MDD, then lexicographic candidate name. If neither qualifies, reject NETF v1 without repair.

The sealed claim requires 2024 and 2025 independently to have positive absolute return, CAGR/strict-MDD at least `3.0`, strict MDD at most `15%`, at least 40 trades, and weekly-cluster `p < 0.10`; combined 2024–2025 must have `p < 0.05`. 2026 cannot alter selection.

## RLLM boundary

If NETF survives sealed validation, the compact causal state—capital direction, crowd direction, tension percentile, active structure marks, confirmation progress, and current position—can become an RLLM observation. The LLM/RL layer may abstain or size but may not reconstruct future labels or mutate the frozen base event using sealed outcomes.
