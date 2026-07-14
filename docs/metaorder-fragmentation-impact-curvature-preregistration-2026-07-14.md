# MFIC alpha preregistration — 2026-07-14

## Status

- **Protocol:** frozen before any return or backtest outcome is opened.
- **Outcome access:** `false` in the support artifact.
- **Primary artifact:** `results/metaorder_fragmentation_impact_curvature_support_2026-07-14.json`.
- **Purpose:** test whether a fragmented hidden metaorder can be distinguished from exhausted/absorbed flow by the curvature of its price impact.
- **No claim of profitability is made by this document.** The mechanism and support passed; returns remain unopened.

## Economic hypothesis

Large parent orders are commonly split into many smaller child orders. This can create persistent order signs even when each public aggregate trade looks ordinary. Persistent flow should continue moving price while marginal impact is strengthening; the same persistent flow should be faded when price extension remains positive but marginal impact collapses, which is interpreted as absorption or exhaustion.

The hypothesis combines four externally supported observations:

1. order signs can exhibit long memory because large hidden orders are split into smaller executions;
2. market impact is concave rather than linear in size;
3. impact can decay through time rather than remain permanent;
4. continuation and contrarian responses can therefore be different phases of the same latent execution process.

These papers support the mechanism, not this implementation's profitability:

- Lillo, Mike, and Farmer, *Theory for long memory in supply and demand*: <https://arxiv.org/pdf/cond-mat/0412708.pdf>
- Tóth et al., *Anomalous price impact and the critical nature of liquidity in financial markets*: <https://arxiv.org/pdf/1105.1694.pdf>
- Bouchaud et al., *Fluctuations and response in financial markets: the subtle nature of random price changes*: <https://arxiv.org/pdf/cond-mat/0307332.pdf>
- Jaisson, *Market impact as anticipation of the order flow imbalance*: <https://arxiv.org/pdf/1402.1288.pdf>
- Benzaquen et al., *Dissecting cross-impact and market stability*: <https://arxiv.org/pdf/1901.05332.pdf>
- Tóth et al., *Why is order flow so persistent?*: <https://arxiv.org/pdf/1104.0587.pdf>

## Data contract

Only official Binance USD-M BTCUSDT daily archives are used.

- Archive root: <https://data.binance.vision/>
- Binance aggregate-trade semantics: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Compressed-Aggregate-Trades-List>
- Binance aggregate-trade stream field semantics: <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams>
- Binance kline semantics: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>

The archive field `m=true` means the buyer is the maker. MFIC therefore assigns the corresponding aggregate trade to aggressive selling; `m=false` is assigned to aggressive buying.

Frozen input hashes:

- aggregate-trade 5-minute features: `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- official daily kline reference: `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- range: `2020-01-01 00:00:00` through `2023-12-31 23:55:00`, UTC

The build manifests must state `outcomes_opened=false`, and their hashes must match before the signal code runs.
The official kline frame must also be a duplicate-free, strictly increasing, gapless 5-minute UTC grid; row-based windows and holds fail closed otherwise.

## Causal feature definition

For completed 5-minute bar `i`, let:

- `s=+1` for aggressive buys and `s=-1` for aggressive sells;
- `f_i = Σ(price × quantity × s)`, signed quote flow;
- `a_i = |f_i|`;
- `c_i = |f_i| / Σ(price × quantity)`, flow coherence;
- `n_i` be the number of aggregate-trade events;
- `hhi_i = Σ(event_notional²) / Σ(event_notional)²`;
- `e_i = (1 / hhi_i) / n_i`, normalized effective event count;
- `r_i = (1 - sign_flip_rate_i + max_same_sign_run_share_i) / 2`;
- `u_i = sign(f_i) × log(last_trade_price_i / first_trade_price_i)`;
- `j_i = u_i / sqrt(max(c_i, 0.01))`, concavity-normalized signed impact.

For a candidate window `W`, define the latent metaorder direction:

`d_t = sign(Σ[i=t-W+1..t] f_i)`.

Only bars whose `f_i` agrees with `d_t` contribute to the directional components below. With absolute-flow weighting:

- `P_t`: agreeing-flow notional divided by all absolute-flow notional;
- `C_t`: weighted mean of `c_i`;
- `F_t`: weighted mean of `sqrt(e_i)`;
- `R_t`: weighted mean of `r_i`;
- `H_t = P_t × C_t × F_t × R_t`, hidden-metaorder score.

The scale of fragmentation changed as the venue matured. An absolute threshold would select an era, not a mechanism. MFIC therefore compares `H_t` with the **strictly lagged** 95th percentile of clean `H` observations over the prior 8,640 bars (30 days), requiring at least 2,016 past observations (7 days). The current bar is excluded with `shift(1)`.

## Impact curvature and decisions

Each candidate uses a recent segment of `S=W/4` bars and the immediately preceding `S` bars. Within each segment, `j_i` is weighted by agreeing absolute flow.

- `J_recent`: agreeing-flow weighted mean impact in the recent segment;
- `J_prior`: agreeing-flow weighted mean impact in the prior segment;
- `K_t = J_recent - J_prior`, impact curvature;
- `X_t = d_t × log(close_t / close_(t-W))`, directional extension.

A bar is mechanism-eligible only when:

1. every lookback bar is outside source quarantine;
2. every lookback bar has at least 64 aggregate-trade events;
3. both the recent and prior segments contain at least two agreeing-flow bars;
4. `P_t >= 0.60`;
5. `C_t >= 0.20`;
6. `H_t` is at or above its lagged 30-day 95th percentile.

The fixed decision branches are:

- **Continuation:** `K_t >= 0.002` and `J_recent > 0`; trade in direction `d_t`.
- **Fade:** `K_t <= -0.002`, `J_prior > 0`, and `X_t > 0`; trade opposite direction `d_t`.

No branch may be removed after returns are observed.

## Frozen candidates

| candidate | lookback | segment | continuation hold | fade hold |
|---|---:|---:|---:|---:|
| `mfic_fast` | 12 bars / 60m | 3 bars / 15m | 3 bars / 15m | 6 bars / 30m |
| `mfic_slow` | 24 bars / 120m | 6 bars / 30m | 6 bars / 30m | 12 bars / 60m |

The signal is available only after bar `t` completes. Entry is at bar `t+1` open. Exit is at the scheduled future open. A new position may enter at the same open at which the previous position exits, but positions may not overlap.

## Source-gap and split containment policy

The verified aggregate-trade source has confirmed gaps on:

- `2020-04-15`
- `2021-02-09`
- `2021-02-24`
- `2021-05-19`
- `2022-09-06`

The full UTC source-gap day, every missing 5-minute slot, and the following 24 bars are quarantined. The signal lookback, signal bar, entry, complete holding path, and exit must all avoid quarantine. A trade whose scheduled exit crosses a split boundary is skipped; it is not force-closed and cannot contribute to either split.

## Support-only result

No open, high, low, close return was computed. OHLC close was used only for the contemporaneously known extension condition.

| candidate | non-overlap total | 2020 | 2021 | 2022 | 2023 | 2023 H1 | 2023 H2 | long | short | continuation | fade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mfic_fast` | 1,566 | 344 | 637 | 367 | 218 | 139 | 79 | 48.91% | 51.09% | 63.79% | 36.21% |
| `mfic_slow` | 1,635 | 290 | 646 | 456 | 243 | 164 | 79 | 49.30% | 50.70% | 60.49% | 39.51% |

Both candidates pass the preregistered support floors: total at least 250, at least 40 in every calendar year, each side at least 25%, and each branch at least 20%.

The first absolute thresholds (`fragmentation >= 0.45`, `run persistence >= 0.50`) produced only 3 and 0 non-overlapping observations. Those thresholds were rejected **without opening returns** because they were incompatible with the measured feature scale and heavily selected the earliest venue era. The final lagged-percentile rule was chosen solely from feature distributions and support counts.

## Return-evaluation protocol

The next commit may implement the evaluator, but it may not change the signal above.

### Windows

1. `train`: `2020-01-01` through `2022-12-31`
2. `select2023`: `2023-01-01` through `2023-12-31`, also reported as H1 and H2
3. sealed `test2024`: full calendar year 2024
4. sealed `eval2025`: full calendar year 2025
5. untouched forward report: 2026 YTD, with an explicit end timestamp

Support-only access to 2023 feature distributions does not expose 2023 returns. Thresholds and both candidates are now frozen before any window is evaluated.

### Execution and metrics

- leverage: `0.5x`
- fee plus slippage: `6 bp` per side
- next-open entry; scheduled-open exit
- CAGR annualizes over the full wall-clock split, including idle cash time
- strict MDD assumes the favorable extreme establishes the high-water mark before the adverse extreme over the complete held path
- every table must show: **absolute return, CAGR, strict MDD, CAGR/strict-MDD, and trade count**
- side counts, branch counts, win rate, mean trade return, and confidence statistics are secondary fields

### Statistical rule

Trades are non-overlapping, but residual serial dependence is still possible. In addition to the conventional t-like diagnostic, evaluation will use this frozen deterministic cluster test:

1. assign every net-of-cost trade return `r_i` to the UTC week containing its **entry** timestamp, with Monday `00:00:00` UTC as the anchor;
2. omit calendar weeks with no trades, include zero-return trades, and compute each nonempty cluster sum `R_g = Σ(i in g) r_i`;
3. use the trade-weighted observed statistic `T_obs = Σ_i r_i / N`, not an equal-week average;
4. initialize `numpy.random.default_rng(20260714)` independently for each evaluated split;
5. for each of 100,000 permutations, draw one independent Rademacher sign `z_g ∈ {-1,+1}` per nonempty week and compute `T_b = Σ_g(z_g × R_g) / N`;
6. report the one-sided positive-mean p-value `(1 + count(T_b >= T_obs)) / 100001`;
7. return `p=1.0` when `N=0` or no nonempty cluster exists.

The grouping, entry-time convention, cluster-sum weighting, comparison operator, correction, permutation count, and seed may not change after outcomes are opened.

### Selection and rejection rule

A candidate advances only if all of the following hold:

- `train` and `select2023` each have positive absolute return;
- CAGR/strict-MDD is at least `3.0` and strict MDD is at most `15%` in both windows;
- `select2023` H1 and H2 each have positive absolute return and at least 40 trades;
- full `select2023` has at least 100 trades;
- the one-sided weekly-cluster p-value is below `0.10` in both `train` and `select2023`.

If both candidates qualify, select the one with the larger `min(train ratio, select2023 ratio)`, then lower `select2023` strict MDD, then lexicographic candidate name. If neither qualifies, **MFIC v1 is rejected**; thresholds, branches, and holds are not repaired after observing returns.

The sealed OOS claim requires `test2024` and `eval2025` independently to meet positive absolute return, CAGR/strict-MDD at least `3.0`, strict MDD at most `15%`, at least 40 trades, and one-sided weekly-cluster `p < 0.10`. The combined 2024–2025 result must have `p < 0.05`. The 2026 YTD result is reported but cannot retroactively select or alter the model.

## Intended RLLM integration boundary

MFIC is first tested as a deterministic causal alpha so that signal validity is separable from model capacity. If and only if it survives the sealed protocol, `H/H_baseline`, `K`, `P`, `C`, `X`, branch, current position, and time-to-exit become compact state tokens for the RLLM policy. The LLM/RL layer may size or abstain; it may not reconstruct future-dependent MFIC labels or change the frozen base signal using sealed outcomes.
