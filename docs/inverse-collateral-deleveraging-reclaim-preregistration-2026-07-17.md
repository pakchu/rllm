# ICDR-144 preregistration — Inverse-Collateral Deleveraging Reclaim

## Status and claim boundary

**No ICDR-144 strategy outcome has been opened.** This document freezes one
candidate, one economic direction, one support-only grid, one confirmation
state machine, one execution policy, and the later strict gates before any
entry-to-exit return is calculated.

- policy: `ICDR-144`
- signal source: official Binance USD-M/COIN-M five-minute positioning metrics
- available history: 2021-07-08 through 2023-12-31 UTC
- action: fixed long BTCUSDT USD-M perpetual
- entry: two five-minute opens after reclaim confirmation
- hold: 144 five-minute bars / 12 hours
- leverage: 0.5x
- base cost: 6 bp/notional/side
- stress cost: 10 bp/notional/side
- 2024, 2025, and 2026 YTD remain sealed

Historical market periods have been seen by unrelated repository research, so
pre-2024 is not a global clean room. The exact ICDR clock and its post-entry
returns remain unopened. The source manifest and panel are frozen in commit
`8d347432cd36d59458ad9a26c7c8aef1ec94b8ee`.

## Why this is a different observable axis

Existing REX, USD-M OI, funding/premium, Kimchi/FX, inferred-liquidation,
cross-venue flow, aggregate-trade, and cross-collateral book experiments do not
observe the same object. ICDR isolates **the relative contraction of a
BTC-margined inverse-perpetual cohort** versus USDT-margined positioning and
requires that the COIN-M cohort's own taker flow recover before entry.

The hypothesis is asymmetric. During a BTC decline, BTC-valued collateral can
lose value while COIN-M participants still carry fixed-USD contracts. An
unusually sharp COIN-M OI contraction accompanied by COIN-M-specific taker
selling may therefore represent a collateral purge. ICDR does not buy the
purge itself. It waits for selling and one-bar OI change to stop, then tests a
12-hour reclaim. No price return, REX state, funding, premium, or macro field is
allowed in the signal.

This mechanism can still fail. Relative COIN-M contraction may merely follow
price, the taker recovery can be noise, or the cohort may be too small to move
BTCUSDT. Those possibilities are handled by frozen controls rather than by
post-outcome repair.

## Source and unit contract

Source audit:
`docs/binance-cross-collateral-positioning-metrics-source-audit-2026-07-17.md`.

The raw product levels are not comparable and are never subtracted.

- `U[t] = log(um_sum_open_interest_value[t])`
- `C[t] = log(cm_sum_open_interest[t])`
- `dU[t] = U[t] - U[t-12]`
- `dC[t] = C[t] - C[t-12]`
- relative purge: `P[t] = dU[t] - dC[t]`

USD-M uses notional OI value; COIN-M uses contract count. Their one-hour log
changes are dimensionless. This avoids comparing USDT notional directly with a
BTC-valued COIN-M field.

For clean completed bar `t`, three-bar taker states are:

- `TU[t] = mean(log(um_taker_ratio[t-2:t]))`
- `TC[t] = mean(log(cm_taker_ratio[t-2:t]))`
- COIN-M sell stress: `S[t] = -TC[t]`
- COIN-M-specific sell gap: `G[t] = TU[t] - TC[t]`

Zero/missing taker ratios and zero/missing OI are unavailable, never clipped or
forward-filled. The current row and required lookbacks must be complete, and
every unavailable row starts a 24-bar post-gap quarantine.

## Frozen setup and reclaim sequence

All quantiles are strictly lagged rolling quantiles over the prior 8,640 bars,
requiring 2,016 available past observations. The current row is excluded.

The only support-varying parameter is `Q` in:

`{0.80, 0.85, 0.90, 0.925, 0.95}`.

A setup begins only on a false-to-true transition satisfying:

1. `dC[t] < 0`;
2. `P[t] >= prior_qQ(P)`;
3. `S[t] >= prior_q90(S)`;
4. `G[t] >= prior_q90(G)`.

After a setup, scan at most the next 12 completed metrics rows. Take the first
clean row `k` satisfying all three reclaim conditions:

1. `cm_taker_ratio[k] >= 1`;
2. `cm_taker_ratio[k] >= um_taker_ratio[k]`;
3. `log(cm_OI[k] / cm_OI[k-1]) >= 0`.

If no such row appears, the setup expires without a trade. New setups do not
replace an active setup. Positions are globally non-overlapping.

The first tradable entry is `k+2` open, leaving one complete five-minute
availability bucket. Exit is the open after 144 held bars. Signal construction
does not read that open or any post-entry OHLC.

## Outcome-blind support gate

Choose the highest frozen `Q` passing every gate:

- at least 100 non-overlapping train events;
- at least 20 in 2021 partial and 50 in 2022;
- at least 75 in 2023 and 30 in each 2023 half;
- confirmation rate between 5% and 80%;
- no month above 15% of selected events;
- signal Jaccard no greater than:
  - CM-only OI `0.75`,
  - no taker gap `0.80`,
  - no reclaim `0.25`,
  - no OI-stop `0.80`,
  - matched USD-M purge `0.20`,
  - one-hour delayed signal `0.05`,
  - one-day shifted signal `0.02`.

Support failure rejects ICDR without loading a strategy return.

## Frozen falsification controls

1. exact short flip on the primary clock;
2. CM-only OI contraction instead of the relative purge;
3. remove the cross-collateral taker-gap threshold;
4. enter from the setup without reclaim confirmation;
5. remove the nonnegative one-bar COIN-M OI confirmation;
6. matched USD-M contraction/sell/reclaim sequence;
7. primary signal delayed one hour;
8. primary signal shifted one day;
9. fixed-seed random side on the primary clock.

Every control reserves its own opportunity clock before split slicing. A
control may falsify the mechanism but cannot replace ICDR after outcomes open.

## Strict staged evaluation

Stage 1 physically parses execution OHLC and funding only through
`2022-12-31 23:55`. Stage 2 may open 2023 only if the unchanged Stage-1 policy
passes every gate.

Every table must show **absolute return, full-clock CAGR, strict MDD,
CAGR/strict-MDD, and trade count**. Strict MDD uses the global/pre-entry high
water, entry and hypothetical liquidation costs, favorable-before-adverse held
OHLC, exact realized funding timing and frozen settlement mark, and scheduled
exit cost.

The candidate advances only if:

- train and 2023 each have positive absolute return, CAGR/MDD at least 3,
  strict MDD at most 15%, weekly-cluster one-sided `p <= 0.10`, and mean gross
  underlying move at least 20 bp;
- train has at least 80 trades and 2023 at least 60;
- 2021 partial, 2022, 2023 H1, and 2023 H2 are individually positive with
  their frozen minimum trade counts;
- 10 bp/notional/side stress remains positive in train and 2023;
- primary minimum train/selection ratio beats every mechanism-removal control;
- neither stale clock nor random side fully qualifies.

Any failure retires the exact candidate. Direction, quantiles, units,
confirmation, delay, hold, and gates may not be repaired.

## Orthogonality and RLLM boundary

Trade/PnL orthogonality is tested **only after standalone passage**. The frozen
limits are exact-entry Jaccard `<=0.02`, candidate entries near existing entries
within six hours `<=0.25` (`<=0.10` preferred), occupied-position Jaccard
`<=0.15`, absolute daily-PnL Pearson `<=0.30`, and at least ten nonzero PnL
days. Portfolio promotion also requires synchronized marginal improvement.

Only after deterministic passage may a compact RLLM receive symbolic purge
rank, sell-stress rank, sell-gap rank, bars since setup, reclaim flags, current
position, and time to exit. It may abstain or size the fixed long. It may not
reverse the side or redesign the base event.
