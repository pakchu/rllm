# RIFT-96 preregistration — 2026-07-14

## Status

**Support-only and outcome-blind.** No RIFT return, future path, win rate,
CAGR, or MDD has been opened. This document and its implementation freeze the
economic object, sequence, support grid, controls, execution timing, and later
return gate first.

- name: **RIFT-96 — Refill Inference from Flow Topology**
- inspected feature horizon: strictly before `2024-01-01`
- direction: fixed long only
- entry: next USD-M five-minute open after a two-completed-bar sequence
- exit: fixed USD-M open 96 bars later (8 hours)

“Refill” is an inference, not direct order-book observation. The measurable
object is persistent positive execution pressure across completed Spot and
USD-M bars.

## Economic sequence

### Setup bar `t-1`

All Spot and USD-M inputs must be complete and unquarantined. Require positive
Spot close displacement from the geometric mean of aggressive buyer/seller
execution centroids, positive Spot and USD-M returns, positive Spot/USD-M
signed taker capital, and nonnegative USD-M event imbalance.

The setup score is:

`positive centroid mark × Spot path quality × USD-M crowd structure`, where

- `Spot path quality = minute price efficiency × minute flow efficiency ×
  rescaled flow/price alignment × (1 - minute flow-sign flip rate)`;
- `USD-M crowd structure = sqrt(event-notional HHI) × rescaled interarrival
  burstiness`.

The score must exceed its strictly lagged 30-day percentile.

### Confirmation bar `t`

On the next completed bar, Spot must remain above its execution-centroid mid,
Spot and USD-M returns must be nonnegative, and Spot price-path efficiency,
USD-M event concentration, and USD-M burstiness must each remain above their
strictly lagged 30-day medians. This second bar is mandatory; the setup bar
cannot trigger entry by itself.

The action is long at the next open `t+1`. The 8-hour hold targets a move large
enough to clear the approximately 12 bp underlying round-trip break-even at
the frozen 0.5x cost model; it is not a repair of CSPR's one-hour horizon.

## Outcome-blind support calibration

The only support-varying parameter is the setup-score percentile in
`{0.80, 0.85, 0.90, 0.925, 0.95, 0.975}`. Select the highest percentile that
passes all frozen floors:

- at least 300 fixed-hold non-overlapping events total;
- at least 40 in every calendar year 2020–2023;
- at least 30 in each 2023 half;
- every scheduled action is long;
- raw-clock Jaccard limits: same-bar `0.05`, no-path `0.40`, no-crowd `0.20`,
  centroid-free momentum `0.75`, Spot-only `0.20`, 1h/24h stale setup `0.10`,
  one-bar signal delay `0.05`, and CSPR primary `0.01`.

If no percentile passes, reject RIFT-96 before returns. The novelty limits
prevent relabeling plain momentum, CSPR, a static one-bar event, or USD-M
topology alone as the new mechanism.

## Frozen controls for the evaluator

1. exact short flip on the primary clock;
2. same-bar static setup, without mandatory confirmation;
3. no Spot path-quality score/confirmation block;
4. no USD-M concentration/burstiness score/confirmation block;
5. centroid-free matched momentum score;
6. Spot-only sequence using Spot quarantine only;
7. setup stale by 1 hour and 24 hours;
8. signal delayed one complete five-minute bar;
9. simple two-bar positive Spot/USD-M momentum.

Every control gets its globally reserved clock before return-window slicing.
Controls may falsify RIFT but may not replace it.
The exact direction flip is short (`-1`); primary and every other control are
long (`+1`). Component-removal controls use finite/source masks that exclude
the removed component rather than silently requiring its availability.

## Frozen return gate

- train: `2020-01-01 <= t < 2023-01-01`;
- selection: full 2023 and fixed H1/H2;
- 2024 test, 2025 eval, and 2026 YTD remain sealed;
- leverage `0.5x`, fee `5 bp`, slippage `1 bp` per notional side;
- exact multiplier `(1-0.0003)*(1+0.5r)*(1-0.0003)`;
- full-clock CAGR including idle cash;
- strict held-path MDD, favorable extreme before adverse, with exit-bar later
  high/low excluded;
- weekly entry-cluster Rademacher test, 100,000 draws, seed `20260714`.

RIFT advances only if train and full 2023 each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, one-sided weekly-cluster
`p<0.10`, and mean gross underlying move strictly above 12 bp. Each 2023 half
must be positive with at least 30 trades, full 2023 must have at least 80
trades, and the primary minimum train/selection ratio must beat every frozen
control. Failure rejects v1 without threshold, sequence, side, hold, or gate
repair. Only after an unchanged pre-2024 pass may RIFT become an RLLM state
token for sizing or abstention; a model may not reverse its fixed direction.
