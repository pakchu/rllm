# CSPR-12 preregistration — 2026-07-14

## Status

**Support-only and outcome-blind.** This commit freezes source checks, the
economic rule, the support-calibration grid, controls, execution timing, and
the later return gate before any CSPR-specific return is opened.

- name: **CSPR-12 — Cash-Sponsored Perpetual Rejection**
- support artifact:
  `results/cash_sponsored_perp_rejection_support_2026-07-14.json`
- inspected source horizon: strictly before `2024-01-01`
- opened CSPR outcomes: **none**

## Economic object

This is not another spot/perpetual price-basis residual. It asks which venue's
aggressors are being marked right at the close of the same completed auction.

At a completed five-minute bar, cash sponsorship direction `d` is the sign of
Binance Spot signed taker quote flow. CSPR requires:

1. strong Spot and USD-M flow coherence versus each venue's strictly lagged
   30-day baseline;
2. Spot flow and Spot price move in direction `d`;
3. USD-M price also moves in direction `d`;
4. both USD-M signed capital and signed aggregate-event count point against
   `d`, so the futures aggressors are marked wrong by their own close;
5. the winning Spot aggressor side obtained the better execution centroid and
   the Spot close settled beyond both side centroids.

For a long, `buyer centroid < seller centroid < close`; the short rule is the
exact mirror. “Cash-sponsored” and “adverse-selected” are economic
interpretations of public executions, not account-identity labels.

The traded instrument is Binance USD-M `BTCUSDT`; Spot supplies the causal cash
direction only. The signal appears after both five-minute bars complete, entry
is the next USD-M five-minute open, and exit is fixed at the open 12 bars later.
There is no persistent or feature-dependent exit in v1.

## Source and quarantine

- verified Spot 1m→5m feature SHA-256:
  `d558239fa7085083aa002b7898b632df0774425719467709680ecb99718035a9`;
- verified USD-M aggTrade feature and official kline manifests are replayed by
  hash;
- Spot and USD-M sources stop before 2024;
- every missing/incomplete source bar and the next 24 bars are quarantined;
- the signal bar must avoid the joint feature quarantine; a later Spot or
  aggTrade outage cannot cancel an already entered fixed-hold trade because
  v1 does not consume either feature source after entry;
- rolling coherence thresholds use clean observations through `t-1` only,
  with an 8,640-bar window and 2,016 minimum observations.

## Outcome-blind support calibration

The only support-varying parameter is the common Spot/USD-M coherence
percentile in `{0.50, 0.60, 0.70, 0.80, 0.90}`. Select the highest percentile
passing all frozen floors:

- at least 300 non-overlapping events total;
- at least 40 in every year 2020–2023;
- at least 30 in each 2023 half;
- each executed side at least 25%;
- centroid removal and USD-M event-count removal must each enlarge the raw
  clock enough that primary retention is at most 80%;
- 1-hour and 24-hour stale-Spot placebos must each have event Jaccard at most
  0.25 with the primary clock.

If no percentile passes, CSPR is rejected before returns. Counts, timestamps,
feature values, and overlap may be inspected in this stage; future OHLC paths,
returns, win rates, CAGR, and MDD may not.

## Frozen controls for the evaluator

1. exact direction flip on the primary event clock;
2. no-centroid clock;
3. no USD-M event-count confirmation clock;
4. Spot-only cash/centroid clock;
5. USD-M rejection-only clock;
6. role-swapped accepted-futures/rejected-cash clock;
7. Spot inputs stale by 1 hour and 24 hours;
8. signal delayed one full five-minute bar.

Controls use their own predeclared clocks where applicable. They may falsify
the economic mechanism but may not replace the primary after outcomes open.

## Frozen return and qualification gate

- train: `2020-01-01..2022-12-31`;
- selection: full 2023, also H1/H2;
- sealed test: full 2024;
- sealed eval: full 2025;
- 2026 YTD: report only after an untouched OOS pass;
- leverage `0.5x`;
- fee `5 bp` plus slippage `1 bp` per notional side;
- exact multiplier `(1-0.0003) × (1+0.5r) × (1-0.0003)`;
- full-clock CAGR including idle time;
- strict held-path MDD with favorable extreme first, adverse extreme second,
  and no later high/low from the scheduled exit bar;
- weekly entry-cluster Rademacher test, 100,000 draws, seed `20260714`.

CSPR advances only if train and full 2023 each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, and weekly one-sided
`p<0.10`; both 2023 halves must be positive with at least 30 trades, full 2023
must have at least 80 trades, and the primary must beat every direct component
control on the frozen minimum train/selection ratio. Failure rejects v1 without
threshold, centroid, direction, hold, stop, or side repair.

Only after this gate may a compact Gemma/RL policy receive the CSPR state to
size or abstain. The model may not create a new direction or recover sealed
outcomes through identifiers.
