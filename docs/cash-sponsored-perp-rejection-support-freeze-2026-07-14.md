# CSPR-12 support decision and event-clock freeze — 2026-07-14

## Decision

CSPR-12 passed its preregistered **outcome-blind support gate** at the `0.50`
Spot/USD-M coherence quantile. No future OHLC path, trade return, win rate,
CAGR, or MDD was opened in this stage.

The selected quantile was not chosen by performance. It is the highest member
of the frozen `{0.50, 0.60, 0.70, 0.80, 0.90}` grid satisfying every count,
side-balance, ablation, and stale-input criterion. All stricter quantiles failed
the frozen count floors.

## Selected support

- raw primary events: **892**
- fixed-hold non-overlapping events: **850**
- 2020 / 2021 / 2022 / 2023: **375 / 194 / 194 / 87**
- 2023 H1 / H2: **41 / 46**
- long / short: **262 / 588** (`30.82% / 69.18%`)
- primary/no-centroid raw-clock retention: **0.420**
- primary/no-USD-M-event-confirmation retention: **0.650**
- 1-hour / 24-hour stale-Spot Jaccard: **0.0597 / 0.0573**

The event-rate decline from 375 events in 2020 to 87 in 2023 is a material
pre-return structural-decay warning. It does not invalidate support, but the
fixed evaluator must reject the candidate if 2023 performance or either 2023
half fails; no threshold relaxation is permitted.

## Frozen artifacts

- preregistration commit:
  `b45fca0eeecc942df3f37b9f057697a117871cc1`
- support JSON SHA-256:
  `6b3d67a2a1ee4feccbab8368ae038929ad22ef9ebe99055d89a00abcb0ca038a`
- selected event clock SHA-256:
  `353e14cc09b79960938802c9882ba36527e9f4c4819f8a492312fdefffdf1c0f`
- clock manifest SHA-256:
  `37132d281942ac8f0e72edf239358ac74b9622db6401581461dca4ae4494caa2`

The clock contains signal, next-open entry, fixed 12-bar exit timestamps, side,
and positions only. It contains no price or return columns. The first signal is
`2020-01-11 17:50:00`; the last is `2023-12-30 21:40:00`.

## Next gate

Implement and commit the evaluator and every frozen control before opening any
return. Then evaluate 2020–2022 train and 2023 selection under the preregistered
cost, full-clock CAGR, strict held-path MDD, weekly-cluster randomization, and
component-control rules. Sealed 2024+ data remains unopened unless that gate
passes unchanged.
