# CTHD-1 rejection — 2026-07-18

CTHD-1 is rejected at sealed Stage 1. Calendar 2023 remains physically unopened.

## Primary result: 2021–2022

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross underlying | Weekly sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | -13.5192% | -7.0096% | 25.3489% | -0.2765 | 156 | -14.2696 bp | 0.6701 |
| 10 bp/notional/side | -18.7645% | -9.8757% | 26.2134% | -0.3767 | 156 | -14.2696 bp | 0.4875 |

Contained subperiods both failed:

- 2021: -11.9319% absolute, -11.9395% CAGR, 25.3489% strict MDD,
  123 trades, -18.2286 bp mean gross underlying return.
- 2022: -1.8024% absolute, -1.8036% CAGR, 9.3345% strict MDD,
  33 trades, +0.4869 bp mean gross underlying return.

## Diagnosis

The source state is real but the proposed cross-asset transmission is not:

- SKEW alone, low VIX alone, and the tail pair all lost money.
- `VVIX/VIX` alone was only +1.6180% absolute with 32.3470% strict MDD and a
  0.0249 CAGR/MDD ratio.
- A one-release delay was only +1.9694% absolute with 30.9116% strict MDD and a
  0.0317 ratio.
- The exact direction flip also lost after costs. This is not a simple sign
  error; the primary mean gross underlying return was already negative.

The central premise—SPX tail-hedge pressure relative to visible VIX predicts the
next Cboe-session BTC direction—therefore has no usable Stage-1 edge. It fails
return, risk, statistical-significance, gross-edge, stress-cost, subperiod, and
mechanism-margin gates.

## No-repair decision

No direction flip, threshold change, holding-period change, component
substitution, BTC regime gate, or 2023 inspection is permitted. Those would be
post-outcome repairs to the same hypothesis. CTHD-1 remains a documented null
result, and the next search must use a genuinely different mechanism.

Integrity:

- evaluator SHA-256:
  `7bdb67fc82b46cfbcca8bdd076b196cf84a9bca9662dd12223b8508939ec6fd5`
- Stage-1 manifest:
  `22b07be2336bc56e92ff36f96cf87cfd4695298e36fe94035304c166192a2b69`
- Stage-1 JSON SHA-256:
  `6f2791c0c373202273bbdd51e81d5c5b000e15a48db31c2058aac3e471f57cdc`
