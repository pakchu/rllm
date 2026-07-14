# RCR-144 frozen 2023 result — 2026-07-14

## Decision

RCR-144 passed outcome-blind support, then its source, 646-event clock, and
return evaluator were separately committed and hash-frozen before prices were
opened:

- preregistration commit: `889fd5d`;
- support commit: `19a386c`;
- event-clock commit: `b47dabb`;
- evaluator source commit: `0065788`;
- evaluator-freeze commit: `9e003c4`;
- evaluator source SHA256:
  `cdd9534a9002f699e903924a863901eadc2f57000b337c9cf2fdbf03acf0a680`;
- result artifact:
  `results/radial_composition_rotation_selection_2026-07-14.json`;
- result SHA256:
  `e89883bc28bcf3d45f2daffe3f11506ab070ffc68d4d327567a9094fb07988b7`.

**RCR-144 is rejected.** Calendar 2024, 2025, and 2026 shell outcomes remain
sealed.

## Frozen statistics

All results use next-five-minute-open entry, 144 held bars, scheduled-open exit,
0.5x, 5 bp fee plus 1 bp slippage per notional side, complete split-clock CAGR,
and held-path strict MDD.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| 2023 H1 | -4.11% | -8.12% | 15.83% | -0.51 | 300 | 0.57821 |
| 2023 H2 | -12.58% | -23.42% | 21.45% | -1.09 | 346 | 0.83169 |
| Q1 | +6.52% | +29.22% | 15.17% | 1.93 | 132 | 0.29620 |
| Q2 | -9.98% | -34.43% | 15.83% | -2.18 | 168 | 0.89223 |
| Q3 | -12.80% | -41.95% | 16.68% | -2.51 | 170 | 0.93911 |
| Q4 | +0.25% | +1.01% | 14.22% | 0.07 | 176 | 0.47098 |

It fails half-year return, CAGR/MDD, MDD, weekly significance, Q2/Q3
positivity, and control-dominance gates.

## Same-clock controls

| policy | H1 return | H1 ratio | H2 return | H2 ratio |
|---|---:|---:|---:|---:|
| RCR-144 | -4.11% | -0.51 | -12.58% | -1.09 |
| exact reverse | -29.03% | -1.58 | -25.69% | -1.70 |
| always long | -0.99% | -0.11 | -1.36% | -0.16 |
| always short | -31.27% | -1.59 | -34.14% | -1.51 |
| sign permutation | -9.77% | -1.07 | -32.46% | -1.63 |
| 5m price momentum | -14.58% | -1.13 | -30.01% | -1.66 |

RCR direction beats its exact reverse in both halves, but does not beat the
always-long control on minimum H1/H2 CAGR/MDD.

## Cost and stability diagnosis

The frozen multiplier is linear in raw return:

```text
net = (1 - 0.0003)^2 * (1 + 0.5 * raw_return) - 1
```

Solving it from mean trade return gives RCR pre-cost account-level directional
markout of approximately `+5.02 bp` per trade in H1 and `+2.35 bp` in H2,
versus fixed round-trip drag of about `6 bp`. Thus RCR is not merely a wrong-sign
feature: the exact reverse has symmetric negative pre-cost markout. The signal
is simply too weak and unstable to clear costs, with Q2 near zero pre-cost edge
and Q3 negative edge.

The apparent Q1 success is not statistically significant and closely matches
always-long exposure. Q4 is also explained better by always-long (`+14.57%`)
than by RCR (`+0.25%`). The predictive object therefore does not supply a
regime-stable, economically sufficient alpha.

## Locked consequence

RCR-144 may not be repaired by changing its threshold, side, 12-hour hold,
re-entry rule, venue aggregation, or composition formula after seeing these
returns. It may not be rescued with an LLM/RL gate or portfolio optimizer on
this clock. A next experiment must use a different predictive object and must
be preregistered before any new outcome window opens.
