# H8DM-1 preregistration — 2026-07-18

## Status

**PREREGISTERED; ALL BTC/FUNDING OUTCOMES SEALED**

This is one candidate, not a family search. No result-aware repair, direction
flip, component promotion, or new regime gate is permitted.

## Fixed mechanism

- source: point-in-time Federal Reserve H.8 dated archive pages;
- feature vintage: seasonally adjusted levels printed in each release;
- components: large-minus-small other-deposit growth, small-bank borrowings
  growth, and negative small-bank cash-asset growth;
- normalization: prior 104-release median/MAD only;
- score: equal mean of the three robust z-scores;
- event: absolute score at or above the prior-52-score q0.50, with at least two
  component signs agreeing;
- direction: positive stress SHORT, negative relief LONG;
- exclusions fixed before outcomes: 2020-10-02, 2023-03-31, 2023-06-30;
- entry: 17:00 New York on Thursday/Friday release dates, 45 minutes after the
  nominal 16:15 publication time;
- exit: exactly 48 hours later;
- size/cost: 0.5x, 6 bp/notional/side; 10 bp stress.

## Source-only threshold selection

| Tail q | Stage1 events | Sealed 2023 events | Support |
|---:|---:|---:|:---:|
| 0.90 | 20 | 2 | FAIL |
| 0.85 | 27 | 4 | FAIL |
| 0.80 | 35 | 6 | FAIL |
| 0.75 | 45 | 7 | FAIL |
| 0.70 | 53 | 10 | FAIL |
| 0.65 | 55 | 11 | FAIL |
| 0.60 | 60 | 16 | FAIL |
| 0.55 | 70 | 19 | FAIL |
| 0.50 | 75 | 24 | PASS |

q0.50 is the highest cell meeting the frozen density, direction, half-year,
and concentration requirements. No BTC or funding row was used.

## Frozen primary clock distribution

| Window | Events | Long | Short |
|---|---:|---:|---:|
| 2017_2019_source_history | 0 | 0 | 0 |
| 2020 | 28 | 16 | 12 |
| 2021 | 19 | 10 | 9 |
| 2022 | 28 | 2 | 26 |
| stage1 | 75 | 28 | 47 |
| 2023_h1 | 8 | 4 | 4 |
| 2023_h2 | 16 | 10 | 6 |
| 2023 | 24 | 14 | 10 |

## Evaluation order

1. freeze source and controls;
2. freeze evaluator source hash with zero outcome rows parsed;
3. open only 2020–2022 Stage1;
4. keep 2023 sealed unless every Stage1 gate passes unchanged;
5. inspect existing-alpha overlap only after standalone Stage2 and pooled pass.

## Identity

- source commit: `14ceca3`
- source panel SHA-256: `c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572`
- primary clock SHA-256: `20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40`
- preregistration manifest: `9e2c1674d46e83e4651f45c778c039c804128d5715915f04fb6263d896dc950a`
