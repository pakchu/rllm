# CIHM-1 outcome-blind support freeze — 2026-07-18

## Result

**PASS — Stage 1 may open only after the evaluator itself is frozen.**

The support build read the frozen Cboe option-flow panel and the already
preregistered policy clock.  It loaded **0 BTC market rows, 0 funding rows, and
0 return rows**.

## Primary clock support

| Window | Events | Shorts | Months | Largest month |
|---|---:|---:|---:|---:|
| 2021 | 74 | 74 | 12 | 10 (13.51%) |
| 2022 | 78 | 78 | 12 | 10 (12.82%) |
| Stage 1, 2021–2022 | 152 | 152 | 24 | 10 (6.58%) |
| sealed 2023 H1 | 30 | 30 | 6 | 8 |
| sealed 2023 H2 | 35 | 35 | 6 | 8 |
| sealed 2023 | 65 | 65 | 12 | 8 (12.31%) |

All preregistered support conditions pass:

- Stage 1 >= 150;
- each Stage-1 year >= 70;
- sealed 2023 >= 60;
- each sealed half >= 25;
- Stage-1 and full-2023 one-month concentration <= 15%;
- short only.

## Frozen control clocks

| Clock | Full frozen-horizon events | Entry Jaccard vs primary |
|---|---:|---:|
| primary | 258 | 1.0000 |
| institutional-gap only | 271 | 0.2747 |
| VIX-call-pressure only | 266 | 0.3266 |
| index-share only | 244 | 0.4261 |
| level composite | 336 | 0.3562 |
| one-release delay | 258 | 0.0840 |
| seven-release placebo | 257 | 0.1731 |

The ledger fixes every observation date, next-session 09:35 ET entry,
following-session exit, short side, source feature, rank, and score before any
BTC return is joined.

## Artifact identities

- Support manifest hash:
  `2fb3872b8e062e4719d3b3f087857561262c7c2bdbccb06eaf780923bb051ae1`
- Support JSON SHA-256:
  `df7ce11f92d50c52ffd61afd7cab6a2a3da5696e08ba45b886e3c37bec7f0a93`
- Frozen clock-ledger SHA-256:
  `5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b`

No support count, control clock, formula, direction, threshold, timing, size,
cost, or gate may change after Stage-1 outcomes are opened.
