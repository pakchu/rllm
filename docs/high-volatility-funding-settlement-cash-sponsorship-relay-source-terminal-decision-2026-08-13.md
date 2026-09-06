# HVFSCS-6 terminal source-availability rejection

The HVFSCS-6 source evaluator was pushed before any incidence was calculated.
Its first execution stopped in source serialization parsing before support was
computed. The evaluator was then changed only to parse mixed ISO timestamp
precision, regression-tested, committed, and pushed before rerunning.

The unchanged source contract then produced zero source-valid events in every
split. The current PostgreSQL snapshot has `3,643,089` BTCUSDT perpetual
`1m` rows but no `5m` rows in `bars_binance`, while the preregistration requires
96 exact native `interval='5m'` rows before each settlement. Consequently every
one of the `4,296` actual funding settlements had zero qualifying perpetual
source rows. Some actual funding timestamps also carry millisecond offsets;
those timestamps correctly fail the exact unrounded spot grid rather than
being silently rounded.

| split | events | long | short |
|---|---:|---:|---:|
| train 2023H2 | 0 | 0 | 0 |
| test 2024 | 0 | 0 | 0 |
| eval 2025 | 0 | 0 | 0 |
| final to 2026-08-01 | 0 | 0 | 0 |

An immediate replay reproduced every panel, clock, control, manifest, and
result artifact byte-for-byte:

- source panel: `21f827dff7be28a1e403719b832fd3e2cb71439f0e7f9d5d29648701fa4d3b7f`
- source manifest: `6be01c15c7fecf17f2b3115f7509e16cb6bcc5c608705510299310fc7fd8f62d`
- primary clock: `c1179d374edbbd39fe43055d4b29d159ff133a97c3233cd43137160f15c2357d`
- result: `108991e4c4bd2492dc2438b9f0f0ca740621fcbeccb68bd520fb263564b2da07`

HVFSCS-6 is rejected unchanged. Aggregating `1m` rows into a replacement
perpetual source or rounding actual settlement timestamps would change the
frozen source/clock contract and is forbidden. Gross9 rows, execution prices,
funding PnL, strict MDD, CAGR, and RV20 remain unopened. No source, interval,
timestamp, path, variation, alignment, cash-confirmation, side, clock, hold,
subset, threshold, comparator, or diagnostic control may be repaired.
