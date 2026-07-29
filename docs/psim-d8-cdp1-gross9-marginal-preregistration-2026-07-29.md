# PSIM-D8-CDP1 Gross9 marginal preregistration

## Decision

The fixed `CDP_S50_G05` standalone policy passed its frozen 2022 selection and
untouched 2023 veto. It is **not promoted**. The remaining Gross9 interaction
question is frozen by
`results/psim_d8_cdp1_gross9_marginal_preregistration_2026-07-29.json`.

## Frozen portfolio question

- Baseline: the frozen Gross9 portfolio at gross `9.0`.
- Candidate weights: `0.25`, `0.50`, `0.75`, `1.00`.
- Selection: 2022 only.
- Future veto: the already-selected portfolio cell on 2023 only.
- Same-gross comparator: multiply every Gross9 weight by
  `(9 + candidate_weight) / 9`.
- Candidate execution: causal next-bar open, 288 five-minute bars, 0.5x unit
  leverage, exact funding, 6 bp/side base cost and 10 bp/side stress cost.
- Risk accounting: shared-clock, same-BTC OHLC strict intraposition MDD.
- Independence controls: exact entry and occupied-bar overlap against every
  Gross9 sleeve, plus daily marked-return correlations.

Future data cannot rerank, select rank 2, change a weight, or repair the CDP1
state machine.

## Authority pause

The two exact Gross9 market payloads are currently absent:

- `cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
  (`a77cd0...b990c`, 66,696,659 bytes)
- `cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz`
  (`dbc9e5...0192`, 72,898,508 bytes)

A DB-derived reconstruction was tested only as recovery evidence. It changed
the frozen train signal counts (`markov 131 vs 143`, `rex-taker 269 vs 274`,
`cand-rex 329 vs 308`) and therefore is not an admissible substitute. The
portfolio interaction remains unopened and paused until the exact bytes are
restored.
