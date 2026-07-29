# PSIM-D8 CDP1 2022 Top1 Freeze

Date: 2026-07-29  
Decision: `pass`  
Selected top1: `CDP_S50_G05`

## Selection result

| Candidate | Eligible | Trades | Long share | 6 bp/side return | Strict MDD | CAGR/MDD | 10 bp/side return |
|---|---|---:|---:|---:|---:|---:|---:|
| `CDP_S35_G05` | no | 76 | 18.42% | +3.15% | 27.96% | 0.113 | -2.92% |
| `CDP_S50_G05` | yes | 56 | 25.00% | +20.52% | 27.96% | 0.735 | +15.29% |

`CDP_S35_G05` failed both the frozen minimum 20% long-share requirement and
the positive stress-return requirement. It is retired and cannot replace the
selected candidate if the future veto fails.

`CDP_S50_G05` passed every frozen 2022 selection requirement and is the sole
candidate authorized for the untouched 2023 veto.

## Execution note

The initial execution attempt stopped before computing any candidate metric
because Binance `funding_time_utc` includes exchange-side millisecond jitter.
The frozen funding artifact separately supplies `mark_open_time_utc`, the exact
settlement-mark grid. A revisioned attempt bound that field before evaluation,
without changing candidates, thresholds, signals, costs, or ranking.

No 2023 market row or numeric funding row was parsed during selection.

Result hash:
`f4530edbe8f26ad5ab4cd549de24fcc9c4d1ffc0dde8d6d3e3420bd88dd4a76b`.
