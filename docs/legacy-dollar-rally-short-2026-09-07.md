# Recovered dollar/rally standalone short — 2026-09-07

## Fixed candidate

Original top[0] from `results/short_base_alpha_scan_fast2_2026-07-08.json`. No gate/hold/stride tuning:
- `dxy_momentum >=0.0021818982809893497`.
- `htf_1d_return_4 >=0.016096783732847175`.
- Standalone short,1x entry notional, next5m open on original hourly stride phase,12h hold.
- No take profit or stop loss. Realized funding, actual notional entry/exit costs.

## Original-clock replay

| Period | 6bp return | MDD | Trades | 10bp return |
|---|---:|---:|---:|---:|
| 2024 | 37.97% | 14.54% | 66 | 30.92% |
| 2025 | 4.64% | 15.23% | 47 | 0.78% |
| full2026H1 | 13.73% | 9.33% | 33 | 10.79% |
| recent | 2.46% | 9.70% | 12 | 1.49% |
| since_aug4 | -4.24% | 7.91% | 5 | -4.63% |
| september_only | 0.00% | 0.00% | 0 | 0.00% |

Recent is June1–September5 00:00UTC, overlapping full2026H1. September-only covers September1–4, not the whole month.

## Timing / source sensitivity

Requiring DXY availability at each historical signal leaves the original historical results unchanged.1h-delayed DXY remains positive in2024/2025/full2026H1, but the recent June–September variant returns-1.39% versus original+2.46%.
24h-delayed DXY makes all three historical periods negative. This is a timing-sensitive intraday hypothesis, not a daily macro allocation. Exact live-arrival parity and execution latency are not proven.

## Decision

Retain as a disabled standalone research candidate. It has positive original-clock base/stress returns in three historical periods and the overlapping recent aggregate, but small2025 margin, weak newest slice, and timing sensitivity prevent a robust live claim.
This is a recovered older idea, not a newly independent signal family. Existing G9/macro correlation and marginal value are not established by standalone profits.
No live change. Disabled config: `configs/shadow/legacy_dollar_rally_short_2026-09-07.json`.

## Source / reproducibility

`training/audit_legacy_dollar_rally_short.py` rebuilds the frozen candidate and source-availability/delay sensitivities.
`training/evaluate_legacy_short_september.py` keeps the original global stride phase in the recent extension.
`research/legacy_dollar_rally_short` and `research/legacy_short_september` contain registrations, exact metrics, trade clocks and source hashes. All periods have prior research exposure; no pristine OOS claim.
