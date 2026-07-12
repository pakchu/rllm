# Path-memory bidirectional alpha search (2026-07-12)

## Standalone scan
- Features: return-sign autocorrelation, variance ratio, sign entropy, directional efficiency, semivolatility skew.
- 2,486 bidirectional variants, 6bp/side, 0.5x.
- No standalone candidate cleared test/eval ratio>=2.5.
- Best standalone test/eval minimum: variance-ratio reversion, ratios 1.50/1.75; failed 2026.

## Alphaization as a fixed gate
The path features were then used only as train-fit gates on the already fixed `funding_relief_vs_fx_stress` bidirectional candidate. Base entry thresholds, TP4%, SL2.5%, cap288 and stride6 were unchanged.

Selected test-only gate:
- `pm_eff_72 <= -0.1633119583` (train q0.10)
- applies to both long and short entries

`pm_eff_72` is signed 72-bar displacement divided by the sum of absolute 5m returns. The gate therefore permits only highly negative, directionally efficient paths.

| split | return | CAGR | strict MDD | ratio | L/S | win L/S | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.31% | 0.08% | 14.72% | 0.01 | 57/34 | 38.6/55.9% | 0.09 |
| test2024 | 12.66% | 12.64% | 2.01% | **6.28** | 12/6 | 66.7/50.0% | 2.37 |
| eval2025 | 11.82% | 11.83% | 2.75% | **4.30** | 17/5 | 52.9/80.0% | 2.05 |
| ytd2026 | -1.44% | -3.43% | 4.33% | **-0.79** | 27/3 | 48.1/33.3% | -0.22 |

## Verdict
This is a mechanical live-grade test/eval candidate but not a robust/live strategy: train edge is absent, OOS samples are sparse, and 2026 fails. Keep candidate-only as evidence that directional-efficiency state can concentrate historical edge. Do not retune the q0.10 gate using 2026.

## Artifacts
- `training/search_path_memory_bidirectional_alpha.py`
- `results/path_memory_bidirectional_alpha_scan_2026-07-12.json`
- `training/search_bidirectional_path_gate_alpha.py`
- `results/bidirectional_path_gate_alpha_scan_2026-07-12.json`
