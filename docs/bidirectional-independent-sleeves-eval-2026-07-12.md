# Bidirectional independent-sleeve evaluation (2026-07-12)

The fixed `funding_relief_vs_fx_stress` signals were re-evaluated with long and short sleeves allowed simultaneously. No entry threshold or exit was changed.

## Base allocation: 0.5x long + 0.5x short

| split | return | CAGR | strict MDD | ratio | L/S trades | L/S win | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 28.55% | 6.48% | 17.78% | 0.36 | 306/108 | 47.4/43.5% | 1.04 |
| test2024 | 20.13% | 20.09% | 4.13% | **4.87** | 37/24 | 56.8/54.2% | 1.91 |
| eval2025 | 15.27% | 15.28% | 6.01% | **2.54** | 54/8 | 55.6/50.0% | 1.69 |
| ytd2026 | 2.86% | 7.00% | 10.61% | 0.66 | 74/8 | 48.6/50.0% | 0.33 |

Independent sleeves remove the one-position opportunity-blocking loss: YTD2026 changes from -0.54% to +2.86%. It still fails the 2026 ratio>=5 requirement and remains strongly long-skewed.

## Test-only weight scan

On a coarse 0.25 grid with gross<=1.5, test2024 selected 0.75x long + 0.75x short:
- test: return 31.30%, ratio 5.08, MDD 6.15%
- eval: return 23.42%, ratio 2.63, MDD 8.93%
- 2026: return 3.90%, ratio 0.62, MDD 15.60%

Higher gross adds return but does not repair 2026 capital efficiency. The 0.5+0.5 allocation is the cleaner candidate baseline.

## Verification note

An initial simultaneous-path result was discarded because TP/SL exit return was double-counted after prior open-to-open marks. The final artifact applies only current-open-to-target return on the exit bar.

## Artifacts
- `training/evaluate_bidirectional_independent_sleeves.py`
- `results/bidirectional_independent_sleeves_eval_2026-07-12.json`
- `results/bidirectional_independent_sleeves_weight_scan_2026-07-12.json`
