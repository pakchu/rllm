# Short-adaptation PA-ext walk-forward (2026-06-24)

## Purpose

The 12M fit / 6M validation / 6M test protocol showed validation-to-test decay. This pass checks whether faster adaptation helps after price-action extreme features were injected.

Input: `data/event_action_compressor_ranker_all_2022_2026_paext_2026-06-24.jsonl`.

## 6M fit / 3M validation / 3M test

Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_2026-06-24/report.json`

- Trades: 206
- CAGR: 4.32%
- Strict MDD: 34.14%
- CAGR / strict MDD: 0.13
- Mean trade return: +0.111%
- p-value approximation: 0.556

This is the first broad rolling aggregate in this lane that turned positive, but the drawdown and statistical weakness make it non-deployable.

Notable folds:

- 2022Q4 test: `CAGR 104.3 / MDD 11.8 / ratio 8.82`
- 2024Q4 test: `CAGR 135.3 / MDD 10.4 / ratio 13.05`
- 2024Q1 test: `CAGR -50.4 / MDD 33.2 / ratio -1.52`
- 2025Q3 test: `CAGR -33.7 / MDD 14.3 / ratio -2.36`

## 6M / 3M / 3M with stricter validation gate

Gate settings:

- `min_val_cagr_pct=10`
- `min_val_ratio=1.0`
- `max_val_strict_mdd_pct=20`
- `min_val_t_stat=0.8`
- `max_val_p_value=0.45`
- `max_val_power_gap=500`

Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_statsgate_2026-06-24/report.json`

- Trades: 126
- CAGR: 5.39%
- Strict MDD: 17.36%
- CAGR / strict MDD: 0.31
- Mean trade return: +0.183%
- p-value approximation: 0.410

MDD improved materially but remains above 15%, and return is far below target. The stricter gate still selected losing next-test folds, especially 2025Q3.

## Conclusion

Shorter adaptation helps directionally: negative long-window aggregate became positive. But validation strength is still not a sufficient predictor of next-window survival.

The next bottleneck is regime relation stability: detect when learned feature/action relations are likely to invert or decay before the next test window. This should be treated as relation-flip detection, not another q/margin gate sweep.
