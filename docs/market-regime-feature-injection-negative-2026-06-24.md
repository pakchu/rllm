# Market-regime feature injection negative result (2026-06-24)

## Purpose

The current best PA-ext ranker needs stronger alpha, so this pass tested adding longer-horizon market regime information:

- rolling returns
- realized volatility proxy
- drawdown/runup
- range position
- range width
- trend-to-vol

Windows: 1D/3D/7D/14D/30D in 5m bars (`288,864,2016,4032,8640`).

## Implementation

- `training/augment_event_candidate_market_regime.py`
  - Builds past-only rolling market regime features.
  - Joins to event candidate rows via backward-asof.
- `training/event_candidate_feature_stability_audit.py`
  - Audits candidate feature/reward relation by calendar year.
- `training/filter_event_candidate_features.py`
  - Creates stable feature subsets for follow-up validation.

## Full mreg injection

Data:

- `data/event_action_compressor_ranker_all_2022_2026_paext_mreg_2026-06-24.jsonl`
- Rows matched: 128,820 / 128,820
- Added features: 35

Walk-forward report:

- `results/event_candidate_pairwise_walkforward_paext_mreg_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`

Result:

- CAGR: -5.10%
- Strict MDD: 30.13%
- CAGR/MDD: -0.17
- Trades: 86

## Stability audit

Audit report:

- `results/event_candidate_mreg_feature_stability_audit_2026-06-24.json`

Stable relationships were mostly volatility/range-width features where high volatility/range width had lower candidate reward across years:

- `mreg_2016_range_width_pct`
- `mreg_288_vol_proxy`
- `mreg_288_range_width_pct`
- `mreg_2016_vol_proxy`
- `mreg_864_range_width_pct`
- `mreg_4032_range_width_pct`
- `mreg_864_vol_proxy`
- `mreg_4032_vol_proxy`
- `mreg_8640_range_width_pct`
- `mreg_8640_vol_proxy`

## Stable subset injection

Data:

- `data/event_action_compressor_ranker_all_2022_2026_paext_mreg_stable_2026-06-24.jsonl`
- Feature count per row: 94

Walk-forward report:

- `results/event_candidate_pairwise_walkforward_paext_mreg_stable_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`

Result:

- CAGR: -2.43%
- Strict MDD: 33.81%
- CAGR/MDD: -0.07
- Trades: 94

## Conclusion

Directly injecting long-horizon market regime features into the current pairwise ranker is harmful. Even stable univariate mreg features destabilize validation selection. These features may still be useful as:

1. a high-level abstain/gating signal selected with enough prior folds, or
2. LLM/text compressor context where the model emits coarse regime summaries rather than raw numeric features.

Do not add raw mreg features to the current ranker by default.
