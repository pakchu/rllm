# Event-action feature drift audit — 2026-06-24

## Goal

Explain why direct LLM selectors, ridge, IC, and pairwise rankers all fail on 2026 despite using leakage-safe event-action candidate data.

## Audit artifact

Script: `training/event_candidate_drift_audit.py`

Input:

- `data/event_action_compressor_ranker_all_2022_2026_2026-06-24.jsonl`

Output:

- `results/event_action_compressor_ranker_drift_audit_2026-06-24.json`

The audit uses reward only for diagnosis, not for training or selecting a policy.

## Overall yearly reward distribution

| Year | Rows | Mean utility | Median | Positive frac | P90 | P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 29,200 | -0.011149 | -0.006150 | 0.290240 | 0.010987 | -0.038877 |
| 2023 | 29,200 | -0.007557 | -0.004062 | 0.285856 | 0.006746 | -0.026122 |
| 2024 | 29,280 | -0.009266 | -0.005148 | 0.309904 | 0.009594 | -0.033677 |
| 2025 | 29,200 | -0.007717 | -0.004410 | 0.305479 | 0.007679 | -0.027048 |
| 2026 | 11,940 | -0.008459 | -0.005024 | 0.302764 | 0.007890 | -0.029295 |

Observation: 2026 candidate reward distribution is not globally worse than 2024/2025. The failure is not simply that all 2026 candidates are bad.

## Candidate group drift

Largest family/side/hold drifts are small, usually around 0.001-0.002 utility. Examples:

- `family=htf_structure_break`: prior mean -0.006324 → 2026 -0.008482, delta -0.002158
- `side=SHORT`: prior mean -0.009315 → 2026 -0.007778, delta +0.001538
- `side=LONG`: prior mean -0.008621 → 2026 -0.009159, delta -0.000538
- `hold_bars=432`: prior mean -0.012761 → 2026 -0.011978, delta +0.000784

Observation: candidate group mix is not the primary explanation.

## Feature IC drift

The strongest signal is feature direction instability / sign flips.

| Feature | Prior mean IC | 2026 IC | Delta | Sign flip |
| --- | ---: | ---: | ---: | --- |
| action_side_sign | +0.023477 | -0.041382 | -0.064859 | yes |
| taker_imbalance | -0.004989 | +0.028469 | +0.033457 | yes |
| bb_z | -0.005731 | +0.022753 | +0.028484 | yes |
| rsi_norm | -0.006770 | +0.021591 | +0.028362 | yes |
| range_pos | -0.003284 | +0.011113 | +0.014397 | yes |
| htf_1d_return_1 | -0.002728 | +0.009992 | +0.012720 | yes |
| usdkrw_zscore | -0.001413 | +0.007487 | +0.008900 | yes |
| htf_1w_return_4 | +0.000040 | +0.056700 | +0.056661 | no |
| volume_zscore | -0.000447 | -0.026858 | -0.026411 | no |
| htf_4h_return_4 | +0.007159 | +0.026852 | +0.019694 | no |

Observation: 2026 regime changes the meaning of direction/price-action features. A static ranker trained across pre-2026 years learns a mixed or stale sign and then fails.

## Token drift

Token mean utility drifts are smaller than feature IC drifts. Examples:

- `side_trend_24=strong_up`: prior -0.012301 → 2026 -0.007531
- `trend_24=strong_down`: prior -0.011661 → 2026 -0.015359
- `htf_1w=strong_up`: prior -0.009206 → 2026 -0.005689
- `kimchi_level=up`: prior -0.009182 → 2026 -0.006586

Observation: deterministic coarse tokens are less expressive than raw feature IC shifts. The LLM compressor should probably emit explicit regime-conditioned feature sign guidance, not just static state buckets.

## Conclusion

The current bottleneck is regime-conditioned feature sign instability, not only model capacity or label format.

Next work should test:

1. Rolling/recent-window rankers that can adapt feature signs using only data before the prediction period.
2. Regime-conditioned rankers that split by price-action/macro regimes before fitting feature signs.
3. LLM compressor targets that explicitly describe which feature families are currently reliable or inverted, then feed those tokens into the ranker.

Do not spend more compute on direct final-action LLM SFT until this drift problem is handled.
