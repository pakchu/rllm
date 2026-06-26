# Sparse setup failure cluster mining (2026-06-26)

## Purpose

The previous fixed failure-regime veto test did not improve test/eval enough. This pass mines signal-time descriptors that separate:

- **train-good events**: train folds with realized utility >= `0.25%` and MAE <= `2.5%`.
- **test-bad events**: 2025 test folds with realized utility <= `-0.25%`.

This is **diagnostic only**. It uses 2025 realized outcomes to mine candidate failure descriptors, so any mined rule is test-informed and not deployment-valid until promoted through a fresh nested train/test/eval protocol.

Implementation:

- `training/sparse_setup_failure_cluster_miner.py`
- `tests/test_sparse_setup_failure_cluster_miner.py`

Input sparse report:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/train_discovery_report.json`

Output artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/failure_cluster_miner.json`

## Main run

Command shape:

```bash
PYTHONPATH=. .venv/bin/python training/sparse_setup_failure_cluster_miner.py \
  --sparse-report results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/train_discovery_report.json \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/failure_cluster_miner.json \
  --train-folds-json "$(tr -d '\n' < results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/train_folds.json)" \
  --test-folds-json "$(tr -d '\n' < results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/test_folds.json)" \
  --eval-folds-json "$(tr -d '\n' < results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/eval_folds.json)" \
  --candidate-limit 80 --max-features 140 \
  --good-min-utility-pct 0.25 --good-max-mae-pct 2.5 \
  --bad-max-utility-pct -0.25 --min-cluster-rows 30 --top-k 80
```

Rows:

| Bucket | Count |
| --- | ---: |
| All candidate events | 212,431 |
| Train-good | 68,967 |
| Test-bad | 12,826 |
| Other | 130,638 |

## Top mined descriptors

The miner ranks one-feature veto descriptors by `coverage_edge = bad_coverage - good_block_rate`.

| Rank | Feature | Rule | Bad coverage | Train-good block | Edge | Interpretation |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `mkt__htf_3d_return_4` | `<= 0.005632` | 75.92% | 43.90% | 32.02% | 2025 bad events cluster in weak/flat 3-day HTF return regimes. |
| 2 | `pa__pa_ext_288_extreme_bar_overlap_pct` | `>= -0.062823` | 75.00% | 45.57% | 29.42% | Bad events often occur near overlapping range extremes rather than clean expansion. |
| 3 | `pa__pa_ext_576_to_min_low_pct` | `<= 0.047842` | 75.00% | 46.12% | 28.88% | Bad events cluster close to the rolling low / insufficient distance from downside extreme. |
| 4 | `pa__pa_ext_144_max_high_bar_spread_pct` | `<= 0.010356` | 75.07% | 47.13% | 27.94% | Bad events cluster when rolling-high bar spread is compressed. |
| 5 | `wave__cvd_mom_96` | `>= -0.003752` | 75.00% | 49.52% | 25.48% | Flow/CVD state helps describe failed setup regions, but blocks many good events too. |

Long-side descriptors are nearly identical because the sparse pool is mostly long. Short-side descriptors have much smaller sample size (`1,060` train-good / `178` test-bad) and should be treated as lower-confidence.

## Interpretation

The strongest signal is not a directional entry by itself. It is a **failure-zone descriptor**:

- Weak multi-day HTF return plus compressed/overlapping rolling extrema separates many 2025 losses from train winners.
- This matches the repeated observation that direct sparse triggers degrade in chop/range-transition regimes.
- The descriptor still blocks 40-50% of train-good events, so naive vetoing may destroy too much opportunity.

## Next validation step

Promote the mined descriptors into a fixed TTE veto test:

1. Discover/fit using train only.
2. Select among descriptor-veto candidates on test only.
3. Refit or freeze the selected candidate per protocol.
4. Evaluate on untouched 2026H1.

First candidate to test:

```text
mkt__htf_3d_return_4 <= 0.0056321971819943695
```

Expected value of this step is not immediate deployment. It checks whether failure-zone mining produces a real out-of-sample reduction in strict MDD without collapsing trade count.
