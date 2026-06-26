# Episode feature stability audit (2026-06-27)

## Purpose

The recent episode-policy searches overfit quickly: 2025 validation could look attractive, while 2026 holdout failed. This audit adds a diagnostic pass that checks whether price-action episode features are sparse, duplicated, or directionally unstable across chronological splits before further optimization.

This is not a selector. It intentionally reads the eval split only to diagnose instability and leakage/overfit risk. Any production candidate must still be chosen without eval feedback and validated by strict bar simulation.

## Command

```bash
.venv/bin/python -m training.audit_episode_feature_stability \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/episode_feature_stability_train2024_test2025_eval2026jm_2026-06-27/report.json \
  --train-start 2024-01-01 --train-end '2024-12-31 23:59:59' \
  --test-start 2025-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --windows 36,72,144,288,576,2016,4032,8640 \
  --horizons 36,72,144,288,432 \
  --min-split-triggers 10 \
  --high-overlap-jaccard 0.80 \
  --top-k 80
```

## Split

- Train: 2024-01-01 through 2024-12-31
- Test: 2025-01-01 through 2025-12-31
- Eval: 2026-01-01 through 2026-06-01
- Input: regenerated `wavefull` 5m market data with macro context restored.

## Results

| Metric | Value |
| --- | ---: |
| Feature columns | 280 |
| Episode event columns | 208 |
| Event-side-horizon templates | 1,040 |
| Train/test sign flips | 473 |
| Test/eval sign flips | 333 |
| Positive test / negative eval | 160 |
| Event-rate drift >=3x | 15 |
| Eval-sparse templates | 15 |

## Key failure mode

The most tempting 2025 short features are not robust. They are sparse in 2026 and flip sign between splits.

Examples:

| Event | Side | Horizon | Train mean | Test mean | Eval mean | Flags |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pae_w8640_failed_mid_reclaim_short` | SHORT | 288 | -0.656% | +0.661% | -1.843% | eval_sparse, train_test_sign_flip, positive_test_negative_eval, event_rate_drift_3x |
| `pae_w8640_lower_high_mid_reject` | SHORT | 288 | -0.246% | +0.620% | -1.825% | eval_sparse, train_test_sign_flip, positive_test_negative_eval, event_rate_drift_3x |
| `pae_w8640_failed_mid_reclaim_short` | SHORT | 144 | +0.293% | +0.304% | -1.737% | eval_sparse, positive_test_negative_eval, event_rate_drift_3x |

Interpretation: the earlier high-performing 2025 short/reject candidates were likely a 2025 regime artifact plus sparsity. They should be blacklisted or heavily downweighted rather than further tuned.

## Stable diagnostic candidates

These are not final strategy picks; they are candidates for strict simulation because they were positive in train/test/eval in this diagnostic open-to-open return audit and had no stability flags.

| Event | Side | Horizon | Train n / mean | Test n / mean | Eval n / mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `pae_w2016_failed_mid_loss_long` | LONG | 288 | 205 / +0.024% | 163 / +0.071% | 81 / +0.704% |
| `pae_w2016_seq_bear_failed_bounce` | SHORT | 432 | 2368 / +0.083% | 3134 / +0.188% | 1468 / +0.487% |
| `pae_w4032_failed_breakout_short` | SHORT | 288 | 133 / +0.022% | 70 / +0.058% | 51 / +0.449% |
| `pae_w2016_seq_bear_breakdown_macro` | SHORT | 432 | 2664 / +0.064% | 3472 / +0.134% | 1668 / +0.375% |
| `pae_w4032_failed_breakout_short` | SHORT | 144 | 133 / +0.039% | 70 / +0.010% | 51 / +0.340% |

Interpretation: the fixed selector underused these stable/high-n sequence bearish features. Next backtests should start from this structurally constrained pool rather than continuing broad template sweeps.

## Duplicate feature issue

Breakout/breakdown volume variants are often identical or near-identical to their non-volume counterparts, so they inflate the candidate count without adding independent evidence.

Examples:

| Left | Right | Jaccard |
| --- | --- | ---: |
| `pae_w2016_break_below` | `pae_w2016_break_below_with_volume` | 1.000 |
| `pae_w4032_break_below` | `pae_w4032_break_below_with_volume` | 1.000 |
| `pae_w8640_break_below` | `pae_w8640_break_below_with_volume` | 1.000 |
| `pae_w4032_break_above` | `pae_w4032_break_above_with_volume` | 0.998 |
| `pae_w576_break_below` | `pae_w576_break_below_with_volume` | 0.997 |

## Decision

1. Stop broad optimization over all episode templates; it is producing unstable, regime-specific picks.
2. Deprioritize sparse 8640 mid-reject / failed-mid-reclaim short candidates despite attractive 2025 metrics.
3. Collapse or dedupe near-identical break/volume features before training or LLM label generation.
4. Continue with strict bar simulation on a small, predeclared stability-first pool:
   - `pae_w2016_seq_bear_failed_bounce` SHORT, h432
   - `pae_w2016_seq_bear_breakdown_macro` SHORT, h432
   - `pae_w4032_failed_breakout_short` SHORT, h144/h288
   - `pae_w2016_failed_mid_loss_long` LONG, h288
5. Treat this audit as a structural diagnostic only. Do not use eval-ranked ordering as a train-time selector.
