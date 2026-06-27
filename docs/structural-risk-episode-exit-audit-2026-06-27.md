# Structural-risk episode exit audit (2026-06-27)

## Purpose

The stable-feature strict backtest showed that overlapping horizon diagnostics do not translate into a deployable strategy. This audit tests whether a more price-action-native exit structure helps: use the signal bar's extreme as structural invalidation, pessimistic same-bar stop-before-TP ordering, optional R-multiple take profit, and max-hold timeout.

This is a fixed-template validator, not a selector. The tested specs came from the prior stability/strict decomposition; eval is reported after the fixed rules are instantiated.

## Main command

```bash
.venv/bin/python -m training.structural_episode_risk_backtest \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/structural_risk_failed_mid_loss_long_r15_h288_2026-06-27/report.json \
  --specs 'pae_w2016_failed_mid_loss_long@288' \
  --train-start 2024-01-01 --train-end '2024-12-31 23:59:59' \
  --test-start 2025-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --max-hold-bars 288 --take-profit-r 1.5 --min-risk-bps 5
```

## Results

### Previously weak-positive long candidate

| Spec | Split | CAGR | Strict MDD | Ratio | Trades | p-value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pae_w2016_failed_mid_loss_long@288` | Train | -5.18% | 6.13% | -0.85 | 137 | 0.0189 |
| `pae_w2016_failed_mid_loss_long@288` | Test | -5.28% | 5.45% | -0.97 | 101 | 0.00029 |
| `pae_w2016_failed_mid_loss_long@288` | Eval | -7.00% | 2.95% | -2.37 | 45 | 0.00009 |

Structural invalidation turned the only train/test-positive fixed-horizon template into a statistically negative strategy. The fixed-horizon gains came from longer drift after surviving early adverse movement, not from a clean structural entry.

### Sequence short candidates

| Spec | Split | CAGR | Strict MDD | Trades | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| `pae_w2016_seq_bear_failed_bounce@288` | Train | -36.41% | 36.61% | 986 | ~0 |
| `pae_w2016_seq_bear_failed_bounce@288` | Test | -42.70% | 42.65% | 1318 | ~0 |
| `pae_w2016_seq_bear_failed_bounce@288` | Eval | -49.90% | 24.71% | 555 | ~0 |
| `pae_w2016_seq_bear_breakdown_macro@288` | Train | -37.83% | 38.12% | 1064 | ~0 |
| `pae_w2016_seq_bear_breakdown_macro@288` | Test | -44.91% | 44.84% | 1411 | ~0 |
| `pae_w2016_seq_bear_breakdown_macro@288` | Eval | -54.48% | 27.72% | 601 | ~0 |

These have plenty of trades and are consistently negative. This is not a sample-size problem.

### Direction inversion check

The same sequence triggers forced to LONG also failed:

| Trigger | Override | Train CAGR | Test CAGR | Eval CAGR |
| --- | --- | ---: | ---: | ---: |
| `pae_w2016_seq_bear_failed_bounce@288` | LONG | -44.08% | -46.87% | -50.68% |
| `pae_w2016_seq_bear_breakdown_macro@288` | LONG | -48.87% | -52.45% | -57.02% |

Interpretation: the failure is not just wrong direction. The sequence features are too dense/choppy for structural stop/TP execution and are eaten by adverse excursions plus costs in both directions.

## Decision

1. Do not continue tuning sequence macro episode templates as direct trade triggers.
2. Do not use signal-bar structural stop/TP as a universal fix for current episode features; it exposes that the entries are not clean enough.
3. The next useful feature work should target entry quality and state compression, not more gates:
   - fewer, higher-conviction events;
   - distance-to-range/extreme and compression/breakout location features;
   - explicit adverse-excursion labels for whether an entry can survive realistic stop placement;
   - LLM text labels should describe setup quality and invalidation distance, not only event names.
