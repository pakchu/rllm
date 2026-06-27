# Setup-quality filter audit (2026-06-27)

## Purpose

After strict execution invalidated most episode templates, this audit checks whether the issue is the event family itself or the lack of setup-quality conditioning. It buckets causal signal-bar attributes using train-only quantiles, applies the bucket unchanged to test/eval, and evaluates strict fixed-horizon execution.

This is still diagnostic. It uses only single-feature buckets to reduce overfit pressure; no multi-gate optimizer is introduced.

## Command

```bash
.venv/bin/python -m training.audit_setup_quality_filters \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/setup_quality_filters_core_specs_2026-06-27/report.json \
  --specs 'pae_w2016_failed_mid_loss_long@288,pae_w4032_failed_breakout_short@288,pae_w2016_seq_bear_failed_bounce@288,pae_w2016_seq_bear_breakdown_macro@288' \
  --train-start 2024-01-01 --train-end '2024-12-31 23:59:59' \
  --test-start 2025-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --min-train-trades 20 \
  --top-k 40
```

## Features bucketed

Thresholds are fit on train triggers only.

- `risk_bps`: next-open entry distance to the signal-bar invalidation extreme.
- `range_bps`: signal candle high-low range normalized by close.
- `body_frac`: absolute body / high-low range.
- `favorable_wick_frac`: lower wick for long, upper wick for short.
- `close_quality`: close position in the candle, side-aligned.

## Main findings

### Promising but weak long setup-quality buckets

All of the useful rows are the same event: `pae_w2016_failed_mid_loss_long@288` LONG. The setup quality filters reduce trades and improve strict MDD.

| Filter | Train CAGR/MDD/Trades | Test CAGR/MDD/Trades | Eval CAGR/MDD/Trades | Test p-value |
| --- | --- | --- | --- | ---: |
| `favorable_wick_frac=low` | 3.15 / 6.57 / 30 | 6.74 / 3.57 / 29 | 34.48 / 1.69 / 12 | 0.187 |
| `range_bps=low` | 3.87 / 6.17 / 23 | 5.32 / 5.01 / 25 | 9.11 / 2.46 / 10 | 0.323 |
| `body_frac=mid` | 1.04 / 8.21 / 29 | 3.29 / 4.32 / 28 | 27.20 / 2.46 / 15 | 0.515 |
| `risk_bps=low` | 4.81 / 5.50 / 26 | 3.17 / 5.13 / 26 | 24.69 / 2.46 / 13 | 0.537 |

Interpretation: setup quality matters. Low-risk / small-range versions of the failed-mid-loss long event are more stable than the raw event. However, trade counts are low and p-values are not strong enough for the target.

### Short sequence features remain rejected

Sequence bearish events can look better in test after bucket filtering, but train remains negative and eval often fails. Examples:

| Filter | Train | Test | Eval | Decision |
| --- | --- | --- | --- | --- |
| `seq_bear_failed_bounce@288`, `range_bps=high` | -12.59 / 17.68 / 59 | 5.58 / 9.62 / 54 | -8.99 / 15.04 / 29 | Reject: train/eval fail |
| `seq_bear_breakdown_macro@288`, `range_bps=low` | -5.50 / 12.59 / 48 | 4.65 / 9.01 / 50 | -21.05 / 12.18 / 28 | Reject: train/eval fail |

## Decision

1. Keep `failed_mid_loss_long@288` as a weak long setup-quality family worth further study.
2. Do not promote it to a strategy yet: current best train/test-positive bucket has only 23-30 train trades and test p-values remain weak.
3. Retire current sequence bearish templates as direct entries; bucket conditioning does not fix their train/eval instability.
4. Next feature work should expand setup-quality descriptors around the long failed-mid-loss event and search for independent weak alphas that can combine without overlapping the same 10-30 trades.
5. For LLM/RLLM, the text target should include setup-quality attributes (`risk_bps`, `range_bps`, wick/body/close quality) and not just semantic event names.
