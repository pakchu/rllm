# Price-action extreme-bar feature audit (2026-06-24)

## Purpose

The previous LLM/ranker lanes failed because validation edges did not survive the next OOS fold. This pass audits a new price-action feature family before integrating it into candidate ranking.

Requested feature idea: for several lookback periods, find the bar that made the window maximum/minimum and use the paired opposite extreme from that same bar:

- highest-high bar → that bar's low (`low_of_max_high`)
- lowest-low bar → that bar's high (`high_of_min_low`)

Implementation: `training/price_action_extreme_feature_audit.py`.

## Leakage contract

At row `t`, every feature uses only candles `<= t`. Forward returns are used only as diagnostic labels with `entry_delay_bars=1`.

## Run

```bash
.venv/bin/python -m training.price_action_extreme_feature_audit \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/price_action_extreme_feature_audit_2026-06-24.json \
  --lookbacks 36,72,144,288,576,2016 \
  --horizons 36,72,144,288 \
  --entry-delay-bars 1 --quantile 0.2 --min-samples 500
```

Input rows: 674,785 5m bars, 2019-12-31 15:00:00 to 2026-05-31 15:00:00.
Feature count: 78.

## Findings

The raw `low_of_max_high` / `high_of_min_low` distance features are not reliably stable by themselves. Some years flip sign, especially 2022/2023 vs 2024/2025/2026.

The more promising derivative is the range of the highest-high bar:

`pa_ext_{lookback}_max_high_bar_spread_pct = (max_high - low_of_max_high) / current_close`

This measures the amplitude of the candle that created the window high. It showed repeated positive top-vs-bottom forward-return spreads across 2023, 2024, 2025, and 2026.

Top stable examples using 2023/2024/2025/eval2026 consistency:

| Feature | Horizon bars | 2023 spread | 2024 spread | 2025 spread | 2026 spread | Notes |
|---|---:|---:|---:|---:|---:|---|
| `pa_ext_72_max_high_bar_spread_pct` | 288 | +0.490% | +0.501% | +0.123% | +0.399% | strongest stable spread; IC weak/mixed in 2023 |
| `pa_ext_288_max_high_bar_spread_pct` | 288 | +0.361% | +0.347% | +0.179% | +0.118% | cleaner IC: +0.048/+0.080/+0.048/+0.052 |
| `pa_ext_36_max_high_bar_spread_pct` | 288 | +0.431% | +0.440% | +0.100% | +0.312% | stable spread; short lookback |
| `pa_ext_36_max_high_bar_spread_pct` | 144 | +0.256% | +0.268% | +0.089% | +0.315% | shorter horizon also stable |
| `pa_ext_72_max_high_bar_spread_pct` | 144 | +0.236% | +0.333% | +0.089% | +0.416% | strong 2026 continuation |
| `pa_ext_72_max_high_bar_spread_pct` | 72 | +0.099% | +0.131% | +0.082% | +0.254% | shorter horizon, still stable |

Counterexamples:

- `range_pos` and distance-to-extreme features often had same IC sign but quantile-spread sign flips, so they are risky as direct policy signals.
- Several high-scoring rows in the full report are unstable because the scoring still rewards IC/absolute movement; they should not be promoted without same-sign yearly spread.

## Conclusion

This is a usable weak-alpha feature family candidate, not a strategy. The highest-high bar spread appears more stable than the originally requested raw paired-price distances. It should be injected into event candidate `feature_snapshot` and tested inside the existing rolling validation-gated walk-forward harness.

Next acceptance condition: adding these features must improve rolling OOS aggregate, not just static validation or one favorable fold.
