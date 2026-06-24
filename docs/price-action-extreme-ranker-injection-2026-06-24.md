# Price-action extreme feature injection into event ranker (2026-06-24)

## Purpose

After the univariate audit found weak but stable price-action signals, this pass injected those features into the event candidate ranker rows and reran the same rolling validation-gated OOS protocol.

Implementation: `training/augment_event_candidate_price_action_extremes.py`.

## Data generation

Full feature set:

```bash
.venv/bin/python -m training.augment_event_candidate_price_action_extremes \
  --input-jsonl data/event_action_compressor_ranker_all_2022_2026_2026-06-24.jsonl \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output-jsonl data/event_action_compressor_ranker_all_2022_2026_paext_2026-06-24.jsonl \
  --summary-output results/event_action_compressor_ranker_all_2022_2026_paext_summary_2026-06-24.json \
  --lookbacks 36,72,144,288,576 --tolerance 5min
```

- Rows matched: 128,820 / 128,820
- Added features: 65
- Join: backward-asof, 5m tolerance

Stable subset:

```bash
--include-features pa_ext_36_max_high_bar_spread_pct,pa_ext_72_max_high_bar_spread_pct,pa_ext_144_max_high_bar_spread_pct,pa_ext_288_max_high_bar_spread_pct,pa_ext_576_max_high_bar_spread_pct
```

- Rows matched: 128,820 / 128,820
- Added features: 5

## Rolling OOS results

Same walk-forward protocol as `docs/event-candidate-pairwise-walkforward-2026-06-24.md`.

### Full 65-feature price-action set

Report: `results/event_candidate_pairwise_walkforward_paext_2026-06-24/report.json`

- Trades: 99
- CAGR: -7.76%
- Strict MDD: 30.40%
- CAGR / strict MDD: -0.26
- Mean trade return: -0.213%

Fold summary:

| Fold | Status | Validation ratio | Test CAGR | Test strict MDD | Test ratio | Test trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | trade | 0.28 | -29.41% | 30.40% | -0.97 | 43 |
| 1 | abstain | -0.05 | n/a | n/a | n/a | 0 |
| 2 | abstain | -1.47 | n/a | n/a | n/a | 0 |
| 3 | trade | 0.09 | 4.75% | 11.77% | 0.40 | 21 |
| 4 | trade | 4.69 | -20.83% | 12.86% | -1.62 | 35 |
| 5 | abstain | -0.50 | n/a | n/a | n/a | 0 |

Stats-gated full set (`min_val_t_stat=1.0`, `max_val_p_value=0.25`, `max_val_power_gap=250`):

Report: `results/event_candidate_pairwise_walkforward_paext_statsgate_2026-06-24/report.json`

- Trades: 35
- CAGR: -3.94%
- Strict MDD: 12.86%
- CAGR / strict MDD: -0.31

The stricter gate reduced MDD below 15% but still selected a fold that lost money OOS.

### Stable 5-feature subset

Report: `results/event_candidate_pairwise_walkforward_paext_stable_2026-06-24/report.json`

- Trades: 165
- CAGR: -14.07%
- Strict MDD: 49.43%
- CAGR / strict MDD: -0.28
- Mean trade return: -0.239%

Fold positives existed (`fold0`, `fold3`), but fold1/fold4 losses dominated.

## Conclusion

Price-action extreme features improve the full feature set versus the prior baseline (`-15.15%` CAGR to `-7.76%`) but do not create a deployable strategy. Stable univariate spread does not survive the current pairwise linear ranker and validation gate.

Current bottleneck is now likely the ranker/adaptation layer rather than feature availability alone. Next useful direction: regime-conditioned or online-adaptive ranking that can flip/disable feature relations by regime, while keeping the same rolling leakage guard.
