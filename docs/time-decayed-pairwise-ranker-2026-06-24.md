# Time-decayed pairwise ranker (2026-06-24)

## Purpose

The short-adaptation PA-ext ranker improved but still suffered from stale feature/action relations inside each fit window. This pass changes the ranker itself: pairwise winner-vs-loser samples can be exponentially weighted by recency inside the fit window.

Implementation:

- `training/event_candidate_pairwise_ranker.py`
  - Adds `pair_half_life_days`.
  - Recent pair samples receive higher logistic-loss weight.
  - `0` preserves the previous unweighted behavior.
- `training/event_candidate_pairwise_walkforward.py`
  - Exposes `--pair-half-life-days` to rolling runs.

## Protocol

Same PA-ext 6M fit / 3M validation / 3M test stats-gated protocol:

- Input: `data/event_action_compressor_ranker_all_2022_2026_paext_2026-06-24.jsonl`
- Validation gate:
  - `min_val_cagr_pct=10`
  - `min_val_ratio=1.0`
  - `max_val_strict_mdd_pct=20`
  - `min_val_t_stat=0.8`
  - `max_val_p_value=0.45`
  - `max_val_power_gap=500`
- No test data used for fitting or policy selection.

## Half-life comparison

| Pair half-life | CAGR | Strict MDD | CAGR/MDD | Trades | Traded folds | p approx | Mean trade |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 / no decay | 5.39% | 17.36% | 0.31 | 126 | 6 | 0.410 | +0.183% |
| 30d | -8.39% | 48.97% | -0.17 | 159 | 8 | 0.424 | -0.167% |
| 45d | 13.26% | 14.10% | 0.94 | 119 | 6 | 0.087 | +0.418% |
| 60d | 7.97% | 22.35% | 0.36 | 133 | 6 | 0.294 | +0.247% |
| 90d | 7.89% | 20.90% | 0.38 | 121 | 6 | 0.280 | +0.265% |

Best report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_statsgate_2026-06-24/report.json`.

Best fold profile:

- 2022Q4: `CAGR 87.7 / MDD 12.5 / ratio 6.99`
- 2023Q1: `CAGR -6.4 / MDD 10.0 / ratio -0.64`
- 2024Q4: `CAGR 120.4 / MDD 11.1 / ratio 10.84`
- 2025Q1: `CAGR 38.5 / MDD 6.5 / ratio 5.95`
- 2025Q2: `CAGR 52.9 / MDD 4.7 / ratio 11.31`
- 2025Q3: `CAGR -35.5 / MDD 14.1 / ratio -2.52`

## Online overlay check

The previously useful risk overlay was too conservative after decay45:

- Decay45 only: `CAGR 13.26 / MDD 14.10 / ratio 0.94`, 119 trades
- Decay45 + overlay: `CAGR 3.22 / MDD 12.96 / ratio 0.25`, 17 trades

Overlay should not be applied unchanged to the decay45 model.

## Conclusion

Time-decayed pairwise learning is the best ranker improvement so far. It reaches strict MDD < 15 and improves p-value to ~0.087, but it still misses the target CAGR/MDD >= 3. The remaining major failure is 2025Q3; future work should either detect/avoid that decay regime or add a complementary short-capable alpha that works in that regime.
