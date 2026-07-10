# REX TTE gate optimization (2026-07-11)

## Purpose

Optimize REX rule-gate candidates under a leakage-safe train/test/eval (TTE) protocol, with full-window CAGR that counts idle/no-trade time.

## Protocol

- Train rows: `data/rex_pair_reclaim075_deep085_h144_ranker_train_2021_2024.jsonl`
- Test rows: `data/rex_pair_reclaim075_deep085_h144_ranker_test_2025.jsonl`
- Eval rows: `data/rex_pair_reclaim075_deep085_h144_ranker_eval_2026h1.jsonl`
- Market bars: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
- Train feature quantiles create primitive gates.
- Gate sets are ranked by train+test only.
- Eval is reported after ranking and is not used for threshold/selection.
- CAGR windows are configured as full split periods:
  - train: `2021-01-01` through `2024-12-31 23:59:59`
  - test: `2025-01-01` through `2025-12-31 23:59:59`
  - eval: `2026-01-01` through `2026-06-01 00:00:00`
- Strict MDD is the strict bar-by-bar adverse-excursion simulator used by `training.economic_action_backtest`.

## Code change

`training/sweep_conjunctive_event_gates.py` now supports:

- explicit `--train-start/end`, `--test-start/end`, `--eval-start/end` so CAGR uses full split windows, not only selected trade span;
- `--side-filter LONG|SHORT` and `--family-filter` for side/family-specific TTE sweeps;
- `--min-train-trades` / `--min-test-trades` for statistically safer candidate screening.

`training/economic_action_backtest.py` now carries a local dependency-light action parser so pure backtests no longer import VLM/model modules and no longer require torch just to run a JSON action backtest.

## Main all-side sweep

Artifact: `results/rex_pair_tte_gate_sweep_fullwindow_2026-07-11.json`

Command:

```bash
python -m training.sweep_conjunctive_event_gates \
  --train-jsonl data/rex_pair_reclaim075_deep085_h144_ranker_train_2021_2024.jsonl \
  --test-jsonl data/rex_pair_reclaim075_deep085_h144_ranker_test_2025.jsonl \
  --eval-jsonl data/rex_pair_reclaim075_deep085_h144_ranker_eval_2026h1.jsonl \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/rex_pair_tte_gate_sweep_fullwindow_2026-07-11.json \
  --max-primitives 20 --max-width 2 --leverage-grid 0.5,1.0,1.5 \
  --train-start '2021-01-01' --train-end '2024-12-31 23:59:59' \
  --test-start '2025-01-01' --test-end '2025-12-31 23:59:59' \
  --eval-start '2026-01-01' --eval-end '2026-06-01 00:00:00'
```

### Top by train+test selection score

| Rank | Gates | Train abs/CAGR/MDD/R/N | Test abs/CAGR/MDD/R/N | Eval abs/CAGR/MDD/R/N | Eval p |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `range_vol >= 0.0218777` & `volume_zscore >= -0.930277` | 77.91 / 15.49 / 14.29 / 1.08 / 489 | 17.74 / 17.75 / 2.33 / 7.62 / 44 | 2.49 / 6.13 / 5.62 / 1.09 / 32 | 0.607 |
| 2 | `rex_144_range_width_pct >= 0.0218772` & `volume_zscore >= -0.930277` | 76.70 / 15.29 / 14.29 / 1.07 / 488 | 17.74 / 17.75 / 2.33 / 7.62 / 44 | 2.49 / 6.13 / 5.62 / 1.09 / 32 | 0.607 |
| 3 | `range_vol >= 0.0218777` & `dxy_momentum >= -0.000293058` | 49.76 / 10.62 / 15.98 / 0.66 / 387 | 15.53 / 15.54 / 2.01 / 7.71 / 39 | 5.59 / 14.07 / 4.32 / 3.26 / 27 | 0.167 |

Interpretation: the best train+test-ranked gate does not generalize strongly to eval. The third ranked candidate clears eval CAGR/MDD > 3, but train ratio is weak and train MDD is slightly over 15, so it is a watchlist candidate rather than a deployable final gate.

### Best eval-ratio rows inside the reported top-50 (diagnostic only)

These are not selection-valid choices unless a future train/test-only objective would have picked them without seeing eval.

| Gates | Train abs/CAGR/MDD/R/N | Test abs/CAGR/MDD/R/N | Eval abs/CAGR/MDD/R/N | Eval p |
| --- | ---: | ---: | ---: | ---: |
| `window_drawdown >= 0.0109957` | 48.07 / 10.31 / 16.79 / 0.61 / 424 | 10.68 / 10.69 / 2.14 / 4.99 / 30 | 3.70 / 9.18 / 1.81 / 5.06 / 17 | 0.027 |
| `range_vol >= 0.0218777` & `window_drawdown >= 0.0109957` | 53.03 / 11.22 / 13.66 / 0.82 / 374 | 10.51 / 10.51 / 2.14 / 4.91 / 24 | 3.61 / 8.96 / 1.81 / 4.94 / 15 | 0.029 |
| `range_vol >= 0.0218777` & `dxy_momentum >= -0.000293058` | 49.76 / 10.62 / 15.98 / 0.66 / 387 | 15.53 / 15.54 / 2.01 / 7.71 / 39 | 5.59 / 14.07 / 4.32 / 3.26 / 27 | 0.167 |

## SHORT-only sweep

Artifact: `results/rex_pair_short_tte_gate_sweep_fullwindow_2026-07-11.json`

Command uses `--side-filter SHORT --min-train-trades 80 --min-test-trades 10`.

### Top by train+test selection score

| Rank | Gates | Train abs/CAGR/MDD/R/N | Test abs/CAGR/MDD/R/N | Eval abs/CAGR/MDD/R/N | Eval p |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `rex_candidate_strength <= 0.296389` & `dxy_momentum >= -0.000512429` | 16.95 / 3.99 / 11.22 / 0.36 / 148 | 18.14 / 18.16 / 2.43 / 7.47 / 31 | 1.19 / 2.91 / 4.80 / 0.61 / 18 | 0.755 |
| 2 | `rex_threshold_ratio <= 1.41523` & `dxy_momentum >= -0.000512429` | 16.95 / 3.99 / 11.22 / 0.36 / 148 | 18.14 / 18.16 / 2.43 / 7.47 / 31 | 1.19 / 2.91 / 4.80 / 0.61 / 18 | 0.755 |
| 3 | `rex_threshold_excess <= 0.0871084` & `dxy_momentum >= -0.000512429` | 16.95 / 3.99 / 11.22 / 0.36 / 148 | 18.14 / 18.16 / 2.43 / 7.47 / 31 | 1.19 / 2.91 / 4.80 / 0.61 / 18 | 0.755 |

### Best eval-ratio rows inside reported top-50 (diagnostic only)

| Gates | Train abs/CAGR/MDD/R/N | Test abs/CAGR/MDD/R/N | Eval abs/CAGR/MDD/R/N | Eval p |
| --- | ---: | ---: | ---: | ---: |
| `rex_2016_cur_to_min_pct >= 0.0503604` & `dxy_momentum >= -0.000512429` | 8.58 / 2.08 / 10.71 / 0.19 / 180 | 13.28 / 13.29 / 2.01 / 6.60 / 32 | 5.20 / 13.04 / 3.08 / 4.23 / 20 | 0.168 |
| `rex_2016_cur_to_min_pct >= 0.0687422` & `dxy_momentum >= -0.000512429` | 4.27 / 1.05 / 10.50 / 0.10 / 156 | 12.75 / 12.76 / 2.01 / 6.34 / 29 | 4.54 / 11.34 / 3.34 / 3.40 / 14 | 0.092 |

## Decision

This TTE optimization improves the measurement pipeline and produces some usable clues, but it does **not** produce a fully promotion-safe rule yet:

- The train+test-selected top all-side gate is robust in 2025 but weak in 2026H1.
- Some diagnostic rows clear eval CAGR/MDD > 3, but their train ratio is too low or trade count is thin.
- SHORT-only rules show very strong 2025 test and weaker 2026H1 eval; this supports the idea that short REX is regime-dependent rather than universally stable.

Next step should be to change the train+test selection objective, not to hand-pick eval winners: penalize weak train ratio more strongly, prefer cross-split lower-bound score, and require either train MDD <= 15 or train ratio >= 1 before eval is opened.
