# Gemma focus-score threshold sweep on random eval1000 — 2026-06-27

## Purpose

The first score-policy converter used top-label agreement (`CLEAN_WIN_PATH` and `UTILITY_HIGH`) and produced only
2 trades on random 1000 eval rows. Inspection showed many `UTILITY_HIGH` scores were tied or near-tied, so this
run relaxes top-label agreement and sweeps causal score thresholds directly.

## Inputs

- Focus predictions: `results/episode_reward_focus_score_policy_random1000_2026-06-27/gemma_focus_eval1000_predictions.jsonl`
- Market data: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
- Split/sample: 2026 eval random 1000, seed 42
- Backtest: strict actual OHLC bar-by-bar, 1-bar entry delay, fees/slippage, strict MDD

## Sweep

Small grid:

- `clean_prob`: `0.02,0.05,0.1,0.2,0.3,0.4`
- `high_prob`: `0.0,0.2,0.333`
- margins disabled (`-999`) to avoid top-label/tie suppression
- min trades: 10

## Best observed rows

| clean_prob | high_prob | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.333 | 79 | 10.46% | 9.84% | 1.06 | 0.058% | 0.669 |
| 0.05 | 0.333 | 56 | 10.34% | 9.84% | 1.05 | 0.077% | 0.603 |
| 0.40 | 0.333 | 40 | 8.48% | 9.56% | 0.89 | 0.090% | 0.648 |

## Decision

Do not promote this as an alpha. Relaxing the top-label condition increases trade count and avoids the accidental
2-trade bottleneck, but the best random eval1000 result is statistically weak and far below the project target.
This also used eval for exploration, so it must not be used as final selection evidence.

## Next step

Run the same Gemma score extraction on a test sample or full test split, select thresholds there, and validate on
held-out eval. If the signal remains weak, the current focused Gemma SFT is not learning the reward surface well
enough for policy use and the next intervention should be target/prompt/model training quality, not more gate tuning.
