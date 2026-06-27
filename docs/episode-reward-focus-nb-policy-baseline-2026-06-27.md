# Episode reward focus NB policy baseline — 2026-06-27

## Purpose

The previous oracle diagnostic showed that the focused reward labels (`path_shape=CLEAN_WIN_PATH` and
`utility_bucket=UTILITY_HIGH`) have a very large upper-bound when they are read directly from future-derived
targets. That result is useful only as a target-surface audit; it is not deployable.

This experiment tests the first causal approximation: fit a train-only bag-of-clause Multinomial Naive Bayes
model on the focused clause prompts, predict only the two focused labels, then trade only when both predicted
labels are favorable.

## Setup

- Dataset: `data/episode_reward_focus_v1_clauses_2026-06-27/plain/`
- Train: 2020-2023, 76,956 rows
- Test: 2024-2025, 41,274 rows
- Eval: 2026-01-01..2026-05-30, 8,304 rows
- Model: train-only clause-token NB, `min_token_count=3`, `alpha=1.0`
- Policy rule: trade candidate side only when predicted `CLEAN_WIN_PATH` and predicted `UTILITY_HIGH`
- Backtest: strict single-policy simulator, actual OHLC bar-by-bar returns, 1-bar entry delay, fees/slippage,
  strict MDD including adverse excursion

## Outputs

- Summary: `results/episode_reward_focus_nb_policy_v1_2026-06-27/focus_nb_policy_summary.json`
- Test predictions: `results/episode_reward_focus_nb_policy_v1_2026-06-27/focus_nb_test_policy_predictions.jsonl`
- Eval predictions: `results/episode_reward_focus_nb_policy_v1_2026-06-27/focus_nb_eval_policy_predictions.jsonl`
- Test strict backtest: `results/episode_reward_focus_nb_policy_v1_2026-06-27/focus_nb_test_strict_backtest.json`
- Eval strict backtest: `results/episode_reward_focus_nb_policy_v1_2026-06-27/focus_nb_eval_strict_backtest.json`

## Label prediction quality

| Split | path_shape acc | utility_bucket acc | LONG | SHORT | NO_TRADE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 42.45% | 41.09% | 1,125 | 609 | 75,222 |
| Test | 40.75% | 40.98% | 654 | 321 | 40,299 |
| Eval | 37.02% | 36.43% | 114 | 159 | 8,031 |

The NB model is slightly informative, but the edge is weak and degrades on 2026 eval.

## Strict backtest result

| Split | Period | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 2024-2025 | 142 | 10.26% | 10.44% | 0.98 | 0.144% | 0.134 |
| Eval | 2026-01-01..2026-05-30 | 32 | -10.17% | 8.66% | -1.17 | -0.129% | 0.535 |

The test result is weak and statistically underpowered (`n_required_for_80pct_power_alpha5pct=497`, gap 355).
The eval result is negative and far below the target.

## Decision

Reject the cheap clause NB policy as a deployable or promotion-worthy path. It is useful as a causal floor:
if a very simple symbolic/text model cannot preserve the oracle edge, the next step must either extract Gemma
focus scores as richer causal features or train a stronger calibrated focus model. The oracle target surface is
still valuable, but the deployable policy needs a causal approximation that survives 2026 eval.

## Leakage note

This run does not echo future targets into the policy. The NB model is fit on train rows only, then applied to
test/eval prompts. The strict backtest consumes policy predictions and actual OHLC bars with 1-bar delayed entry.
The future-derived `focus_target` remains in the prediction rows only for audit/metrics, not for action selection.
