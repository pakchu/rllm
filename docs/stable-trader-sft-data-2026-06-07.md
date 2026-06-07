# Stable trader SFT/RL data export (2026-06-07)

## Purpose
The current best deployable-shaped baseline is weak but fold-stable. This export turns that stable baseline into an out-of-fold LLM/RL trader dataset so the next Gemma4 trader stage can learn the stable action policy without using single-period oracle labels.

## Policy used for labels
```json
{
  "horizon_bars": 144,
  "target_pct": 1.8,
  "stop_pct": 1.5,
  "level": "teacher_only",
  "min_n": 20,
  "min_score": 0.0005,
  "score_mode": "mean",
  "side_gate": "free"
}
```

For every chronological fold, the label policy is fit only on rows at or before that fold's train end. Test-fold rows are then labeled out-of-fold.

## Output files
- `data/stable_trader_policy_h144_t1p8_s1p5_train.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_val.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_eval.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_all.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5.summary.json`

## Dataset size
| Split | Rows | Trades | Mean realized reward/trade |
| --- | ---: | ---: | ---: |
| Train | 1275 | 372 | +0.0061% |
| Val | 552 | 92 | +0.0116% |
| Eval | 535 | 96 | -0.0029% |
| All | 2362 | 560 | n/a |

Action distribution:
- `NO_TRADE`: 1802
- `LONG`: 341
- `SHORT`: 219

Reward buckets:
- `NO_TRADE`: 1802
- `HIGH_WIN`: 192
- `LARGE_LOSS`: 193
- `SMALL_WIN`: 75
- `SMALL_LOSS`: 80
- `FLAT`: 20

## Prompt / target shape
Prompt contains only compact past analyzer context and the stable policy anchor. Target is strict JSON:
```json
{"action":"LONG|SHORT|NO_TRADE","risk":"LOW|MEDIUM|HIGH"}
```

Rows also include policy score, bucket metadata, realized reward, reward bucket, and leakage guard fields for RL/reward-modeling.

## Interpretation
This is not a profitability breakthrough by itself. It is a cleaner LLM+RL training surface:
- avoids direct oracle best-action labels,
- avoids using eval data to choose policy labels,
- preserves the fold-stable baseline behavior,
- gives the trader stage reward metadata for later RL/preference tuning.

## Next step
Fine-tune or distill a Gemma4 trader on train rows, select checkpoint on val action fidelity plus reward-aware backtest, then evaluate once on eval. The model must beat the stable baseline's 4/4 fold positivity before replacing it.
