# Episode Reward Components v1 Clause SFT Export (2026-06-27)

## Purpose
Direct `TRADE/NO_TRADE` and direct pairwise `A/B` Gemma SFT both failed out-of-sample. This dataset changes the LLM task: instead of asking Gemma to make a final trade decision, it predicts decomposed future path components from causal price-action/setup/macro clauses.

The intended RLLM structure is now:

1. LLM predicts structured reward/path descriptors.
2. A calibrated downstream reward/policy model consumes those descriptors.
3. Strict backtest/RL promotion happens only after component predictions show out-of-sample signal.

## New exporter
Added:

`training/export_episode_reward_component_sft_data.py`

Inputs are the no-leak natural survival SFT rows:

`data/episode_survival_sft_v1_natural_2026-06-27/`

The exporter adds causal market-regime clauses using the already-tested history construction from pairwise exports and writes component targets from `target_audit`.

## Target schema
Each target JSON contains:

```json
{
  "net_bucket": "NET_WEAK|NET_MID|NET_STRONG",
  "mae_bucket": "ADVERSE_LOW|ADVERSE_MID|ADVERSE_HIGH",
  "mfe_bucket": "FAVORABLE_LOW|FAVORABLE_MID|FAVORABLE_HIGH",
  "mfe_to_mae_bucket": "PAYOFF_POOR|PAYOFF_MID|PAYOFF_GOOD",
  "utility_bucket": "UTILITY_LOW|UTILITY_MID|UTILITY_HIGH",
  "path_shape": "CLEAN_WIN_PATH|HIGH_ADVERSE_PATH|FAILED_FOLLOW_THROUGH|LOW_EDGE_PATH|MIXED_PATH"
}
```

Component thresholds are fit on train only:

| component | low/mid cut | mid/high cut |
| --- | ---: | ---: |
| net_pct | -0.3213 | 0.2428 |
| mae_pct | 0.6993 | 2.0259 |
| mfe_pct | 0.7001 | 1.8708 |
| utility_pct | -0.6716 | 0.0835 |
| mfe_to_mae | 0.4372 | 2.1414 |

## Generated dataset
Output directory:

`data/episode_reward_components_v1_clauses_2026-06-27/`

| split | rows | prompt mean chars |
| --- | ---: | ---: |
| train | 76,956 | 1,012.1 |
| test | 41,274 | 1,011.5 |
| eval | 8,304 | 1,011.8 |

Train component buckets are approximately balanced by construction because thresholds are train terciles. Eval/test use train cuts, so their distributions are natural out-of-sample distributions.

## Leakage guard
Prompt contains only:
- candidate metadata
- setup quality buckets/values from the signal bar
- macro z-score/change values available at the signal bar
- causal market history ending at `signal_pos`

Future path values are used only in `target`, `target_audit`, and train-fitted bucket cuts.

## Validation
Commands run:

```bash
.venv/bin/python -m py_compile training/export_episode_reward_component_sft_data.py
.venv/bin/python -m training.export_episode_reward_component_sft_data ...
.venv/bin/python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/episode_reward_components_v1_clauses_2026-06-27/plain/train.jsonl \
  --output-dir checkpoints/episode_reward_components_v1_clauses_gemma4_dryrun_2026-06-27 \
  --max-samples 256 --max-seq-length 1280 --dry-run
```

Dry-run result:
- rows: 256
- task: `episode_reward_component_sft`
- prompt chars: min 993, mean 1014.45, max 1043
- target chars: min 170, mean 178.12, max 183

## Decision
This dataset is ready for the next bounded Gemma PoC. Promotion gate is not portfolio PnL yet. First gate is component predictability on eval; if component prediction cannot beat simple baselines, downstream trading logic should not be built on it.
