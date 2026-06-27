# Episode Reward Focus v1 Clause Dataset (2026-06-27)

## Purpose
The six-component reward target was too broad. Small eval probes only showed useful signal on:
- `utility_bucket`
- `path_shape`

This dataset narrows the LLM target to those two fields so SFT/evaluation can test the useful part directly.

## Added
- `training/export_episode_reward_focus_sft_data.py`
  - converts reward-component rows into focused targets only.
- `training/eval_reward_focus_logprob.py`
  - evaluates only `path_shape` and `utility_bucket` using teacher-forced option logprob.
- `training/train_text_sft.py`
  - target summary and balanced sampling now handle focus targets.

## Dataset
Output:

`data/episode_reward_focus_v1_clauses_2026-06-27/`

| split | rows | prompt mean chars | target mean chars |
| --- | ---: | ---: | ---: |
| train | 76,956 | 1,012.1 | 62.0 |
| test | 41,274 | 1,011.5 | 61.4 |
| eval | 8,304 | 1,011.8 | 61.5 |

The focused target is about 65% shorter than the six-component target (`~62` vs `~178` chars).

## Eval split target distribution
- utility: LOW 2,383 / MID 2,962 / HIGH 2,959
- path shape: HIGH_ADVERSE 1,920 / MIXED 3,043 / CLEAN_WIN 2,592 / LOW_EDGE 702 / FAILED_FOLLOW_THROUGH 47

## Validation
Commands run:

```bash
.venv/bin/python -m py_compile training/export_episode_reward_focus_sft_data.py training/eval_reward_focus_logprob.py training/train_text_sft.py
.venv/bin/python -m training.export_episode_reward_focus_sft_data ...
.venv/bin/python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/episode_reward_focus_v1_clauses_2026-06-27/plain/train.jsonl \
  --output-dir checkpoints/episode_reward_focus_v1_clauses_gemma4_dryrun_2026-06-27 \
  --max-samples 512 --sample-mode balanced --max-seq-length 1152 --dry-run
```

Dry-run result:
- rows: 512
- target counts include all utility buckets and all path-shape classes
- prompt mean chars: 1,013.85
- target mean chars: 63.29

## Next gate
Run a bounded Gemma SFT on this focused target and compare against majority baseline on eval. Promotion requires both fields to beat majority on a sufficiently large eval sample, not just eval50.
