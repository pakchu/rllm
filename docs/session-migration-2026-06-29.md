# Session migration checkpoint — 2026-06-29

## Why this exists

The current OMX/Codex runtime state is tangled enough that the next session should not depend on `.omx/` state being intact. This document is the repo-native continuation artifact for a fresh OMX install or a new Codex session.

## Runtime/session identity

- Repo: `/home/pakchu/rllm`
- Branch: `feat/v3-text-analyzer-trader`
- Current tmux session observed: `omx-rllm-feat-v3-text-analyzer-trader-1782657911113-hooqti`
- Current pane: `/home/pakchu/rllm`, pane `%9`
- Old/native resume note: previous tangled session used native Codex id `019ca270-ed25-73f3-af31-f5984f8742a1`; prefer this document over relying on native resume.

## Git baseline

Latest commits before migration:

```text
7cb616a Export same-signal listwise preferences for RLLM
b5815a7 Reject linear event pairwise ranker as weak RLLM baseline
b55743d Reject current Gemma focus score policy after test-eval validation
cca581d Sweep Gemma focus thresholds without target echo
57168d4 Convert Gemma focus scores into policy rows
6d8238f Preserve row metadata in Gemma focus evaluations
7f9effa Reject cheap clause NB as downstream focus policy
e2ed877 Prune obsolete model checkpoints from the workspace
```

Known dirty runtime/tooling files to avoid staging unless intentionally cleaning OMX/Codex install:

- `.codex/**`
- `AGENTS.md`
- `omx_wiki/`

## Disk/checkpoint state

- `/` usage at checkpoint: about `228G / 1007G`, 24%.
- `checkpoints/`: about `1.5G`.
- Retained checkpoints:
  - `checkpoints/episode_reward_focus_v1_clauses_gemma4_sft64_2026-06-27`
  - `checkpoints/event_candidate_listwise_pref_gemma4_dpo_s64_2026-06-27`

Policy: keep WSL root under 300GB; prune obsolete checkpoints after each experiment branch.

## Current research conclusion

The profitable oracle surface exists, but causal approximations have not recovered it yet.

Rejected paths:

1. Focus-label oracle: huge upper bound, but future-label leakage by design.
2. Train-only clause NB: test weak, eval negative.
3. Gemma focus-score policy: test-selected threshold did not transfer to held-out eval.
4. Linear event pairwise ranker over wavefull ext+micro candidates: validation negative, eval statistically weak.

Current active direction:

- Same-signal listwise preference data for `LONG` / `SHORT` / `NO_TRADE`.
- This matches the RLLM intent better than independent categorical label prediction.
- The model should learn relative action preference at a timestamp, not absolute future buckets.

## Important generated data

Listwise preference export:

- Train: `data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_train_2026-06-27.jsonl`
- Eval: `data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_eval_2026-06-27.jsonl`
- Summary: `results/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_2026-06-27/summary.json`

Stats:

- Train: 12,507 preference pairs / 6,880 signals
- Eval: 3,431 preference pairs / 1,907 signals
- Prompt length mean: about 1,830 chars
- Chosen actions are mixed across NO_TRADE, LONG, SHORT.

DPO PoC checkpoint:

- `checkpoints/event_candidate_listwise_pref_gemma4_dpo_s64_2026-06-27`
- Gemma 4 E4B DPO, 1,024 gate-balanced samples, 64 steps
- Runtime: about 1h30m
- Train loss: about 0.692, so learning signal looked weak in training logs.

## Last interrupted task

A held-out preference logprob evaluator was added as `training/eval_preference_logprob.py`.

Purpose:

- Score `chosen` and `rejected` responses under base/adapted Gemma.
- Compute chosen-vs-rejected margin and accuracy.
- Use this before any generated-action backtest.

Compilation passed:

```bash
.venv/bin/python -m py_compile training/eval_preference_logprob.py
```

First eval attempt failed because the sandbox blocked a Hugging Face `custom_generate/generate.py` metadata request. After migration with normal network available, rerun:

```bash
MPLCONFIGDIR=/tmp/mpl HF_HUB_DISABLE_TELEMETRY=1 \
.venv/bin/python -m training.eval_preference_logprob \
  --eval-jsonl data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_eval_2026-06-27.jsonl \
  --output results/event_candidate_listwise_pref_gemma4_dpo_s64_eval_2026-06-28/eval128_adapter_report.json \
  --predictions-jsonl results/event_candidate_listwise_pref_gemma4_dpo_s64_eval_2026-06-28/eval128_adapter_predictions.jsonl \
  --model-name gemma4 \
  --adapter-dir checkpoints/event_candidate_listwise_pref_gemma4_dpo_s64_2026-06-27 \
  --max-samples 128 \
  --sample-mode gate_balanced \
  --seed 42 \
  --batch-size 8 \
  --max-length 2048
```

Recommended comparison after that:

```bash
MPLCONFIGDIR=/tmp/mpl HF_HUB_DISABLE_TELEMETRY=1 \
.venv/bin/python -m training.eval_preference_logprob \
  --eval-jsonl data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_eval_2026-06-27.jsonl \
  --output results/event_candidate_listwise_pref_gemma4_dpo_s64_eval_2026-06-28/eval128_base_report.json \
  --predictions-jsonl results/event_candidate_listwise_pref_gemma4_dpo_s64_eval_2026-06-28/eval128_base_predictions.jsonl \
  --model-name gemma4 \
  --max-samples 128 \
  --sample-mode gate_balanced \
  --seed 42 \
  --batch-size 8 \
  --max-length 2048
```

Promotion rule:

- Adapter must beat base on chosen-vs-rejected accuracy and margin.
- If adapter remains near 50% or margin near 0, reject this DPO PoC and do not backtest generated actions.

## Next technical steps after migration

1. Run adapter vs base preference logprob eval on held-out eval sample.
2. If adapter is not clearly better, remove or mark checkpoint as rejected.
3. If adapter is better, scale eval to 512 or 1,024 rows.
4. Only after preference eval passes, implement action scoring/generation for `LONG`/`SHORT`/`NO_TRADE` and strict backtest.
5. Avoid further gate optimization on weak signals; focus on learning target quality and same-signal relative choice.

## Safe cleanup/reinstall plan

Before reinstalling OMX:

1. Keep this repo branch and commits.
2. Do not delete `data/`, `results/`, or the two retained checkpoints unless intentionally pruning experiments.
3. Back up current runtime state if desired:

```bash
tar -czf /tmp/rllm-omx-runtime-backup-2026-06-29.tgz .omx .codex AGENTS.md omx_wiki
```

After reinstall:

1. Run `omx setup` from `/home/pakchu/rllm`.
2. Read this document first.
3. Re-run the held-out preference eval commands above.
4. Continue with normal commit-per-unit discipline.
