# Episode Survival Pairwise v3 Clause Prompt Export (2026-06-27)

## Purpose
The Gemma4-E4B v2 numeric prompt PoC failed below the logistic baseline. The next iteration changes the representation instead of merely scaling the same SFT: numeric-heavy JSON dumps are converted into compact causal price-action/regime clauses so the LLM sees a pattern-description task rather than raw scalar regression.

## Code change
`training/export_episode_survival_pairwise_data.py` now supports:

```bash
--prompt-style json      # existing v2-style JSON prompt, default
--prompt-style clauses   # new v3-style clause prompt
```

The label construction is unchanged:
- same source rows from natural survival SFT v1
- same timestamp pair construction
- same minimum utility gap: 0.35%
- same max pairs per signal: 3
- same future utility target used only for offline labels

Only the prompt representation changes.

## Generated dataset
Output directory:

`data/episode_survival_pairwise_v3_clauses_2026-06-27/`

Split sizes match pairwise v2:

| split | rows | A labels | B labels | mean utility gap |
| --- | ---: | ---: | ---: | ---: |
| train | 31,010 | 15,489 | 15,521 | 1.3983% |
| test | 15,447 | 7,754 | 7,693 | 1.1224% |
| eval | 3,257 | 1,655 | 1,602 | 1.1410% |

Prompt length comparison:
- v2 numeric JSON prompt mean: ~2,326 chars
- v3 clause prompt mean: ~1,326 chars
- reduction: ~43%

## Clause content
The clause prompt encodes:
- trend stack and alignment
- volatility expansion/compression across horizons
- 12/48/144/576 lookback return direction, price-zone, and drawdown direction
- SMA side and regime age buckets
- candidate side/event/episode/horizon
- setup quality buckets: close/risk/range/wick/body
- macro buckets: kimchi, DXY, USDKRW, kimchi change
- competition clauses: same side/type, horizon relation, A-minus-B quality deltas

The clauses intentionally avoid future path metrics. Future utility remains only in `chosen_audit`, `rejected_audit`, and `target`, which are training/evaluation labels, not prompt inputs.

## Validation
Commands run:

```bash
.venv/bin/python -m py_compile training/export_episode_survival_pairwise_data.py
.venv/bin/python -m training.export_episode_survival_pairwise_data ... --prompt-style clauses
.venv/bin/python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/episode_survival_pairwise_v3_clauses_2026-06-27/plain/train.jsonl \
  --output-dir checkpoints/episode_survival_pairwise_v3_clauses_gemma4_dryrun_2026-06-27 \
  --max-samples 256 --sample-mode balanced --max-seq-length 1536 --dry-run
```

Dry-run result:
- rows: 256
- target counts: A=128, B=128
- prompt chars: min 1273, mean 1327.69, max 1387
- target chars: 72

## Decision
v3 clauses are ready for a small Gemma4 SFT PoC. Promotion gate remains strict: the adapter must beat the logistic pairwise v2 eval baseline of 52.35% on a bounded, no-leak eval probe before any strict portfolio backtest or RL stage.
