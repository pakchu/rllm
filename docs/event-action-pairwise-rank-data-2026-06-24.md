# Event-action pairwise rank data — 2026-06-24

## Why this exists

The Gemma4 E4B SFT64 candidate-value experiment failed on full 2026. The main structural weakness was the binary `TAKE/SKIP` target:

- train candidate rows: SKIP 107,399 / TAKE 9,481
- eval candidate rows: SKIP 11,113 / TAKE 827
- full 2026 train-calibrated threshold sweep was negative across q50/q75/q90/q95

The next target should make the LLM compare actions, not learn an imbalanced absolute gate. `training/event_action_pairwise_rank_data.py` converts existing value rows into within-signal pairwise ranking rows.

## New dataset shape

Each row asks the model to choose `A` or `B` between two candidate actions from the same signal.

Prompt contains only:

- past-only state,
- prompt-visible action book,
- candidate A action spec,
- candidate B action spec.

Prompt excludes:

- `rank_utility`, `net_return`, `mae`, `mfe`,
- old per-row `Candidate action:` line,
- future path labels.

Future realized strict utility is used only to decide `target` and to write audit metadata.

## Dry-run sizes

Built from the 2026-06-24 value datasets with:

```bash
python -m training.event_action_pairwise_rank_data \
  --min-utility-gap 0.002 \
  --max-pairs-per-signal 6
```

| split | source rows | signals with pairs | pair rows | target A | target B | mean utility gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-2026 train | 116,880 | 5,835 | 35,007 | 17,504 | 17,503 | 0.03531 |
| 2026 eval | 11,940 | 597 | 3,582 | 1,791 | 1,791 | 0.03248 |

This removes the SKIP-dominance problem while preserving enough samples for a small Gemma4 LoRA POC.

## Leakage boundary

- Pair candidates come from the same signal and the same past-only action book.
- Future utility is not embedded in prompt text.
- `chosen_action_audit` and `rejected_action_audit` are labels/audits only.
- This is not a backtest result. It is a training/evaluation data transformation.

## Next step

Train a small pairwise rank SFT/ORPO-style POC that outputs `A`/`B`, then aggregate pairwise preferences into an action winner per signal. Live-compatible selection should be:

1. generate candidate book from past-only features,
2. run pairwise tournament or score-via-comparisons,
3. trade only if the winner also passes a train-calibrated absolute quality/risk filter from a separate frozen calibration split.

Do not tune that final filter on the same eval segment.
