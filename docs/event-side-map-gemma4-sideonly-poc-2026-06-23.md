# Event side-map Gemma4 side-only SFT POC (2026-06-23)

## Purpose

The first event side-map Gemma4 POC trained a full JSON target:

```json
{"side_map":"normal","confidence":"...","reason_code":"..."}
```

Evaluation, however, scored only `side_map` candidates. This POC removes that train/eval surface mismatch by projecting every target to a single-key JSON:

```json
{"side_map":"normal"}
```

The test is intentionally small: verify whether a cleaner side-only target can beat the already-known non-LLM baselines before spending more compute.

## Data

Source event labels:

- `data/event_side_map_reliability_h288_start2022_2026-06-23.jsonl`
- 2,792 event proposals from 2022-start rolling h288 setup

Projected side-only data:

- `data/event_side_map_reliability_h288_start2022_sideonly_2026-06-23.jsonl`
- `results/event_side_map_reliability_h288_start2022_sideonly_summary_2026-06-23.json`

Projected target counts:

| Target | Count |
| --- | ---: |
| normal | 1,328 |
| inverse | 1,292 |
| unreliable | 172 |

Eval split:

- `data/event_side_map_reliability_h288_start2022_sideonly_eval2026_2026-06-23.jsonl`
- 201 event rows from 2026-01-02 through 2026-05-29

## Code added

- `training/project_json_key_target.py`
  - copies source rows unchanged except for `target`;
  - replaces target JSON with a single requested key;
  - marks `leakage_guard.target_projected_to_single_key`.
- `tests/test_project_json_key_target.py`
  - covers successful projection and missing-key failure.

## Training

Checkpoint:

- `checkpoints/event_side_map_sideonly_gemma4_e4b_sft16_2026-06-23`

Config summary:

| Field | Value |
| --- | --- |
| Model alias | `gemma4-e4b-it` (`google/gemma-4-E4B-it`) |
| Samples | 768 balanced |
| Max sequence length | 1536 |
| LoRA | r=8, alpha=16, dropout=0.05 |
| Steps | 16 |
| Batch / grad accumulation | 1 / 8 |
| 4-bit | false |

Training evidence:

- runtime: 109.3s
- train loss: 3.889 average
- step loss decreased from 5.136 to 2.745
- target counts in sampled training set:
  - normal: 381
  - inverse: 344
  - unreliable: 43

## 2026 eval label accuracy

| Score normalization | Accuracy | Prediction distribution |
| --- | ---: | --- |
| mean | 4.98% | 201 unreliable |
| sum | 11.94% | 162 unreliable / 31 normal / 8 inverse |

Mean-logprob still fully collapsed to `UNRELIABLE`. Sum-logprob produced some tradable labels but remained heavily biased toward `UNRELIABLE` and had poor label accuracy.

## Strict replay

Strict replay used actual OHLC bars, one-bar delayed entry, fee/slippage, and strict MDD including intrabar adverse excursion.

| Method | Trades | CAGR | Strict MDD | CAGR / strict MDD | Mean trade ret | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| side-only SFT16 mean-logprob | 0 | 0.00% | 0.00% | 0.00 | 0.00% | 1.000 |
| side-only SFT16 sum-logprob | 30 | -30.94% | 14.64% | -2.11 | -0.485% | 0.072 |
| token_signature memory baseline | 79 | 9.37% | 11.38% | 0.82 | n/a | n/a |
| monthly history-majority baseline | 61 | 16.48% | 8.31% | 1.98 | n/a | n/a |

## Decision

No-go.

The side-only target fixed one legitimate mismatch, but it did not create a usable LLM trading head. The failure is informative:

1. The model can reduce training loss on short JSON side labels, but candidate scoring is still not calibrated for action selection.
2. `UNRELIABLE` remains an attractor despite being the minority class in both full and sampled datasets.
3. Sum-logprob partially escapes collapse but selects a losing subset; it underperforms simple memory baselines and is not statistically sufficient.
4. More steps on this exact target/scoring setup are unlikely to be the right next move. The issue is not only undertraining; it is action-label semantics and scoring calibration.

## Next direction

Do not integrate this checkpoint.

The next LLM/RLLM attempt should change the decision surface rather than merely increasing steps:

- train pairwise or preference-style side decisions (`normal` vs `inverse`) separately from a calibrated abstain/reliability head;
- remove `UNRELIABLE` from the same three-way candidate competition, because it dominates candidate logprob;
- make the LLM rank compact price-action states or event memories, then let a small causal execution policy map ranks to trade/no-trade;
- validate against `history_majority` and `token_signature` baselines before any long training run.
