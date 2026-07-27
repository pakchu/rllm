# PSIM-D8-RLLM2-S1 source-feature seal preregistration

Date: 2026-07-27
Status: **preregistered; no market, funding, reward, or economic payload opened**

## Purpose

The RLLM2 base memorization gate passed at chance-level accuracy and authorized
source-feature construction only. This stage freezes the last missing
pre-market artifacts:

1. the deterministic selected-subcard source rows;
2. exact `POSITION_FLAT` policy prompts;
3. exact randomized A–F relation-teacher prompts;
4. frozen Gemma 4 E4B last-token embeddings; and
5. frozen source-only relation-teacher logits.

No profitability claim is made here. The sole success action authorizes a
separate, later opener for **2020 train outcomes only**.

## Frozen authority

- Model: `google/gemma-4-E4B-it`
- Revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`
- RLLM2 base-gate result SHA-256:
  `0abf3b5babe9e398e97721ddcc3e29b6d23cc742345cd5f804e78d507982818f`
- RLLM2 base-gate result hash:
  `8debfe4b37a6be1f65b306cce5b1408bf21a01a7f316254e4b42c2529a851ce3`
- Inherited scientific contract hash:
  `59a7c1dd03155d8552614e4886087ca1dd08db4cc8c8953257c2f6f68d28af23`

The predecessor passed with 15/128 correct memorization choices versus
12.5% chance and did not authorize market access.

## Source roster

The source roster is fixed before model inference:

- 1,461 daily `ARCHIVE_D90` rows;
- 731 train rows (2020–2021);
- 365 test rows (2022);
- 365 eval rows (2023);
- 117 rows with no eligible relation after the frozen memorization
  quarantine; and
- 1,344 rows requiring a relation-teacher forward.

Every row binds the card and selected-subcard hashes, selector digest, redacted
source payload, exact policy prompt, exact relation-teacher code permutation,
both prompt hashes, and a canonical row hash.

## Model operations

One quantized Gemma process is used with micro-batch one.

- **Embedding:** all 1,461 rows use the exact policy prompt with
  `CURRENT_POSITION=POSITION_FLAT`. The artifact stores the final non-padding
  vector from `model.model(...).last_hidden_state`, shape `(1461, 2560)`,
  `float32`, with `use_cache=False` and without all-layer hidden-state dumps.
- **Relation teacher:** 1,344 eligible rows use a separate prompt ending in
  `RELATION_CODE=`. Only the exact single-token logits A–F are consumed.
  There is no decoding. Exact finite ties choose the lexically first code.
  Nonfinite code logits become `ABSTAIN`.
- **No eligible relation:** the teacher forward is skipped and the frozen
  relation is `INSUFFICIENT_EVIDENCE`.

No prompt may exceed 32,768 tokens. There is no truncation, replacement
subcard, resampling, or output-based prompt change.

## Interruption and failure rules

The attempt sentinel is written before model weights load. A process
interruption may resume only when:

- the attempt exists and no terminal result exists;
- execution commit, runner, preregistration, and source roster are unchanged;
- checkpoint shards form a contiguous hash-verified prefix; and
- the explicit resume mode is used.

A caught post-sentinel exception writes terminal rejection. It cannot be
repaired, rerun, resampled, or switched to another model. Checkpoints are
deleted only after every final artifact and the terminal result verify.

## Access boundary

This preregistration and its runner may read frozen D8 source cards, the exact
RLLM2 authority artifacts, tokenizer/runtime files, and the pinned model
snapshot. They may not open, parse, hash, or infer from:

- BTC market data;
- funding data;
- rewards or action values;
- test/eval outcomes; or
- any economic metric.

The next stage remains closed until a successful source-feature seal is
committed and independently verified.
