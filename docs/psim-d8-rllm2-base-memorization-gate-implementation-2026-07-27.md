# PSIM-D8-RLLM2 base memorization gate implementation

Date: 2026-07-27 KST

Status: implementation and non-inference validation pending final review;
official RLLM2 attempt not yet consumed

## Scope

This runner implements the separately preregistered
`PSIM-D8-RLLM2` operational successor. It reuses RLLM1's committed pure
source, redaction, prompt, roster, tokenizer, and exact-binomial functions.
It does not mutate or rerun RLLM1.

No market, funding, reward, or economic payload is available to this stage.
There is no absolute return, CAGR, strict MDD, trade count, or profitability
claim.

## Scientific identity

- inherited scientific contract hash:
  `59a7c1dd03155d8552614e4886087ca1dd08db4cc8c8953257c2f6f68d28af23`
- unchanged case-roster hash:
  `5065cd58322aee8f38f11ec2c4a186fb1a7ba8133aa2b2bb0182f67322a8bf39`
- exact model:
  `google/gemma-4-E4B-it`
- exact revision:
  `ee0ef6023621cff504d758262d4e04895a5af4a2`
- cases: 128;
- scoring: one forward, first assistant-token A-H logits;
- generated text: none;
- chance: `1/8`;
- Bonferroni rejection: `p < 0.01 / 3`;
- prompt truncation: prohibited;
- market access after this gate: still false.

## Sole runtime correction

The loader still passes `device_map={"": 0}` and keeps the exact 4-bit
NF4/double-quant/BF16/SDPA configuration.

Placement validation now requires:

- exact `Gemma4ForConditionalGeneration` class;
- quantized model;
- `model.device == cuda:0`;
- first parameter on `cuda:0`;
- positive CUDA allocation;
- if the optional `hf_device_map` is populated, every target must normalize
  to `cuda:0`;
- an empty optional `hf_device_map` is accepted.

The 128 real forwards remain the final functional placement check.

## Failure sealing

Unlike RLLM1, RLLM2 writes a terminal result artifact for ordinary Python
exceptions after the attempt sentinel:

- current stage;
- exception type and message;
- forwards started;
- predictions completed;
- zero-market/economics boundary;
- rerun authorization false.

The same fixed result path therefore contains either the completed
memorization verdict or a terminal operational failure. A process kill or
power loss can still leave only the immutable attempt sentinel, which also
prohibits rerun.

## Fixed one-shot sequence

1. Reject an existing RLLM2 attempt or result, including symlinks.
2. Require a clean `HEAD == origin/main`.
3. Validate the RLLM2 preregistration and all committed RLLM1 predecessor
   evidence.
4. Run inherited exact model-file/runtime, source, tokenizer, chat-template,
   roster, and all-prompt capacity validation.
5. Create the RLLM2 attempt sentinel.
6. Load the exact quantized model.
7. Validate concrete CUDA placement.
8. Score all 128 cases once.
9. Atomically write success/rejection or terminal operational failure.

`--validate-only` stops after step 4, loads no model weights, and creates
neither one-shot artifact.

Fixed paths:

- preregistration:
  `results/psim_d8_rllm2_operational_successor_preregistration_2026-07-27.json`
- attempt:
  `results/psim_d8_rllm2_base_memorization_gate_attempt_2026-07-27.json`
- result:
  `results/psim_d8_rllm2_base_memorization_gate_2026-07-27.json`

## Required evidence before official execution

- dedicated unit and guard tests pass;
- direct script entrypoint works without `PYTHONPATH`;
- `--validate-only` reproduces the frozen roster and prompt maxima;
- official RLLM2 attempt/result remain absent;
- independent code review approves;
- implementation is committed and pushed;
- disk remains below 300 GiB.

Current non-inference evidence:

- dedicated runner tests: `7 passed` in 7.28 seconds;
- direct `--validate-only`: passed;
- inherited policy prompts: 1,461, maximum 30,961 tokens;
- inherited challenge prompts: 128, maximum 10,291 tokens;
- case-roster hash reproduced exactly;
- scientific-contract hash reproduced exactly;
- model weights loaded: false;
- official attempt consumed: false;
- WSL usage: approximately 298 GiB.
