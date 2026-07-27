# PSIM-D8-RLLM1 base memorization gate implementation

Date: 2026-07-27 KST

Status: implementation and source/tokenizer validation only; official model
inference not yet executed

## Purpose

This gate asks whether the exact pretrained Gemma model can recover proposal
identity from the supposedly redacted D8 source evidence at a rate above the
preregistered `1/8` chance baseline. A rejection terminates
`PSIM-D8-RLLM1` before any market, funding, reward, or economic payload is
opened.

This is not an alpha backtest and contains no absolute return, CAGR, strict
MDD, trade count, or profitability claim.

## Pre-model correction

Review found that the original source-only challenge roster admitted 64
events with empty redacted evidence. Running that roster would have tested
candidate-number priors rather than evidence-based identity recovery.

Before any model inference:

- eligibility was narrowed to nonquarantined events with non-empty redacted
  normalized delta text;
- the original event-selection, decoy, and decoy-order salts were retained;
- true codes were assigned by a separate frozen hash, exactly twice per
  protocol/effective-year;
- all eight codes are therefore true eight times per protocol and sixteen
  times combined.

The canonical corrected preregistration is:

- artifact SHA-256:
  `6e9b034744acc1b701a283c7ba34e2bcc533e781c33bbe95657060f78c67732e`
- manifest hash:
  `31b86a25fedfe9c3ef98cfa4a3b617a8df7bddf68cc6eab6be33fa66069e4d89`
- challenge version: `PSIM_MEMORIZATION_V1_ERRATUM1`

## Frozen challenge

- 16 Ethereum and 16 Bitcoin events for each effective year 2020-2023;
- 128 total cases;
- case-roster hash:
  `5065cd58322aee8f38f11ec2c4a186fb1a7ba8133aa2b2bb0182f67322a8bf39`;
- one true proposal ID and seven same-protocol, same-year decoys;
- raw user content ends with `ANSWER=`;
- the frozen Gemma chat template then appends its three-token assistant prefix;
- one forward pass scores exact single-token codes `A` through `H`;
- no generated or decoded answer text;
- exact ties choose lexical code;
- exact one-sided binomial tests for Ethereum, Bitcoin, and combined;
- Bonferroni rejection at `p < 0.01 / 3`.

The exact rejection boundaries are:

- Ethereum or Bitcoin: 16/64 does not reject; 17/64 rejects;
- combined: 27/128 does not reject; 28/128 rejects.

## Exact runtime

- model: `google/gemma-4-E4B-it`
- revision:
  `ee0ef6023621cff504d758262d4e04895a5af4a2`
- architecture: `Gemma4ForConditionalGeneration`
- one visible CUDA device;
- 4-bit NF4 with double quantization and BF16 compute;
- SDPA attention;
- maximum input length: 32,768 tokens;
- no truncation;
- maximum peak allocated VRAM: 30 GiB;
- no image, audio, video, generation, market, or funding input.

The local snapshot is pinned by all seven preregistered file hashes and the
Transformers source revision. Exact challenge token IDs are:

| Code | Token ID |
|---|---:|
| A | 236776 |
| B | 236799 |
| C | 236780 |
| D | 236796 |
| E | 236788 |
| F | 236811 |
| G | 236823 |
| H | 236814 |

## One-shot execution safety

The official path has no configurable output override.

1. Require a clean worktree and `HEAD == origin/main`.
2. Validate the corrected preregistration, frozen source hashes, exact model
   files/runtime, tokenizer, chat template, and all prompt capacities.
3. Atomically create the fixed attempt sentinel before model construction.
4. Load the model and score all 128 cases.
5. Atomically create the fixed result artifact.

If the process fails after the attempt sentinel is created, the attempt is
consumed and cannot be rerun by the implementation. `--validate-only` never
loads model weights and never creates the sentinel.

Fixed paths:

- attempt:
  `results/psim_d8_rllm1_base_memorization_gate_attempt_2026-07-27.json`
- result:
  `results/psim_d8_rllm1_base_memorization_gate_2026-07-27.json`

## Access boundary

The implementation may read only:

- corrected preregistration artifact;
- frozen D8 event source;
- frozen D8 card source;
- exact local Gemma snapshot/runtime metadata and weights.

It does not import a dataframe, database, backtest, execution, or economic
module. Market and funding paths are neither read nor hashed.

## Current validation and next action

At implementation time:

- official attempt artifact: absent;
- official result artifact: absent;
- WSL usage: approximately 298 GiB, below the 300 GiB cap;
- exact E4B snapshot: present;
- official model-weight inference: not yet run.

Fresh validation evidence:

- dedicated runner tests: `12 passed` in 92.56 seconds;
- corrected preregistration tests: `10 passed` in 118.79 seconds;
- `--validate-only`: completed without creating an attempt;
- policy prompts: 1,461, range 320-30,961 tokens;
- challenge prompts: 128, range 159-10,291 tokens;
- all 1,589 prompts fit the 32,768-token cap without truncation;
- model weights loaded during validation: false.

The next permitted action is:

1. pass the dedicated runner tests;
2. run `--validate-only`;
3. obtain independent code-review approval;
4. commit and push this implementation;
5. execute the official gate exactly once.
