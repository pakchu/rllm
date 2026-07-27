# PSIM-D8-RLLM1 semantic/RLLM preregistration

Date: 2026-07-27 KST

Status: frozen source-only preregistration; no market/model outcome opened

Candidate: `PSIM-D8-RLLM1`

## Decision

PSIM-D8 passed its one and only source-support run and is permanently closed
to source reruns, repairs, or a D9 successor. This document does not modify
PSIM-D8. It freezes a new downstream candidate that consumes the committed
D8 artifacts read-only.

The first economic path is deliberately conservative:

1. select exactly one D8 relation subcard by an outcome-blind hash;
2. filter memorization-quarantined units only at the model boundary;
3. produce a fully redacted, digit-free selected-subcard prompt;
4. run the base-model identity-recovery gate before any market access;
5. test frozen Gemma semantic embeddings with deterministic fitted-Q
   policies;
6. authorize QLoRA-RL only if those embeddings transfer and beat every
   nonsemantic control;
7. select on 2022 and evaluate one immutable policy on 2023.

This hierarchy prevents a high-capacity LoRA policy from hiding the absence
of source-semantic alpha.

## Frozen D8 authority

- terminal-result repository commit:
  `f5695fdf3e67144f7a8741fa6486ce41df7ade3b`
- sealed source-execution implementation commit:
  `17e17fa96ddb7866ffda0d67727b8630737188f5`
- source result:
  `results/protocol_specification_intent_maturity_d8_source_support_2026-07-27.json`
  - SHA-256:
    `0b92b476b654cd76f0cf9dc004690cbcb78e7a5e73917b5d66611c0460d00204`
  - result hash:
    `7104593f0c0aa32e9f1219ab075fa10261058b57460286eaddf3e6764626fba5`
- events:
  `data/protocol_specification_intent_maturity_d8_events_2020_2023.jsonl.gz`
  - SHA-256:
    `d7308789176af4bfe1bb2f5f13c89d6811bc7f938f3ecec08b1bf8acc5f7e2b2`
- cards:
  `data/protocol_specification_intent_maturity_d8_cards_2020_2024q1.jsonl.gz`
  - SHA-256:
    `ce1bd1bd9a24068e6e223efca323db805781e912eadb0d2a8b7d63610fab96c1`

The downstream relation is explicitly named
`SELECTED_SUBCARD_RELATION`. It is not presented as a semantic aggregation of
the complete logical day.

## Selected-subcard mechanism

Only `ARCHIVE_D90` is economically admissible. For each logical card:

```text
material =
  prior_card_hash || NUL ||
  complete_relation_roster_sha256 || NUL ||
  decision_at || NUL ||
  "PSIM_D8_RLLM1_SUBCARD_SELECTOR_FIXED_20260727"

ordinal = uint64_be(SHA256(material)[:8]) mod subcard_count
```

The selector inputs are never model-visible. Every logical-card, complete
roster, manifest, contiguous range, payload hash, and subcard hash-chain
binding must validate before the slice is used. There is no fallback to a
different subcard after quarantine, excessive token count, model failure, or
poor economics.

The model payload removes `memorization_excluded=true` relation units without
changing the frozen source artifact. If no eligible relation remains:

```text
RELATION = INSUFFICIENT_EVIDENCE
TARGET   = current target, or flat at a split boundary
```

Within-card event payload deduplication is lossless dictionary compression:
every ordered relation edge and multiplicity remains. Calendar rows and train,
test, and eval denominators are never deduplicated.

## Model-visible boundary

The prompt includes only:

- one verified selected-subcard slice;
- every eligible relation edge in original order;
- redacted event text and frozen categorical metadata;
- local alphabetic event references;
- current position.

It excludes proposal IDs and titles, hashes, paths, dates, authors, URLs,
emails, addresses, release/fork names, status, raw numeric fields, market
data, funding, rewards, PnL, and economic statistics.

Redaction is deterministic and ordered:

1. Unicode NFKC and newline normalization;
2. title/name fields;
3. URLs and email addresses;
4. explicit EIP/ERC/BIP references;
5. addresses, hexadecimal values, and long hashes;
6. dates and versions;
7. a frozen proposal-lifecycle status lexicon;
8. a frozen release/fork lexicon;
9. every remaining digit run; and
10. horizontal whitespace.

The lifecycle scrub includes `draft`, `review`, `last call`, `final`,
`withdrawn`, `stagnant`, `active`, `rejected`, `deferred`, `replaced`,
`obsolete`, and the other frozen lifecycle terms. They become the
non-directional `<LIFECYCLE>` marker before model rendering.

Numeric bucket names are replaced by deterministic letters-only opaque
tokens, preserving category identity without exposing their numeric edges.
All 1,461 historical policy prompts contain zero literal digits.

No prompt truncation is allowed. The frozen maximum is 32,768 tokenizer
tokens; any over-cap prompt retires the candidate rather than changing the
slice or clipping evidence.

## Source-only capacity result

The preregistration inspected source artifacts only:

- D90 logical decision cards: 1,461
  - train 2020-2021: 731
  - test 2022: 365
  - eval 2023: 365
- selected relation units:
  - minimum: 1
  - maximum: 64
  - mean: 2.9514
- eligible units after quarantine:
  - minimum: 0
  - maximum: 64
  - mean: 2.5051
  - no-eligible cards: 117
- rendered prompt UTF-8 bytes:
  - maximum: 122,113
  - mean: 7,730.10

The tokenizer runtime gate, not this byte count, decides whether every prompt
fits the exact model context.

## Memorization gate

The inherited `PSIM_MEMORIZATION_V1` challenge is unchanged:

- at most 16 nonquarantined events per protocol/effective-year, selected by
  lowest frozen event hash;
- one true proposal ID and seven distinct same-protocol, same-year decoys;
- forced choice without abstention;
- exact one-sided binomial tests against `1/8` for Ethereum, Bitcoin, and the
  combined sample;
- Bonferroni rejection at `p < 0.01 / 3`;
- minimum 32 events per protocol.

Source support provides 64 challenge events for Ethereum and 64 for Bitcoin.
The exact base model must pass before any market row is opened. The final
fine-tuned model must pass again after 2022 selection and before 2023 market
access. Failure permits no redaction repair, resampling, model swap, or
candidate repair.

## Model and RLLM structure

Exact base:

```text
google/gemma-4-E4B-it
revision ee0ef6023621cff504d758262d4e04895a5af4a2
```

Runtime:

- one visible CUDA device;
- exact `Gemma4ForConditionalGeneration` checkpoint with no
  image/audio/video input;
- LoRA attached only to `model.language_model`;
- 4-bit NF4 double quantization with BF16 compute;
- last non-padding hidden state;
- one bias-free selected-subcard relation head;
- one bias-free three-action target head;
- no decoded trading generation;
- no additive long/short calibration;
- one model forward per logical decision.

The repository retains its previously validated pinned Transformers revision.
Current upstream examples recommend newer releases, so compatibility is a
mandatory synthetic runtime gate rather than an assumption:

- [Transformers Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [Google Gemma 4 prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [Google Gemma 4 QLoRA](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)

The base model first creates source-only relation teacher labels under a
hash-permuted forced-choice code. Invalid output becomes `ABSTAIN`.

The cheap semantic gate uses frozen Gemma embeddings, train-year-only
32-component PCA, and fixed ridge/Extra-Trees fitted-Q learners. It must
transfer from 2020 to 2021, beat every nonsemantic control, and then pass a
from-scratch 2020-2021 to 2022 transfer gate before QLoRA is authorized.

Conditional QLoRA is frozen to:

- seeds: `20260727`, `20260728`;
- checkpoints: optimizer steps `80`, `160`, `240`;
- LoRA rank 16, alpha 32, dropout 0.05;
- q/k/v/o and gate/up/down projections;
- micro-batch 1, accumulation 8;
- AdamW learning rate `2e-5`, weight decay `0.01`;
- no future-best hard action label.

The actor consumes fitted action values from the selected semantic-encoder
policy:

```text
Q = center_per_state(Q)
Q = clip(Q, -0.10, +0.10)

loss =
  -sum(softmax(action_logits) * stop_gradient(Q))
  + 0.01 * KL(policy || uniform)
  + 0.05 * selected_subcard_relation_cross_entropy
```

## Economics and statistical gates

The candidate reuses `training/bctp_strict_economics.py` and
`training/bctp_transition_labels.py`:

- targets: -0.5 / 0 / +0.5 account gross;
- base cost: 6 bp;
- stress cost: 10 bp;
- execution-delay stress: one complete five-minute bar;
- one-day staleness: mandatory diagnostic, never a rescue result;
- exact interior and conservative boundary funding;
- full-calendar CAGR using 365.2425 days, including all flat time;
- one global strict-MDD high-water mark;
- favorable then adverse intrabar path;
- virtual liquidation and terminal flatten costs.

Mandatory controls include constants, persistence, exact memory, metadata and
topology baselines, shuffled relations/rewards, old/new disruption,
future/status scrub, one-protocol ablations, semantic masking, direction flip,
and neutral action-code permutation. Failed and flat variants remain in the
shared weekly max-stat family.

The selected QLoRA checkpoint must pass independently on 2022 and the one
immutable policy must pass on 2023:

- positive absolute return;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- at least 100 nonflat intervals;
- long and short each at least 20% of nonflat intervals;
- positive first-half and second-half returns;
- positive 10 bp stress and delayed returns;
- corrected weekly significance.

Every economic report must include absolute return, CAGR, strict MDD,
CAGR/strict-MDD, trade count, direction shares, half-year returns, stress and
delay returns, and the weekly p-value.

## Access ledger and current result

This preregistration read only the five frozen D8 source-authority paths.

- market rows parsed: 0
- funding rows parsed: 0
- market/funding payload bytes hashed: false
- model loaded: false
- model outputs created: 0
- rewards created: 0
- economic metrics computed: 0
- test outcomes opened: false
- eval outcomes opened: false

Canonical artifact:

```text
results/psim_d8_rllm1_preregistration_2026-07-27.json
SHA-256
6f143fdb5f61697defe2cbc9b7b15ce8aaf0da4980c3ff0ba0f4f994cc68a78f
manifest hash
e7a5630aa877ede9d97ee1376acd24f101243ab203281d531e096a3fdfa096bc
```

There is no absolute return, CAGR, strict MDD, trade count, or profitability
claim yet. The next authorized unit is source-only implementation and review
of selected-subcard export, redaction, exact-model runtime, and memorization
gates.
