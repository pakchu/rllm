# PSIM-D8 Cross-Protocol Disagreement Persistence Preregistration

Date: 2026-07-29  
Candidate: `PSIM-D8-CDP1`  
Artifact: `results/psim_d8_cross_protocol_disagreement_persistence_preregistration_2026-07-29.json`

## Decision

Resume alpha discovery with a new, source-structural PSIM-D8 family while the
PPOSM Gross9 future veto is paused for its missing exact frozen cache files.
The missing PPOSM inputs must not be rebuilt or substituted because doing so
would change the frozen candidate identity.

This registration was written before reading the PSIM-D8 card/event payloads,
computing source incidence, or opening market and funding outcomes.

## Why this is a new family

The previous PSIM-D8 RLLM2 line was terminally rejected after its frozen 2021
report-only transfer. `PSIM-D8-CDP1` does not repair or succeed that model:

- no Gemma or other language model output;
- no teacher relation labels or logits;
- no embeddings or generated text;
- no selected-subcard selector;
- no Ridge, FQI, Q-values, or residual rewards;
- no 2020 or 2021 economic outcomes.

The old terminal result and report are hash-bound in the registration so that
later execution cannot silently reinterpret this candidate as an RLLM2 repair.

## Frozen mechanism

For the `ARCHIVE_D90` daily card, every eligible Ethereum–Bitcoin relation unit
is compared on nine already-frozen structural fields:

1. event type;
2. revision-count bucket;
3. window-age bucket;
4. update-gap bucket;
5. dependency-delta state;
6. dependency-edge-delta-count bucket;
7. line-change-count bucket;
8. changed-section-count bucket;
9. Jaccard distance between changed-section sets.

The first eight components are exact categorical mismatches. Their mean with
the section Jaccard distance is the unit disagreement score. The arithmetic
mean across all eligible units is the daily disagreement score.

Two causal EWMAs are maintained over nonmissing daily scores:

- fast half-life: 3 eligible cards;
- slow half-life: 30 eligible cards.

Signals are disabled until 30 nonmissing cards have initialized the state.
Empty days do not alter either EWMA and cannot emit a signal.

## Frozen family

The nine candidates are the Cartesian product of:

- slow disagreement floor: `0.35`, `0.50`, `0.65`;
- absolute fast/slow gap: `0.05`, `0.10`, `0.15`.

Direction is fixed ex ante:

- short when persistent disagreement is high and accelerating;
- long when persistent disagreement remains high but is resolving;
- flat otherwise.

Entry is delayed by one complete five-minute bar after the daily decision.
Each position holds exactly 288 five-minute bars. Overlap is forbidden and the
first signal wins.

## Evaluation order

1. Source-only support gate, with no market or funding access.
2. 2022 selection among the frozen nine-member family.
3. Commit the single selected top1.
4. Untouched 2023 future veto.
5. Only after a standalone pass, separately preregister same-gross portfolio
   promotion once the exact Gross9 inputs are restored.

Costs are fixed at 6 bp per side, with 10 bp per-side stress, exact funding
cashflows, and strict intra-position adverse-excursion drawdown.

Failure at any stage retires `CDP1`; rank-2 substitution, threshold repair, or
post-result reinterpretation is forbidden.

## Access boundary at registration

- PSIM-D8 source payload rows parsed: `0`
- source incidence or values computed: `false`
- market rows parsed: `0`
- funding rows parsed: `0`
- economic metric sets computed: `0`

The generated manifest hash is
`8f564a0ceb967df9c7bb2cd3b719156aa75cde7883c3426d07a5be191ef5ae5d`.
