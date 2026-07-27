# PSIM-D8-RLLM2-S2 chunked source-feature terminal rejection

Date: 2026-07-27

Status: **TERMINAL REJECT — S2 may not be resumed, repaired, or rerun**

## Result

The official source-only S2 attempt ran from execution commit
`2c4c89ec41675144a67fe1e0254737dfb9953dcf`. It rejected at the frozen
pre-market equivalence gate before the long-context capacity gate or full
source extraction:

- equivalence cases passed: **3 / 10**;
- completed source rows: **0 / 1,461**;
- one-pass reference forwards started: **20**;
- chunk forwards started: **102**;
- total model forwards started: **122**;
- long-context capacity gate executed: **false**;
- market/funding rows parsed: **0 / 0**;
- rewards/economic metrics created: **0 / 0**; and
- 2020, test, and eval outcomes opened: **false**.

The terminal result sets `resume_authorized=false` and
`rerun_authorized=false`. No S2 checkpoint or final source-feature artifact
exists.

## Exact evidence

- Attempt SHA-256:
  `d5f69f4bbdfb99d6a6f04e0cf30d5c9a212a80edc50c875a3d4b8dba6076529d`
- Attempt hash:
  `956c542f6cc3fa7190aa88933d9f4e62e4d759eed9484cb38d260cfb7757fa7c`
- Equivalence result SHA-256:
  `4d145777f91b6d2777412e280f967a9d21c7349dc7993bd18d9b08e42efb0b80`
- Equivalence result hash:
  `86cb443a75ce46347f5e32f7ecb7dc5da37c9969a7e3e82c1a16c0164b243f13`
- Failure result SHA-256:
  `85cac32a947fb417686183e0447fdd248bfa54754fe4a1de98fafc1cd80f5613`
- Failure result hash:
  `305e53e231edc942adf7f3de73aff20f816cb6eae38dd4530bab263d6cc082a9`
- Executed runner SHA-256:
  `fe798475ae32e7c2ab42a1a14a1f85b350cc2f7b928f7d3450fb1b46408b5849`
- Raw failure log SHA-256:
  `1e13e833438a360c6acf25ea4408b1d29662576b8a86d839d422831996011631`

## Equivalence evidence

The three 327/328-token cases, which fit in one 512-token chunk, matched
exactly. Every case requiring more than one chunk failed at least one frozen
numeric threshold.

Across the seven multi-chunk failures:

- minimum embedding cosine similarity:
  `0.9995441801113741`;
- maximum embedding RMS absolute delta:
  `0.16501405624525725`;
- maximum embedding absolute delta: `1.125`;
- relation code agreement: **6 / 7**; and
- row 1265 relation code changed from reference `A` to candidate `B`.

The evidence therefore rejects the claim that the fixed-512 cache scan is the
same source operator as the frozen one-pass SDPA computation. The exact
single-chunk matches and immediate multi-chunk divergence localize the
operator difference to cache/chunk execution rather than tokenization or
prompt reconstruction. BF16/SDPA execution order and Gemma4 hybrid
sliding/global cache behavior are plausible contributors, but this terminal
record does not claim a uniquely proven kernel-level cause.

## Authorized research direction

S2 itself remains terminal and none of its model outputs may be reused.
2020 outcomes remain sealed.

A fresh successor may be considered only under a new preregistration that:

1. binds this exact terminal evidence and prohibits S1/S2 checkpoint or model
   output reuse;
2. makes no one-pass-equivalence claim for the fixed-chunk representation;
3. defines the chunked-native representation as a scientifically distinct,
   source-only operator before opening outcomes;
4. uses a fixed repeatability/integrity gate and the original long-context
   capacity row before full extraction;
5. keeps the exact model, revision, tokenizer, prompts, source roster,
   relation mapping, and no-market boundary unchanged; and
6. terminally rejects on repeatability, placement, memory, source-integrity,
   or artifact failure.
