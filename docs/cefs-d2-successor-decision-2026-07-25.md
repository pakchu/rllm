# CEFS-D2 successor decision

Date: 2026-07-25

## Decision

Select **CEFS-D2 — Cboe Edge-Flip Sequence Policy, clean-room successor** as
the next source-only candidate.

CEFS-D2 is not a retry or reinterpretation of CEFS-D1. CEFS-D1 remains
terminally retired by:

```text
results/cboe_edge_flip_sequence_policy_source_rejection_2026-07-25.json
SHA256 4c2839e5ac59738367d5116ff05ed50c900d16f06de8c4d4cc724fd25978c169
result_hash 9963981f6d56fcff65f1367fc7c3c1fc006b60b821894b3e0ff59c6b9aa35d7b
```

That rejection occurred at the authority gate because an external zsh
preflight loop assigned to the reserved variable `path`, which erased
`PATH`. It occurred before the source loader and opened no Cboe value row,
relation edge, prompt, control, BTC row, funding row, return, reward, model
row, action, trade, PnL, CAGR, or MDD.

## Why a successor is admissible

The exact CEFS mechanism remains unobserved. No source-support incidence and
no economic outcome informed this successor decision. CEFS-D2 therefore
changes only execution identity and authority handling.

The alternative CLOR release-ordering family remains a reserve because it
still carries:

- source-composition failures in predecessor repo mechanisms;
- conservative multi-day release availability;
- current-vintage revision risk; and
- no frozen joined live-parity source.

Switching to CLOR would add source risk without reducing contamination:
CEFS-D1 did not reach source evidence.

## Frozen no-scientific-change rule

CEFS-D2 must inherit byte-bound scientific authority from CEFS-D1:

```text
CEFS-D1 boundary
  docs/cboe-edge-flip-sequence-policy-boundary-2026-07-25.md
  SHA256 d0b522a7ac87e3526d6cd740bb81304bd73042bc327978660eb551b159c16ec3

CEFS-D1 preregistration
  results/cboe_edge_flip_sequence_policy_preregistration_2026-07-25.json
  SHA256 5e515663e99ef4aa322cae25cfb2c07f69b3e24f289bc2f0f79463aca64a8878
  manifest_hash 9aa7c891ec241d4733db215068bed3507f41c03cbae7198c906a079ddb6467bf

sealed pure support engine
  training/build_cboe_edge_flip_sequence_policy_support.py
  commit d7213f647128fc6160672bc61f080b3dcf7d1f42
  SHA256 2069084d65146540488672115ee09f292cd31e6611bf92a569d534ab8a74c688
```

The successor may not change:

- any Cboe source, field, header, date set, or source horizon;
- any of the twelve exact relation formulas or edge levels;
- the five ordered states;
- any prompt byte or current-position context;
- the `TARGET_LONG|TARGET_FLAT|TARGET_SHORT` action space;
- the D+1 09:30/09:35 New York availability/entry clock;
- the exact 288 × 5-minute hold;
- global overlap reservation or equality acceptance;
- TRAIN 2020–2021, TEST 2022, EVAL 2023;
- any source-support count, diversity, drift, or control threshold;
- any of the eight controls; or
- any first-stop, no-repair, forbidden-access, or outcome boundary.

## Only permitted change

CEFS-D2 receives new:

- policy ID;
- boundary, preregistration, contract, runner, tests, execution seal;
- source/control/pass/rejection artifact paths; and
- absolute Git executable authority.

The runner must use:

```text
/usr/bin/git
SHA256 2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

It must verify that `PATH` exists and includes `/usr/bin`, but its authority
checks may not depend on executable-name lookup. The official command must
invoke the repository Python directly and may not use a shell preflight
variable named `path`.

## Stop rule

After a new execution seal is committed, CEFS-D2 receives one source-only
run. Any failed gate retires CEFS-D2 unchanged. No D3 may adjust a scientific
field, threshold, clock, control, or source based on D2 incidence.

Only a complete source-support pass may authorize a separately frozen
economic/RLLM evaluator. Until then CEFS-D2 has no alpha, profitability, or
deployability claim.
