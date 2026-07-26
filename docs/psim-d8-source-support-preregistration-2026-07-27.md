# PSIM-D8 source-support preregistration

Date: 2026-07-27

Status: frozen source-only preregistration

Candidate: `PSIM-D8`

Policy: final PSIM source-representation successor; **NO D9**

## Decision

PSIM-D7 was rejected at source Gate 5 before a daily card, model output,
market row, outcome, trade, PnL, CAGR, or strict MDD existed. Its terminal
result hash is
`45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b`.

The read-only post-terminal forensic result
`620d81baadafaa9d5cee1e5c38883846d1ac2df60acd00b67117241d87184144`
showed a narrow representation-capacity mismatch:

- 24 logical schedule/day cells exceeded the inherited limit;
- the first overflow had 143 relation units;
- the largest cell had 1,221 relation units;
- the inherited model-card limit was 64;
- no market, model, outcome, reward, trade, or PnL data were accessed.

PSIM-D8 therefore changes only the relation-card representation. Every
unrelated D7 source, parser, event, split, schedule, quarantine, gate, and
control contract remains frozen.

## Frozen mechanism

Selected mechanism:
`PSIM_D8_LOGICAL_DAY_CARD_WITH_ORDERED_RELATION_SUBCARDS_V1`.

Mechanism probe:

- commit:
  `211454a96695de44af3e009b751eff7df9e3ae5f`
- result:
  `results/protocol_specification_intent_maturity_d8_mechanism_probe_2026-07-27.json`
- artifact SHA-256:
  `9c926f1fc44e60e4fcf92679dfd36db8d410220dcbbecec8c71e05bba0076d76`
- result hash:
  `3b690e6e11399a12aca41a2ba79f74f5d8642f029dc5241d72d342a6f3706672`
- synthetic scenario roster:
  `9a718845c1af15904a9d263511c601432d1ae3e2ddd17bad9e9bfb2fbefcc00c`
- producer SHA-256:
  `b869c73b4ce1de783ff67888825ba3d0c41d8075050afe1c9b8ee6579f76fb4d`
- test SHA-256:
  `9a2a99ab3ad1e4e4bf7ff98515831532142d39761f007ecc26e2d3b280e8fbc9`
- decision document SHA-256:
  `a46b2960586a2e80f78600a4069481de03ca80048c436e67fe09afe513d36385`

The synthetic battery passed 12/12 scenarios, including relation counts
1, 64, 65, 70, 143, and 1,221 and tamper rejection for ranges, payloads,
hash chains, complete rosters, logical-card identity, and empty rosters.

## Exact D8 representation contract

1. There is exactly one logical `DailyCard` per archive schedule and
   decision day.
2. Its source/audit payload retains the exact complete D7 ordered
   `relation_units` roster.
3. The roster is partitioned into deterministic contiguous slices:
   `start = 64*k`,
   `end_exclusive = min(64*(k+1), N)`.
4. Every model-eligible subcard contains at most 64 relation units.
5. Slices are complete, ordered, non-overlapping, gap-free, and
   duplication-free.
6. Each manifest row binds schedule, decision time, ordinal, count, range,
   subcard payload hash, prior subcard hash, and complete-roster hash.
7. The completed subcard manifest is bound by the logical local-payload
   hash and the inherited logical card hash.
8. The full logical card and audit manifest are not model-visible.
9. A later model may receive only one verified subcard slice under a
   separate preregistration.
10. Model aggregation is
    `UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION`.
11. Dropping, sampling, summarizing, raising the cap, or choosing
    partitions using market/outcome information is forbidden.
12. Relation-control denominators remain unique logical decision days,
    never subcard counts.

Contract hash:
`c86aaf1e9975d62c88c45f89dc6943fef7e2ed8902ecc840ea9f569e09e1e0fb`.

## Inherited source contract

Unchanged from D7:

- official EIP/BIP repositories and sealed tips;
- 2020-01-01 through 2023-12-31 source interval;
- archive schedules and split boundaries;
- D6 lossless UTF-8 event transport;
- D7 Bitcoin grammar overlay;
- quarantine roster;
- all 13 source gates in the same order;
- all seven relation controls;
- unique-day control eligibility and denominator;
- no market/model/outcome access before every source gate passes.

Only namespaces, failure actions, and relation-subcard representation are
different. The recursive D7-to-D8 delta contains exactly 47 paths.

- authorized delta hash:
  `33db9ba0fea552e24d62d16cd4bda84973fdae351977eb35f376df08599c543f`
- batch hydration contract hash:
  `7eab28547cabb3aacf0c2cfa0498cc26e3de6b36d7e2f8d7b1a80fd6823d048d`
- execution authorization contract hash:
  `06f1012e5fe9246286a9d0b28da53877f846ef3c917a3023158893a601c0456f`
- exact D7 authority path/commit/artifact binding hash:
  `662a7fba3c6d5c86590c472e208f6479aaab6f593e04405652793bdec747a80f`

## Fresh-root and execution boundary

- D8 source root: `/tmp/psim-d8-source`
- D8 sealed ref: `refs/psim-d8/sealed-tip`
- `/tmp/psim-d7-source` is terminal forensic residue and must never be
  opened, repaired, reused, deleted, or rerun by D8.
- D1-D7 source-object reuse is forbidden.
- This preregistration **does not authorize official source execution**.
- A reviewed D8 implementation commit, reviewed D8 test commit, and a
  canonical direct-child execution seal are all required first.
- The synthetic mechanism probe does not authorize source execution.
- D8 may receive one official source attempt after those prerequisites.
- If any D8 source gate fails or another source change is needed, PSIM is
  permanently retired. **NO D9** is authorized.
- A source pass would still not establish alpha. Memorization/model and
  economic stages require separate preregistrations.

The D8 preregistration imports the D8 mechanism module, whose authority
replay dependency graph reaches the D7 read-only forensic and source-runner
modules. This transitive import is allowed only to reuse canonical hashing
and authority schemas. Import and preregistration tests explicitly fail if
the D7 audit or official source runner is invoked, and audited file reads
reject every `/tmp/psim-*` source-root access.

## Canonical preregistration artifact

- artifact:
  `results/protocol_specification_intent_maturity_d8_preregistration_2026-07-27.json`
- artifact SHA-256:
  `4dd083cfd54b227c6e5d373564270bcc7fb2f1002bec78a870d9d609176bb605`
- manifest hash:
  `d06124a48dde7e79f18d7627297f3d17196f6296fdb6221e42c40e7b032313dc`
- producer:
  `training/preregister_protocol_specification_intent_maturity_d8.py`
- producer SHA-256:
  `ea9d35658fa497df1dac4226bd71e485611406317208c2bdfe55cb0e04886158`
- tests:
  `tests/test_preregister_protocol_specification_intent_maturity_d8.py`
- test SHA-256:
  `d4b51287f7ade6c8885736f7c5a519056795457fc0785e0f51f55fae990f7ebc`

## Economic status

No return, absolute return, CAGR, strict MDD, CAGR/strict-MDD, trade count,
hit rate, or significance statistic exists for PSIM-D8. It is only a
source-support preregistration.
