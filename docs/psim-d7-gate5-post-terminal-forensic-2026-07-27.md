# PSIM-D7 Gate-5 post-terminal forensic audit

Date: 2026-07-27 KST

## Result

The exact Gate-5 exception was:

```text
ValueError: PSIM-D7 relation card exceeds frozen event bound
```

This result came from a separate read-only forensic audit. It did not invoke
the official D7 `run` command, repair D7, continue D7, use the network, or
open market/model/outcome data.

Canonical result:

```text
results/protocol_specification_intent_maturity_d7_gate5_forensic_2026-07-27.json
SHA-256   35f961d2bde8a71045209698eee1c5508108218726b73fd2d3ceff35de85ab9b
result    620d81baadafaa9d5cee1e5c38883846d1ac2df60acd00b67117241d87184144
```

## Cardinality statistics

The frozen card limit was 64 relation units.

| statistic | result |
|---|---:|
| all source events | 5,356 |
| model-visible events | 4,261 |
| administratively quarantined events | 1,095 |
| overflowing schedule/day cells | 24 |
| overflow cells per archive schedule | 6 |
| completed daily cards | 0 |

The first overflow in official construction order was:

| field | value |
|---|---:|
| schedule | `ARCHIVE_D2` |
| decision day | 2020-10-02 |
| Ethereum events | 143 |
| Bitcoin events | 0 |
| relation units | 143 |

The maximum observed cardinality was:

| field | value |
|---|---:|
| schedule | `ARCHIVE_D2` |
| decision day | 2022-05-08 |
| Ethereum events | 407 |
| Bitcoin events | 3 |
| same-day events | 410 |
| Cartesian relation units | 1,221 |

The same source-day bursts shift by the frozen 2-, 7-, 30-, and 90-day
archive delays, so each schedule contains six over-limit cells.

## Source composition

| protocol | events | model-visible | administrative |
|---|---:|---:|---:|
| Ethereum | 4,985 | 3,890 | 1,095 |
| Bitcoin | 371 | 371 | 0 |

Ethereum administrative events consist of 730 ordinary administrative
quarantines and 365 exact migration restorations. They were correctly
excluded before card construction and did not cause the overflow.

## Root cause

For a day containing only one protocol, D7 creates one relation unit per
new event. For a day containing both protocols, it creates the full
same-day Cartesian product:

```text
relation_units = ethereum_events * bitcoin_events
```

The source history contains legitimate bulk proposal updates. The fixed
64-unit bound therefore conflicts with the preregistered lossless
no-truncation relation construction. The D7 Bitcoin grammar fix succeeded;
the new rejection is a separate representation-capacity defect.

## Integrity evidence

The forensic audit:

- executed 10,573 local Git commands;
- executed zero network commands;
- left the source-tree manifest unchanged;
- left the terminal artifact unchanged at SHA-256
  `36702b4737f1bb37e901241a96e04f30e77132bb6a18ade1fab277a83f15557e`;
- did not produce cards, controls, models, trades, PnL, CAGR, or strict MDD.

PSIM-D7 remains terminally rejected. Its source root is not a successor
candidate and remains forensic residue only.

## Successor requirement

A successor must preregister a lossless deterministic representation before
opening market/model/outcome data. The smallest defensible change is ordered
subcard chunking:

1. preserve the exact relation-unit order;
2. split it into chunks of at most 64 units;
3. bind each subcard to schedule, decision day, chunk ordinal, chunk count,
   prior hash, and complete relation-roster hash;
4. prohibit dropping, sampling, cap-raising after observation, or
   outcome-dependent chunking; and
5. rerun all inherited source gates from fresh roots under a new version.

This addresses the capacity defect without weakening D7's no-truncation or
anti-leakage boundary.
