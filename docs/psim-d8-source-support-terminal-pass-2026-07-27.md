# PSIM-D8 Source-Support Terminal Pass

Date: 2026-07-27 KST

Status: terminal source pass; source execution is permanently closed.

## Decision

The single authorized PSIM-D8 official source attempt passed all 13 frozen
source gates.

- result hash:
  `7104593f0c0aa32e9f1219ab075fa10261058b57460286eaddf3e6764626fba5`
- result artifact SHA-256:
  `0b92b476b654cd76f0cf9dc004690cbcb78e7a5e73917b5d66611c0460d00204`
- terminal action:
  `ACCEPT_PSIM_D8_SOURCE_SUPPORT_ONLY_NO_PROFITABILITY_CLAIM`
- first failed gate: none
- official source attempts: exactly one

This result closes the PSIM source-successor sequence. PSIM-D8 must not be
rerun, repaired, or modified, and no PSIM-D9 is authorized.

## Frozen source census

- 5,356 proposal events:
  - Ethereum: 4,985
  - Bitcoin: 371
- 11,280 historical proposal blobs decoded across the two independent
  replicas.
- Semantic decode errors: zero.
- 6,208 logical daily cards:
  - four archive schedules;
  - 1,552 decisions per schedule;
  - exactly one logical card per schedule and decision day.
- 6,308 deterministic relation subcards.
- Maximum relation units in any subcard: 64.
- Every logical card and relation-subcard manifest passed hash, chain,
  identity, ordering, and complete-roster validation.

The D7 cardinality failure was therefore resolved without dropping,
sampling, summarizing, or changing the logical-day denominator.

## Split support

The source-only split remains frozen:

- train: source years 2020-2021, 1,750 events;
- test: source year 2022, 1,352 events;
- eval: source year 2023, 1,159 events.

Each split passed the frozen annual, quarterly, unique-day, proposal,
counterpart, concentration, vocabulary, and relation-control support gates.

## Artifact authority

- events:
  `data/protocol_specification_intent_maturity_d8_events_2020_2023.jsonl.gz`
  - rows: 5,356
  - SHA-256:
    `d7308789176af4bfe1bb2f5f13c89d6811bc7f938f3ecec08b1bf8acc5f7e2b2`
  - canonical JSONL SHA-256:
    `b6f1e1733d423fd0fd88f7008d1e505d3a513c0d2bec692446c6e2cf32196ac0`
- logical daily cards:
  `data/protocol_specification_intent_maturity_d8_cards_2020_2024q1.jsonl.gz`
  - rows: 6,208
  - SHA-256:
    `ce1bd1bd9a24068e6e223efca323db805781e912eadb0d2a8b7d63610fab96c1`
  - canonical JSONL SHA-256:
    `cd73cd6f7f82a02b8662ef4689a721fa32698f73f37aebc1f1041dbfab3fb071`
- relation controls:
  `results/protocol_specification_intent_maturity_d8_source_controls_2026-07-27.json`
  - rows: 7
  - SHA-256:
    `6c24b5d6ea693e19a90972a31ae96a24ac28a1f1a6b20be63418d0b5881551b1`
  - control row hash:
    `d3c4f4868de128328aa36eda11764914fe3714fb57fd20ddb691822684f712ac`

## Access boundary

The official source run accessed only the preregistered Git remote identity,
commit metadata, proposal path incidence, and historical proposal blobs.

Gate 12 confirmed zero access or construction for:

- pre-2020 or post-2023 proposal blobs;
- BTC market, funding, or future-return rows;
- models or model outputs;
- rewards, trades, or PnL;
- CAGR or strict MDD.

`outcomes_opened` and `profitability_result` are both false.

## What this does not prove

PSIM-D8 is a validated causal source representation, not a profitable alpha.
It has no absolute return, CAGR, strict MDD, CAGR/strict-MDD ratio, trade
count, hit rate, or significance result.

Any model-visible subcard selection or aggregation, memorization audit,
training, market join, trade rule, or OOS economic evaluation requires a new,
separate preregistration that treats the frozen D8 source artifacts as
read-only input.
