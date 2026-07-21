# GNRC source-support protocol

## Purpose

This stage decides only whether the frozen 24-member GNRC family has enough
source incidence and temporal dispersion to justify opening BTC outcomes. The
family, feature formulas, thresholds, holds, scheduler, and support gates were
committed before any GDELT count artifact was opened.

Frozen prerequisites:

- candidate-family commit: `4da9604`
- write-once preregistration commit: `e2c0b95`
- preregistration artifact SHA-256:
  `ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3`
- original source transport commit: `22af253848939855aa456b2e2a5dda02e01a84c5`
- sparse-safe v2 transport commit: `ec3749c`
- v2 transport-amendment artifact SHA-256:
  `9244fc5ab203abe1866a1960c9b652ec725a8e37a1196ea5e784c742d1bc9f18`

## Inputs

The evaluator accepts only the frozen source artifacts:

- `results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json`
- `data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz`
- `data/gdelt_bitcoin_narrative_timeline_raw_2020_2023.jsonl.gz`
- `results/gdelt_gnrc_source_access_seal_2026-07-22.json`

Before importing mutable policy/source modules, the evaluator verifies the
committed preregistration artifact, its exact policy source/document, the v2
transport amendment, and both v1/v2 source builders. After acquisition and
before parsing daily counts, a separately committed outer source-access seal
must bind the manifest, daily/raw artifacts, this evaluator, and this protocol.
A later launcher hard-codes that seal's SHA-256 before it imports the evaluator.

The evaluator validates the v2 manifest hash, transport contract,
source-builder hashes, four request hashes, both output hashes, exact CSV
schema, all 1,461 sorted unique UTC dates over
`[2020-01-01, 2024-01-01)`, the `+48h15m` availability clock, count subset
consistency, and the two frozen all-zero global outage dates. The raw response
bundle is hash-checked but not parsed.

## Frozen evaluation

For each exact preregistered variant, source scores are computed for every
split-implied source date. The executable scheduler then produces train
(2021–2022) and selection (2023) support statistics. The evaluator applies all
per-variant year, half, side, activity, and month-concentration gates followed
by the family gate:

- at least eight fully passing variants;
- every score archetype represented;
- both `(7,28)` and `(14,56)` windows represented.

The evaluator cannot change signs, thresholds, windows, holds, evidence gates,
or support gates. If the family gate fails, the decision is
`retire_without_repair`. If it passes, the decision is `advance_to_market`.

## Outcome boundary

This stage reads no BTC candle, funding, future-return, PnL, CAGR, MDD, or OOS
2024+ news value. Its write-once output is:

`results/gdelt_narrative_rotation_clearing_source_support_2026-07-20.json`

Only a committed `advance_to_market` result permits construction of the frozen
economic evaluator. A failed family cannot be repaired on the same source
artifact.
