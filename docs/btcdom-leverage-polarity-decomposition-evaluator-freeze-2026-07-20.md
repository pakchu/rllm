# DLPD-12 strict evaluator freeze — 2026-07-20

The DLPD-12 evaluator was frozen before any BTC execution OHLC or funding row
was parsed and before any execution-data byte was hashed.

## Frozen scope

- Phase one: `train=2022`, then `test=2023` only if train passes every gate.
- Phase two: `eval=2024-2025` and `final=2026H1` require a separate immutable
  signal-and-execution freeze after train and test pass, before any 2024+
  outcome row is opened.
- Entry: completed-hour decision followed by the next 5-minute open (`+5m`).
- Exit: exact 12-hour open-to-open hold.
- Exposure: `0.5x` equity.
- Costs: `6bp` per notional side base and `10bp` per notional side stress.
- Funding: exact settlement events and marks; exact entry/exit credits are
  dropped while debits are retained.
- strict MDD: global/pre-entry high-water mark, entry cost, funding marks, every
  held 5-minute favorable-then-adverse OHLC path, virtual adverse exit cost,
  and actual exit cost.
- CAGR: the full declared calendar, including warm-up and idle cash.

## Frozen controls

The source-only clocks remain diagnostic and cannot repair a failed primary:

- `btc_only_tail`
- `dom_only_mirror`
- `same_sign`
- `stale_btc_1h`
- `stale_dom_1h`

Three evaluator controls were frozen before outcomes:

- `direction_flip`
- deterministic random side from ASCII SHA-256
- `extra_latency_1h`

Primary counts are 237 in 2022 and 184 in 2023. The complete evaluation clock
contains 5,095 rows across all source and evaluator controls.

## Physical source isolation

The freeze only binds the already frozen parent manifests and their expected
SHA-256 identities. It does not hash the parent `.gz` data files.

After this freeze, phase-one source preparation may compare the precommitted
SHA-256 of each **compressed parent container**. That operation does not
decompress or expose numeric rows to the strategy. The slicer then converts
timestamps only, stops at the stage boundary, copies only the declared physical
window, validates its exact 5-minute/8-hour grids, and records zero parsed
post-stage numeric rows. Evaluation independently reconstructs the stage slice
from the frozen parent and rejects path redirection or altered slice bytes.

## Frozen identities

- evaluator source SHA-256:
  `748fae0511cfe3f3eca48f43627bc1a6b728253c225ee1fc6e2aba14f390b17c`
- evaluation clock SHA-256:
  `38ccc18df700d24462d0cae91e34733856ed053dc400c584a3eedaf3f9ed60f1`
- evaluator manifest hash:
  `c55b8b23f1ec45821ca670fab6f5811826b37d839c01d1e44d2a0ca1b30a31fd`
- evaluator JSON file SHA-256:
  `e96d299daa27ca598ce7ebe14a3f42eacc5a677703356c8ab193d8da1b8a371d`

At freeze time: opened windows `[]`; sealed windows
`[train, test, eval, final]`; parsed execution/funding rows `0/0`; simulation
run `false`.
