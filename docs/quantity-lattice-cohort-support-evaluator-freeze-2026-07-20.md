# QLCD-288 source/support evaluator freeze

This work unit freezes the QLCD quantity transform, fail-closed source builder,
source-access sealer, support scheduler, and novelty evaluator before any real
QLCD cohort incidence is read.

## Bound preregistration

- artifact:
  `results/quantity_lattice_cohort_disagreement_preregistration_2026-07-20.json`;
- file SHA-256:
  `eb7920891cb2c9c5753f08e5a3ebfd3c3d39de28fdb2a245fc5b6d978c0f84d9`;
- canonical manifest hash:
  `9fd76b3dd9fd0d900689684c9d6b1d2c57ede9877eec73979b3ff11d29f59a16`;
- source incidence opened: `false`;
- outcomes opened: `false`.

## Frozen implementation

- exact-lattice transform: `preprocessing.quantity_lattice_cohort`;
- source builder: `training.build_binance_quantity_lattice_cohort`;
- hash-only source sealer:
  `training.freeze_quantity_lattice_cohort_disagreement_source_access`;
- support/novelty evaluator:
  `training.evaluate_quantity_lattice_cohort_disagreement_support`.

The builder verifies every official archive checksum against the prior audited
manifest, validates timestamps and aggregate/underlying ID contracts before
reindexing, materializes every 5m calendar slot, and applies six full source-gap
days plus the following 24 bars.  Only audit-proven zero-volume/zero-trade slots
are complete empty bars.  Raw ZIPs are streamed and not persisted.

The transform parses boolean maker flags fail-closed and requires each positive
quantity to lie on the exact milli-BTC lattice.  Coarse, medium, and fine event
counts and quantities partition each observed bar.  The evaluator independently
replays those accounting identities, the coarse share/coherence, fine signed
share, side, opposition, and final score.

## Disabled source access

Support execution is intentionally disabled while the source-seal, source, and
source-manifest hash constants are `None`.  After the full blinded build, the
only permitted evaluator edit is filling those three constants from a
`source_rows_parsed=0` hash-only seal.  Signal fields, q99.75 threshold,
eligibility, 24-hour scheduler, support gates, comparator registry, matching
logic, and output format may not change.

Outputs are deterministic write-once artifacts.  Clock gzip uses `mtime=0`.
Any support or novelty failure returns `REJECT_NO_REPAIR`; no economic row may
be opened unless the frozen decision is `PASS_SUPPORT`.
