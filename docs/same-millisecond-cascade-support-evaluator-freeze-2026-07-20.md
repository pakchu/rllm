# SMCC-144 source-support evaluator freeze

This work unit freezes `training.evaluate_same_millisecond_cascade_support`
before the real same-millisecond source is rebuilt or its incidence is read.
No post-signal market price, funding, return, excursion, PnL, equity, CAGR, or
MDD is accessible to this evaluator.

## Bound preregistration

- path: `results/same_millisecond_cascade_preregistration_2026-07-20.json`
- file SHA-256:
  `49fb04bc666f56f00efebf1f03c08e4a386f69b12fb12feeac22f9fde4ff9111`
- canonical manifest hash:
  `5628bdf5e9f6079ebb14585886738c9176ae869e249bcad89927775c7dafa302`
- source incidence opened: `false`
- outcomes opened: `false`

## Fail-closed checks

The evaluator verifies the source artifact and manifest hashes, complete UTC
five-minute grid, audited archive/audit hashes, source flags, 24-bar post-gap
quarantine, source-gap day set, and algebraic identities for group share,
coherence, side, and score.  The q99.5 threshold masks invalid rows, shifts one
row, and then uses the frozen 8,640/2,016 calendar-row rolling contract.

Only the decision bar and strictly prior baseline must be source-complete.
Future source gaps cannot cancel a selected event.  Scheduling is deterministic
`t+2` entry, 144-bar hold, chronological non-overlap, and pre-2024 containment.

Every sparse comparator path, hash, member, coverage, and time column comes
from the preregistration registry.  Missing files, changed hashes, malformed
members, duplicate entries, or invalid times produce a deterministic novelty
failure.  Exact and +/-12-bar matching use the preregistered nearest-unused
one-to-one algorithm.  Dense BAFR is exact-match report-only.

The support clock and JSON are byte-stable write-once artifacts.  A support or
novelty failure emits `REJECT_NO_REPAIR`; it does not trigger a threshold,
direction, hold, delay, support-floor, or comparator repair.

Execution remains disabled until a hash-only source-access seal binds the built
source bytes and source-manifest bytes.  After the build, the only permitted
evaluator edit is populating the seal/source hash constants; signal, scheduling,
support, novelty, or artifact logic may not change.

## Pre-access source amendment

The first source build failed closed on one exact duplicate underlying-trade
range in the hash-bound `2020-01-15` archive before support incidence was read.
The preregistration and its two evaluator hash constants were amended only to
add that full UTC day to source quarantine.  Policy, threshold, scheduler,
support gates, novelty gates, and output logic are unchanged.  The hashes above
are the post-amendment bindings.
