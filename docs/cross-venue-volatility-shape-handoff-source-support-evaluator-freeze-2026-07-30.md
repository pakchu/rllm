# CVVH-432 source-support evaluator freeze — 2026-07-30

## Boundary

This evaluator is frozen before the first Binance BTCBVOL or Deribit DVOL data
row is decoded.  Synthetic tests may exercise identical schemas, but canonical
source values, candidate incidence, comparator rows, Gross9 clocks, BTC
execution prices, funding, returns, and PnL remain unopened.

The canonical preregistration is bound by SHA-256
`9e6901e67e36d6d9170dae54548419cbfb9178cf84338e3c5652012896ee6604`
and manifest
`dd3e334939b24cb5508c66ed9787c2c3e1a0ad006dcadcbfb673e541d6520cad`.

## One authoritative attempt

The command first proves a clean branch, exact canonical remote SHA, committed
evaluator closure, preregistration identity, and exact source hashes/headers.
Every path in the preregistration's original protocol seal must still have its
exact registered Git blob and SHA-256; a later committed mechanism change is
therefore terminal rather than silently rebound by the evaluator commit.
The extended evaluator closure also seals `training/__init__.py` and
`tests/conftest.py`, so import bootstrap behavior and pytest outcome rewriting
cannot change outside the reported evaluator-closure hash.
All configured source, preregistration, claim, output, and failure paths must
equal the frozen canonical paths. The claim also binds the exact source grid,
mechanism cutoff, repository commit, and evaluator-closure hash.
The hash pass retains the exact compressed BVOL and DVOL bytes in memory.
After the claim, value rows are decompressed from those retained immutable
bytes; source paths are never reopened. A path replacement or in-place write
between hash and decode therefore cannot change evaluated bytes.
It then atomically writes
`results/cross_venue_volatility_shape_handoff_source_support_attempt_claim_2026-07-30.json`.
Only after that durable claim exists may a source loader decode the first
value row.

The claim permits one authoritative attempt.  A crash, exception, failed gate,
or preexisting claim forbids retry, resume, fallback, threshold repair, or
verification replay.  Exceptions produce a write-once terminal failure receipt
when the process remains able to write.

## Exact source validation

- BVOL must be the exact 26,568-hour grid starting 2023-06-20.  Completed-hour
  availability and trade-earliest time equal candle-open time plus one hour.
  `source_complete` is equivalent to 3,600 source seconds.  Valid rows have
  exact positive OHLC and reason `ok`; invalid retained rows have blank OHLC
  and a non-`ok` reason.
- DVOL must contain the exact 26,569 inclusive hourly opening timestamps from
  2023-06-20 through 2026-07-01, with `close_time=date+1h` and exact positive
  OHLC. Raw rows are validated before the frozen `close_time <
  2026-07-01T00:00:00Z` join filter.
- Join is exact on BVOL availability and DVOL close time.  There is no fill,
  imputation, tolerance, clipping, or nearest match.
- Mechanism rows stop at the preregistered economic horizon
  `2026-06-01T00:00:00Z`; pre-source June 2023 remains full-calendar idleness.
- Both producers serialize timezone-naive timestamps as UTC. The parser pins
  those exact naive tokens to UTC; any explicit nonzero offset is terminal.

## Frozen evidence

The evaluator builds independently reserved primary, Deribit-led,
body-lead-only, range-lead-only, and stale-Deribit clocks, plus all five
accepted-parent controls.  It evaluates the preregistered support floors,
side/month composition, 90-day gap, 12-event same-side run, exact selection
prefix append invariance, and structural-control distinctness.

The structural 24-hour match is deterministic one-to-one: maximum cardinality,
minimum exact absolute lag, then lexicographically smallest timestamp pairs.
Lag arithmetic is exact integer microseconds, and the pair list is reconstructed
against the optimum suffix objective rather than an implementation-order
tie-break. All gates use exact rational comparisons.

Append invariance is not a duplicate prefix calculation. The evaluator first
builds states, raw candidates, and accepted clocks from the complete source
input and projects the result to signal times before the selection cutoff. It
then physically removes later source rows, rebuilds, and requires byte-identical
validity, states, IDs, sides, and entry/exit clocks.

Clock files and `report.json` are staged under one temporary directory and
fsynced before Linux `renameat2(RENAME_NOREPLACE)` atomically publishes the
directory; an existing or racing destination cannot be replaced. The parent
directory is fsynced after publication. The no-replace rename is the commit
point: a later parent-fsync error is terminal and reported to the process, but
cannot create a contradictory failure receipt beside the already published
bundle. Gzip output has fixed metadata, controls use frozen order, and events
use the canonical scheduling order.
The bundle contains no feature values or outcomes, only canonical clock fields
and support evidence.

Any failed support check retires exact `CVVH-432` before novelty.  No control
can replace the primary.
