# CDLTR-72A source evaluator v2 failure and v3 correction — 2026-07-21

## Disposition

The hash-frozen v2 evaluator was executed once and stopped before relay clock
construction.  Its strict uniqueness check rejected duplicate network
`available_at` timestamps.  This is an evaluator event-stream representation
defect, not a CDLTR source-support or novelty-gate result.

The already-frozen Coin Metrics source documentation states that early history
contains a large 2021 backfill.  Multiple historical daily observations can
therefore become causally available in one simultaneous publication batch.
There is no defensible temporal ordering among rows with the same
`available_at`; treating their CSV order as sequential reports would invent
causal order.

## Failed-run boundary

The v2 failed process loaded the same bound inputs as v1.  RRP and Cboe vote
frames were completed.  All 1,461 network source rows were transformed in
memory before the duplicate-availability validator stopped the process.
`build_clocks` was never reached.

- source value rows loaded from bound files: `4,468`;
- sanitized comparator rows loaded: `9,985`;
- RRP source-vote rows derived in memory: `1,498`;
- Cboe source-vote rows derived in memory: `1,508`;
- raw network source-vote rows derived in memory: `1,461`;
- CDLTR candidate/event rows derived: `0`;
- support, control-support, or novelty verdicts produced: `0`;
- BTC price, funding, return, PnL, equity, CAGR, or MDD rows read: `0`;
- economic simulations run: `0`;
- output report or clock files written: `0`.

The failed process printed no source values, source sides, comparator events,
or event incidence.  Its traceback exposed only that duplicate network
availability timestamps exist.

## Exact v3 correction

Version 3 changes only simultaneous network-publication batching:

1. derive each observation's frozen 7-calendar-day vote exactly as before;
2. sort causally by `available_at`, then by `observation_date`;
3. for rows sharing one exact `available_at`, retain only the row with the
   latest `observation_date` as the report visible at that timestamp;
4. never sequence older rows inside the same batch; and
5. continue using the resulting unique report timestamps in the unchanged
   first-report relay state machine.

Selecting the latest observation is an as-of rule: at one simultaneous release
time it is the most current state and causally supersedes older backfilled
observations.  The rule is deterministic, uses no BTC outcome or event result,
and is identical to ordinary live operation when one observation arrives per
timestamp.

No source value, feature formula, side rule, lookback, expiry, relay order,
deadline, latency, hold, exposure, split, support gate, control, comparator, or
novelty threshold changes.  A v3 evaluator commit and v3 freeze must precede
the next run.  The v1 and v2 freeze artifacts remain immutable audit records.
