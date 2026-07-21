# CDLTR-72A source evaluator v1 failure and v2 correction — 2026-07-21

## Disposition

The hash-frozen v1 source-only evaluator was executed once after its freeze and
stopped before relay incidence or any economic outcome calculation.  This was
an evaluator input-representation defect, not a source-support or novelty-gate
failure.

The v1 evaluator expected every `observation_date` value to be serialized as
`YYYY-MM-DD`.  The already-bound Coin Metrics builder and manifest serialize
that field as the canonical midnight form `YYYY-MM-DD 00:00:00`.  The mismatch
was visible before incidence in:

- `training/download_coinmetrics_btc_network_daily.py`, which writes
  `_format_dt(observed)`; and
- `results/coinmetrics_btc_network_daily_pre2024_manifest_2026-07-16.json`,
  whose frozen row range uses midnight timestamps.

## Failed-run boundary

The failed process loaded the three bound source tables and the sanitized
comparator table, then derived the RRP and Cboe source votes in memory.  It
failed while parsing the network observation-date representation, before
`build_clocks` was reached.

- source value rows loaded from bound files: `4,468`;
- sanitized comparator rows loaded: `9,985`;
- RRP source-vote rows derived in memory: `1,498`;
- Cboe source-vote rows derived in memory: `1,508`;
- network source-vote rows derived: `0`;
- CDLTR candidate/event rows derived: `0`;
- support, control-support, or novelty verdicts produced: `0`;
- BTC price, funding, return, PnL, equity, CAGR, or MDD rows read: `0`;
- economic simulations run: `0`;
- output report or clock files written: `0`.

No source value, side incidence, or comparator event was printed or persisted
by the failed process.  The traceback exposed only the representation suffix
`00:00:00` at row position zero.

## Exact v2 correction

Version 2 changes only date representation normalization:

1. accept exact `YYYY-MM-DD` or exact `YYYY-MM-DD 00:00:00`;
2. require the timestamp form to be midnight, with no other time accepted;
3. validate the first ten characters as a real Gregorian calendar date; and
4. reduce both accepted forms to the same `datetime.date` value before any
   source feature or relay logic runs.

No source, value, threshold, side rule, lookback, availability timestamp,
expiry, relay order, latency, hold, exposure, split, support gate, control,
comparator, or novelty definition changes.  CDLTR-72A may be rerun only after
the v2 evaluator implementation is committed and a new v2 evaluator-freeze
artifact is committed.  The v1 freeze remains immutable as the audit record of
the failed implementation.
