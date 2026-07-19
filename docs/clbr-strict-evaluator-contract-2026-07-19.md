# CLBR-24 strict evaluator contract — 2026-07-19

This document freezes the only allowed executable evaluation for the
outcome-blind `CLBR-24` clock. No threshold, stop, holding period, direction,
cost, bootstrap, or promotion gate may change after the evaluator freeze.

## Physical staging

- Freeze binds the committed preregistration, source builder, manifests, all
  execution-file hashes, evaluator bytes, and immutable configuration before a
  return is calculated.
- The source-only combined clock is physically split into train, test, and eval
  clock files during freeze.
- A stage loads only its own clock, USD-M `BTCUSDT` 5m OHLC, and funding file.
- Test cannot open unless the write-once train result passes. Eval cannot open
  unless the write-once train and test results both pass.
- Freeze, split clocks, and stage results use exclusive creation and cannot be
  overwritten.

## Ledger

- Calendar windows are `[start, end)` and CAGR counts the entire interval using
  `365.2425` days per year, including idle time.
- Entry is the exact frozen `entry_time` market open. Planned exit is the exact
  `planned_exit_time` market open. Held bars are
  `[entry_position, planned_exit_position)`.
- A long with `entry <= stop`, or short with `entry >= stop`, is invalid and is
  skipped before sizing.
- Quantity is fixed for the trade at
  `pre-entry equity * 1.0 leverage / entry price`.
- Base transaction cost is 6bp per side; stress cost is 12bp per side. Entry
  and exit fees use their respective notionals.
- A later open beyond the stop exits at that open before any post-exit high or
  low. Otherwise an intrabar stop assumes the favorable extreme occurs first,
  then fills at the stop. Its accounting timestamp is the next open minus one
  nanosecond so funding at the start of that bar is included but funding at the
  next bar is not.
- Funding is included exactly when `entry_time < funding_time <= exit_time`.
  Cash flow is
  `-direction * fixed_quantity * settlement_mark * funding_rate`, so a long
  pays positive funding. Funding exactly at a gap/time-exit boundary is applied
  before that boundary exit; a later millisecond-offset event is excluded.
- Strict MDD keeps one global high-water mark across trades. It includes entry
  and exit fees, funding cash at its settlement mark, and every held-bar path.
  Within each non-gap held bar, the favorable high/low is marked before the
  adverse low/high. Stop bars use the favorable extreme followed by the stop;
  gap-stop bars never use post-exit extremes.

Every stage reports absolute return, full-calendar CAGR, strict MDD,
CAGR/strict-MDD, executable and invalid trade counts, long/short counts, win
rate, exposure, costs, funding, and its trade ledger.

## Statistical contract

The one-sided significance statistic is a circular stationary block bootstrap
of net trade returns:

- centered null: subtract the observed mean before resampling;
- mean block length: 8 trades;
- resamples: 10,000;
- seed: `20260719`;
- p-value:
  `(1 + count(centered bootstrap mean >= observed mean)) / (B + 1)`.

## Promotion gates

| stage | positive return | ratio | strict MDD | executable trades | both sides | 12bp/side positive | bootstrap |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | yes | >= 2.0 | <= 15% | >= 30 | yes | not required | reported only |
| test | yes | >= 2.0 | <= 15% | >= 60 | yes | yes | p <= 0.10 |
| eval | yes | >= 3.0 | <= 15% | >= 60 | yes | yes | reported only |

Failure is a rejection, not an invitation to repair the candidate. The archive
ends in October 2024, so even a full pass promotes only to forward live shadow,
not production capital.
