# IFDA-72 source rejection — 2026-07-20

## Decision

**Reject IFDA-72 without repair before support clocks or market outcomes.** The
official Binance USD-M `BTCUSDT-trades-2020-01-02.zip` archive violates the
precommitted exact-`+1` individual trade-ID continuity gate. The failure is in
the checksum-verified source itself, not in an economic result.

- decision: `REJECT_NO_REPAIR`
- source rejection result:
  `results/individual_fill_dispersion_absorption_source_rejection_2026-07-20.json`
- result hash:
  `e52de36347ac8d57bfa05462d3dca4d19e127e9f859e583fe115650ff70a1ebf`
- result file SHA-256:
  `96a16d5f8f12549433fb1c0b9baed6a399a82eb1740e6906d57db4f02acbb6b1`

## Exact frozen failure

The committed source builder verified every published `.CHECKSUM` before
parsing. The first archive day passed. The second did not:

| UTC day | raw rows | first ID | last ID | contiguous cardinality | non-`+1` transitions | missing IDs |
|---|---:|---:|---:|---:|---:|---:|
| 2020-01-01 | 101,871 | 25,247,504 | 25,349,374 | 101,871 | 0 | 0 |
| 2020-01-02 | 224,747 | 25,349,375 | 25,574,220 | 224,846 | 45 | **99** |

The largest observed transition delta was eight. Early examples include
`25487817 -> 25487819`, `25487825 -> 25487827`, and
`25487828 -> 25487834`.

The independent aggregate source corroborates that this is not a CSV header or
row-order parser artifact:

- aggregate-event IDs themselves are exact `+1` continuous;
- aggregate underlying ranges have 15 between-span gaps containing 39 IDs;
- another 60 missing raw IDs lie inside aggregate underlying spans; and
- `39 + 60 = 99`, matching the missing raw-ID count exactly.

Checksums for the failing day were:

- `trades`: `15bcd04fdd1b286a9ca4608b3452ce96bad0192483f5da627c73fc5b9aa2f464`
- `aggTrades`: `d7aa5d060eb4530dffa7609211881bb7caeef9d143b9e1ead2a459ca8e583ba9`

## Why this cannot be repaired

The frozen preregistration requires individual trade IDs to be strictly
increasing, unique, and exact `+1` continuous across rows and archive
boundaries. It also states that a source failure retires the singleton without
repair. Relaxing the continuity gate, dropping 2020-01-02, starting later,
imputing missing rows, or switching source representation after observing this
failure would be post-source policy selection.

No threshold, direction, hold, scheduler, support gate, or aggregate-control
change is authorized. IFDA-72 is permanently excluded from alpha and portfolio
claims.

## Opened and closed boundaries

Only source data for 2020-01-01 and 2020-01-02 was opened. A one-day feature
smoke was inspected before the full build; therefore partial source incidence
is disclosed as opened. The full build stopped before writing a monthly source
file or manifest.

No candidate clock, control-clock bundle, support count, novelty statistic,
Binance market/funding row, future return, PnL, CAGR, or strict MDD was loaded or
calculated. Raw ZIP bytes were never persisted. Transient one-day smoke files
were deleted after the audit.

The next candidate must use a genuinely different observable and mechanism; it
may not repair or reuse the IFDA clock.
