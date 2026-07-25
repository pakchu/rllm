# TRACER-4H source-support retirement

Date: 2026-07-25

## Decision

**RETIRE TRACER-4H unchanged before reward construction or market-outcome
access.**

The committed official runner was:

```bash
PYTHONPATH=. .venv/bin/python \
  -m training.build_tracer4_tri_surface_relational_executor_support
```

It completed normally at commit
`24f7fe5015d89ff7e02593cf01c2943626ecc96d`, but the frozen conjunctive
source-support decision was:

```text
support_pass = false
decision = retire_tracer4_unchanged_before_outcomes
first_failure = gate_03_2023_core_valid_min
```

Machine-readable retirement evidence:
[`tracer4_tri_surface_relational_executor_source_retirement_2026-07-25.json`](../results/tracer4_tri_surface_relational_executor_source_retirement_2026-07-25.json).

## Failed frozen gates

Only four checks failed:

```text
gate_03_2023_core_valid_min
gate_04_2023Q3_quarter_ready_min
gate_04_2023_sequence_ready_min
gate_05_2023_source_invalid_max
```

| Year | Nominal boundaries | Core valid | Core-valid share | Source invalid | Invalid share | Sequence ready |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 2,195 | 2,175 | 99.09% | 20 | 0.91% | 1,793 |
| 2021 | 2,190 | 2,144 | 97.90% | 46 | 2.10% | 2,118 |
| 2022 | 2,190 | 2,166 | 98.90% | 24 | 1.10% | 2,134 |
| 2023 | 2,190 | 1,997 | **91.19%** | 193 | **8.81%** | **1,805** |

The frozen requirements were at least 95% core-valid, at most 5% invalid, and
at least 2,000 sequence-ready boundaries in 2023. All three annual constraints
therefore fail materially rather than by rounding.

The quarterly concentration is also decisive:

| 2023 quarter | Nominal | Core valid | Sequence ready | Required sequence ready |
|---|---:|---:|---:|---:|
| Q1 | 540 | 520 | 502 | 450 |
| Q2 | 546 | 530 | 502 | 450 |
| Q3 | 552 | 429 | **325** | 450 |
| Q4 | 552 | 518 | 476 | 450 |

## Source-only diagnosis

The physical join itself was not the problem. In 2023, exact join shares were
100% for leadership and premium, and 99.997% for aggregate trades. The
source-only boundary rebuild instead found these invalid-reason incidences:

```text
leadership validity flag / nonfinite projected value  185 boundaries
premium validity flag / nonfinite OHLC                  8 boundaries
aggregate-trade grid / row-count gap                    1 boundary
```

Reason incidences can overlap; there were 193 unique invalid 2023 boundaries.
Q3 alone contained 123 leadership-invalid boundary incidences. The longest
continuous 2023 invalid run covered 13 four-hour boundaries from
`2023-08-12T00:00:00Z` through `2023-08-14T00:00:00Z`.

This is exactly what the preregistered validity gates were designed to reject:
the dense four-hour language cannot be formed reliably enough in the frozen
2023 source stream.

## Integrity checks that passed

- Exact source hashes, physical headers, projected headers, clocks, cut hashes,
  deterministic gzip metadata, and zero forbidden-column counters reproduced.
- All four append-prefix rebuilds were byte-identical.
- All three controlled token streams differed from the primary.
- Category support, directional incidence, signature diversity, and every
  adjacent-year JSD check passed.
- The source-cut manifest, token table, and support report are write-once
  artifacts.

Evidence hashes:

```text
source-cut manifest
  a2b3996af789e79ebc4ae56496e3cd050f25016a951fc172b13afa11f1f2b2ed
token support
  b8f0c59ca5f41f9bda7b5b09fc143b55f3ad1a89f5b1835554e6e9fb868b43db
support report
  b67ee0402dfb19d16d9a1bcb6d4134d0674fd37074a5c5f39bc84dbad9dbcc81
support-report manifest hash
  2c8cc538359c86c7ec11a07859614f09bf151085aecebb4811bdf8cb19490919
```

## Outcome boundary

Every forbidden counter remained zero:

```text
execution-market rows opened       0
funding rows opened                0
future-return rows opened          0
reward rows built                  0
model rows built                   0
PnL / CAGR / MDD values computed   0
post-2023 numeric source rows      0
```

TRACER therefore has no profitability result. This is a source-support
retirement, not a negative backtest.

## No repair

Do not drop invalid boundaries, relax the 95%/5%/quarterly gates, fill invalid
leadership rows, change the clock, change the source, alter the rank history,
or simplify the token language under the `TRACER-4H` identity. Those changes
would be informed by the failed support evidence.

Stage 0.5, reward construction, Gemma training, and market/funding evaluation
are permanently unauthorized for this identity. A successor must have a new
mechanism, source boundary, name, and preregistration.
