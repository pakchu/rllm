# LAMB-21 source-support retirement

Date: 2026-07-25

## Decision

**RETIRE LAMB-21 unchanged before reward construction, model training, or
market-outcome access.**

The clean committed runner was:

```text
commit  bf98bb458188e3963c224974bc05871190c64189
runner  training/build_lamb21_source_support.py
SHA256  769934e271dfbfb48f95bc7be145e6c60805a04cfdffc1c39aec3d9b738dcd9a
tests   tests/test_build_lamb21_source_support.py
SHA256  07e48c713ab27bdd28ff27e746b8133c54ce34087b02d4abc33dda79046329e4
```

Its clean protocol guard passed. The official source-only run then stopped at
the first frozen gate:

```text
gate             gate_01
check            cascade_transaction_clock
decision         fail
failure action   retire_lamb21_unchanged_before_rewards
```

Machine-readable evidence:
[`lamb21_source_support_rejection_2026-07-25.json`](../results/lamb21_source_support_rejection_2026-07-25.json).

## Exact failure

The frozen Source-D clock rule is:

```text
date_ms <= first_transact_time_ms <= last_transact_time_ms
        < date_ms + 300000
```

The exact frozen cascade source contains **15,295 observed rows** whose
`last_transact_time_ms` equals the exclusive five-minute bar end. None is
later than the end; equality alone violates the preregistered strict
inequality.

| UTC year | Physical rows | Violations |
|---|---:|---:|
| 2020 | 105,408 | 1,659 |
| 2021 | 105,120 | 5,461 |
| 2022 | 105,120 | 4,629 |
| 2023 | 105,120 | 3,546 |
| **Total** | **420,768** | **15,295** |

The first violation is the bar labeled `2020-01-01T03:50:00Z`; the last is
`2023-12-31T23:35:00Z`. The failure spans every source year and is not a
rounding edge in one isolated interval.

## Fail-closed result

The runner stopped before building a joint state:

```text
source value rows decoded   843,347
joint state rows built            0
```

It did not write either frozen support output:

```text
data/lamb21_source_support/token_support.csv.gz
results/lamb21_source_support_2026-07-25.json
```

Every forbidden counter remained zero:

```text
execution-market rows opened   0
funding rows opened            0
future-return rows opened      0
reward rows built              0
model rows built               0
trades built                   0
PnL / CAGR / MDD values        0
post-2023 source rows opened   0
```

LAMB-21 therefore has no profitability, model, trade, CAGR, or MDD result.

## No repair under this identity

Do not change `< bar_end` to `<= bar_end`, subtract one millisecond, relabel
the physical rows, drop the 15,295 rows, or reinterpret them after seeing this
failure. Each would alter the frozen source boundary with knowledge of its
incidence.

Stage 0.5 and every economic/RLLM stage are permanently unauthorized for
`LAMB-21`. A successor may use a separately justified event-clock definition,
but it requires a new identity, boundary, preregistration, and clean
source-support run.
