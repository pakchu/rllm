# H8DM-1 source-support freeze — 2026-07-18

## Decision

**FREEZE_EVALUATOR**

No BTC OHLC, funding, return, or existing-alpha overlap source was opened.
The primary and control clocks are now immutable.

## Primary distribution

| Window | Events | Long | Short |
|---|---:|---:|---:|
| 2017_2019_source_history | 0 | 0 | 0 |
| 2020 | 28 | 16 | 12 |
| 2021 | 19 | 10 | 9 |
| 2022 | 28 | 2 | 26 |
| stage1 | 75 | 28 | 47 |
| 2023_h1 | 8 | 4 | 4 |
| 2023_h2 | 16 | 10 | 6 |
| 2023 | 24 | 14 | 10 |

## Checks

| Check | Result |
|---|:---:|
| `source_hashes_match` | PASS |
| `source_rows_exactly_365` | PASS |
| `selected_tail_q50_replays` | PASS |
| `all_controls_present` | PASS |
| `all_control_clocks_valid` | PASS |
| `primary_clock_replays` | PASS |
| `primary_source_counts_replay` | PASS |
| `stage1_events` | PASS |
| `each_stage1_year` | PASS |
| `stage1_direction_support` | PASS |
| `stage1_month_concentration` | PASS |
| `sealed_2023_events` | PASS |
| `sealed_2023_direction_support` | PASS |
| `each_sealed_2023_half` | PASS |
| `sealed_2023_month_concentration` | PASS |
| `structural_break_releases_excluded` | PASS |
| `direction_flip_is_exact` | PASS |
| `market_or_funding_rows_opened_zero` | PASS |

## Frozen control clocks

| Control | Events |
|---|---:|
| `primary` | 99 |
| `migration_only` | 100 |
| `borrowings_only` | 96 |
| `cash_only` | 104 |
| `no_agreement` | 104 |
| `nsa_primary` | 95 |
| `direction_flip` | 99 |
| `one_week_delay` | 99 |
| `four_week_placebo` | 99 |

The mechanism controls are each component alone, the composite without the
two-of-three agreement rule, and the exact not-seasonally-adjusted replay. The
falsification controls are an exact direction flip, one-week delay, and
four-week placebo.

## Identity

- preregistration SHA-256: `0705042e6fceb5e183e5967be846bc106ea860f642fd44cba72dfa214eb09432`
- clock ledger SHA-256: `1d1774123480868e36b8f76e28ee11c918f18d6694ce8f16056f338710f040ba`
- support manifest: `591f9acd889735cf638d73b1d905be8438eb6c066fae8aa082e73c7988047c42`
- market/funding rows opened: `0`
