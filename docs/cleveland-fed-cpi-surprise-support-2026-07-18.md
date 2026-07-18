# CFCS-1 source-support freeze — 2026-07-18

## Decision

**FREEZE_EVALUATOR**

No BTC OHLC, funding, return, or existing-alpha overlap source was opened.
The exact primary and control clocks are now immutable.

## Primary source distribution

| Window | Events | Long | Short |
|---|---:|---:|---:|
| 2019_source_history | 6 | 3 | 3 |
| 2020 | 9 | 5 | 4 |
| 2021 | 8 | 2 | 6 |
| 2022 | 9 | 3 | 6 |
| stage1 | 26 | 10 | 16 |
| 2023_h1 | 4 | 4 | 0 |
| 2023_h2 | 4 | 4 | 0 |
| 2023 | 8 | 8 | 0 |

## Checks

| Check | Result |
|---|:---:|
| `source_hashes_match` | PASS |
| `source_rows_exactly_60` | PASS |
| `all_controls_present` | PASS |
| `all_control_clocks_valid` | PASS |
| `primary_clock_replays` | PASS |
| `primary_source_counts_replay` | PASS |
| `stage1_events` | PASS |
| `each_stage1_year` | PASS |
| `stage1_direction_support` | PASS |
| `sealed_2023_events` | PASS |
| `each_sealed_2023_half` | PASS |
| `month_concentration` | PASS |
| `direction_flip_is_exact` | PASS |
| `market_or_funding_rows_opened_zero` | PASS |

## Frozen controls

- headline-only, core-only, and no-concordance mechanism controls;
- exact direction flip;
- one-calendar-day delay and seven-calendar-day placebo;
- all entries use the same 08:35–16:00 America/New_York wall-clock window.

## Identity

- preregistration SHA-256: `9c252a988885c7fa1975b6f7190af4efeab50ee8541a67c0bb8f8882a3fa3e0d`
- clock ledger SHA-256: `d1223d848a47de64224f9b21429287cbec51052e52c4184b4f10efe79d361511`
- support manifest: `c95bab576f9e134df3960893adcf40a89938861f138e582439f2607871adfec0`
- market/funding rows opened: `0`
