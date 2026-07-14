# CLASP-24 evaluator freeze — 2026-07-14

This document defines the pre-outcome CLASP-24 evaluator boundary. The evaluator
source must be committed and hashed before it loads USD-M execution OHLC or
realized funding.

## Opened windows after freeze

Only these windows may be evaluated:

- train: `2020-01-01 <= t < 2023-01-01`
- select2023: `2023-01-01 <= t < 2024-01-01`
- select2023_h1: `2023-01-01 <= t < 2023-07-01`
- select2023_h2: `2023-07-01 <= t < 2024-01-01`

Calendar 2024, 2025, and 2026 YTD remain sealed unless the frozen pre-2024 gate
passes.

## Execution assumptions

- entry: next Binance USD-M five-minute open after the completed CLASP signal
  bar;
- exit: scheduled USD-M five-minute open exactly 24 completed bars after entry;
- leverage: `0.5x`;
- fees and slippage: `5 bp + 1 bp` per notional side, account cost `3 bp` per
  side at 0.5x leverage;
- realized funding: settlement times satisfying
  `entry_time <= funding_time <= exit_time`;
- strict MDD: entry cost first, then favorable path before adverse path, funding
  debits before adverse, funding credits do not raise intratrade peak, and the
  exit bar's later high/low is excluded;
- CAGR: full split wall-clock, including idle periods.

## Gate

Primary must pass train and select2023 with positive absolute return,
`CAGR / strict MDD >= 3`, strict MDD at most 15%, weekly-cluster sign-flip
one-sided p-value below 0.10, and mean gross underlying move above 12 bp.
Both 2023 halves must have positive absolute return and at least 65 trades.
Primary must also beat every score-bearing control on minimum train/select
CAGR-to-strict-MDD.
