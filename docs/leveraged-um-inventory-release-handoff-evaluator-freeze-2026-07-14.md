# LURI-48 evaluator contract freeze — 2026-07-14

## Scope and sequencing

This document fixes the only permitted pre-2024 return evaluation for
**LURI-48 — Leveraged USD-M Inventory Release Handoff**. The evaluator source,
tests, and this contract must be committed before its source hash can be
written to the pre-outcome freeze manifest. No price, held path, realized
funding, or LURI return may be parsed before that manifest verifies.

Execution order is mandatory:

1. verify every preregistration, support, clock, feature, market-manifest, and
   funding-manifest hash;
2. load only the causal feature frame ending in 2023;
3. replay the exact selected `0.40` primary clock and all frozen control clocks;
4. globally reserve each control's own non-overlapping 48-bar schedule;
5. verify all raw and scheduled counts against the support freeze;
6. only then parse the pre-2024 USD-M OHLC and realized-funding files;
7. slice the already-reserved global clocks into the fixed evaluation windows;
8. evaluate once and accept or reject without repair.

## Immutable inputs

- primary clock: 432 events, SHA-256
  `50765cfed0c3ec6a0d1df18857c4e0a3e574d1aa449538c9b89cfac1fff67095`;
- feature source: 419,855 available rows ending `2023-12-31 23:55:00`,
  SHA-256
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`;
- USD-M OHLC: 420,768 five-minute rows, SHA-256
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`;
- realized funding: 4,383 exact BTCUSDT settlements, SHA-256
  `c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7`.

The primary policy, exact direction flip, one-bar delayed falsification, and
all nine score-bearing controls are fixed by the preregistration. Direction
flip uses the exact primary timestamps with negated side. Every other control
uses its own globally reserved clock. No split-local rescheduling is allowed.

## Fixed windows

| Window | Inclusive start | Exclusive end |
|---|---|---|
| train | 2020-01-01 | 2023-01-01 |
| select2023 | 2023-01-01 | 2024-01-01 |
| select2023 H1 | 2023-01-01 | 2023-07-01 |
| select2023 H2 | 2023-07-01 | 2024-01-01 |

A trade belongs to a window only when its signal, entry, and exit are all
inside that window. Boundary-crossing trades are excluded rather than rebuilt.
Calendar 2024, 2025, and 2026 remains sealed.

## Execution and account return

- signal: completed five-minute bar `t`;
- entry: USD-M open at `t+1`;
- exit: USD-M open exactly 48 bars after entry;
- leverage: `0.5x`;
- fee: `5 bp` and slippage: `1 bp` per notional side;
- account execution cost: `(0.0005 + 0.0001) * 0.5 = 0.0003` per side;
- gross underlying move:
  `r = side * (exit_open / entry_open - 1)`;
- realized funding events: every frozen settlement satisfying
  `entry_time <= funding_time <= exit_time`;
- funding factor:
  `product(1 - 0.5 * side * funding_rate)`;
- final trade multiplier:
  `(1-0.0003) * (1+0.5*r) * funding_factor * (1-0.0003)`.

Funding uses exact returned settlement timestamps, not an assumed cadence.
Mean gross underlying move is calculated directly from `r`, before costs and
funding, rather than reverse-engineered from net account returns.

## Strict MDD order

The scheduled exit bar's high and low are excluded. For the held bars from
entry through the bar immediately before exit:

1. entry cost is debited;
2. the favorable held extreme is applied first and may set the intratrade peak;
3. all funding factors below one are conservatively applied before the adverse
   held extreme;
4. funding credits never establish an intratrade peak and never offset those
   debit factors for the adverse-path MDD mark;
5. the adverse held extreme is marked against that peak;
6. the scheduled-open price return, full realized-funding factor, and exit cost
   determine ending equity;
7. realized ending equity may establish the next intertrade peak.

This convention is intentionally more adverse than an unknown OHLC/funding
intrabar ordering and prevents funding receipts from hiding held-path risk.

## Statistics and gate

- absolute return is compounded account return;
- CAGR uses the entire wall-clock split, including idle periods;
- strict MDD includes entry cost, held-path excursion, funding debit, and exit
  cost;
- weekly entry-cluster Rademacher sign-flip test: 100,000 draws, seed
  `20260714`, one-sided;
- trade statistics are computed from funding- and cost-adjusted trade returns;
- report long/short counts, settlements crossed, and mean gross underlying bp.

LURI advances only when all conditions hold:

1. train and full 2023 each have positive absolute return;
2. train and full 2023 each have `CAGR / strict MDD >= 3`;
3. train and full 2023 each have `strict MDD <= 15%`;
4. train and full 2023 each have weekly-cluster `p < 0.10`;
5. train and full 2023 each have mean gross underlying move strictly above
   `12 bp`;
6. both 2023 halves are positive and contain at least 45 trades;
7. the primary's minimum train/full-2023 ratio strictly exceeds the same
   minimum ratio for every score-bearing control.

Direction flip and one-bar delay remain reported falsifications but are not
selection competitors. Failure rejects LURI-48 v1. No threshold, side, hold,
control, funding endpoint, cost, MDD convention, or gate may be changed after
the outcome is opened.
