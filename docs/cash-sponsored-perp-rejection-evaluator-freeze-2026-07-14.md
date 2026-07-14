# CSPR-12 evaluator contract freeze — 2026-07-14

## Status

**Evaluator design only; no CSPR return opened.** The evaluator source, tests,
and this document must be committed first. A second commit records the exact
evaluator source hash with `opened_windows: []`. The evaluator refuses to load
its price frame unless that freeze manifest is present and exact.

## Immutable execution and metrics

- primary quantile: `0.50`, selected by the outcome-blind support gate;
- primary opportunity clock: exact 850-row frozen CSV, replayed field-for-field;
- all control clocks: built globally through `2023-12-31` before split slicing;
- entry: next USD-M five-minute open after the completed signal bar;
- exit: USD-M open exactly 12 completed bars after entry;
- leverage: `0.5x`;
- fee/slippage: `5 bp + 1 bp` per notional side;
- account multiplier: `(1-0.0003)*(1+0.5*r)*(1-0.0003)`;
- CAGR: full split wall clock, including idle cash;
- strict MDD: entry cost, then favorable held extreme, then adverse held
  extreme, then scheduled exit; the exit bar's later high/low is excluded;
- weekly entry-cluster Rademacher test: 100,000 draws, seed `20260714`.

## Frozen control clocks, actions, and source quarantine

1. `direction_flip`: exact primary timestamps and holds; action is negative
   primary side; joint Spot/USD-M signal-bar quarantine.
2. `signal_delay_1bar`: primary mask and Spot-flow action shifted by one
   completed bar; source cleanliness is shifted with the originating signal;
   entry is still the next open after the delayed signal timestamp.
3. `no_centroid`: own globally non-overlapping clock; Spot taker-flow side;
   joint Spot/USD-M quarantine.
4. `no_perp_event_confirmation`: own clock; Spot taker-flow side; joint
   quarantine.
5. `spot_only`: own clock; Spot taker-flow side; Spot quarantine only. A USD-M
   feature outage cannot remove a valid Spot-only feature event, although the
   traded USD-M market grid must still exist.
6. `perp_only`: own clock; side is the sign of the completed USD-M price return;
   USD-M quarantine only. A Spot outage cannot remove it.
7. `role_swap`: own clock; action is the completed USD-M price-return direction
   (the exact opposite of Spot-flow side under the role-swap predicate); joint
   quarantine.
8. `spot_lag_1h`: own clock; Spot inputs/action stale by 12 bars, current USD-M
   inputs, USD-M current quarantine plus lagged Spot cleanliness embedded in
   the mask. Current Spot availability cannot alter this placebo.
9. `spot_lag_24h`: identical contract with Spot stale by 288 bars.

No control may alter its rule, hold, cost, or clock after returns open. Controls
are falsification tests and cannot replace a failed primary.

## Open and sealed windows

The first evaluator run may open only:

- train: `2020-01-01 <= t < 2023-01-01`;
- selection: `2023-01-01 <= t < 2024-01-01`;
- selection H1/H2 as the two fixed 2023 halves.

Calendar 2024 test, calendar 2025 eval, and 2026 YTD remain sealed. A trade is
included only when signal, entry, and exit timestamps all lie inside the split;
the global schedule is sliced, never rebuilt at a boundary.

## Qualification

The primary advances only if train and full 2023 each have positive absolute
return, CAGR/strict-MDD at least 3, strict MDD at most 15%, and weekly-cluster
one-sided `p < 0.10`; each 2023 half must be positive with at least 30 trades;
full 2023 must contain at least 80 trades. Its minimum train/full-2023
CAGR/strict-MDD must strictly exceed that of every frozen control above.
Failure rejects CSPR-12 without repair.
