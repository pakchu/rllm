# CATCH-12 evaluator contract freeze — 2026-07-14

## Status

**Evaluator design only; no CATCH return opened.** The evaluator, tests, and
this contract are committed before any USD-M execution price or held high/low
is loaded. A separate pre-outcome manifest records the committed evaluator
hash with `opened_windows: []`; the evaluator refuses to load outcomes unless
that manifest matches exactly.

## Immutable execution and metrics

- selected strictly-prior setup-score quantile: `0.975`;
- primary clock: exact frozen 3,957-row CSV, replayed field-for-field;
- every control clock: built globally through 2023 before any return is loaded
  or any split is sliced;
- next Binance USD-M five-minute open entry;
- fixed 12-bar / one-hour scheduled-open exit;
- leverage `0.5x`, fee `5 bp`, slippage `1 bp` per notional side;
- exact account multiplier `(1-0.0003)*(1+0.5*r)*(1-0.0003)`;
- full split wall-clock CAGR, including idle cash;
- held-path strict MDD with the favorable extreme applied first and the adverse
  extreme second; the scheduled exit bar's later high/low is excluded;
- weekly entry-cluster Rademacher test: 100,000 draws, seed `20260714`;
- mean gross underlying move recovered algebraically from the exact multiplier,
  with a strict `>12 bp` train and full-2023 hurdle.

## Frozen control clocks

1. `direction_flip`: exact primary timestamps and holds, opposite side.
2. `venue_swap`: USD-M flow direction and its own strictly-prior score threshold.
3. `reverse_time`: reverse within-bar minute ordering placebo.
4. `simultaneous_only`: simultaneous-minute agreement without ordered handoff.
5. `aggregate_only`: aggregate Spot flow coherence and magnitude only.
6. `basis_only`: accepted Spot direction plus residual basis only.
7. `asymmetry_only`: aggregate transfer/return asymmetry only.
8. `no_basis_lag`: primary handoff without the residual-basis requirement.
9. `no_activity_order`: primary handoff without Spot-before-USD-M activity order.
10. `stale_1h` / `stale_24h`: complete primary state stale by 12 / 288 bars,
    each with its own strictly-prior threshold.
11. `signal_delay_1bar`: already-selected primary side delayed one completed bar.

Every non-flip control uses its preregistered direction and independently
reserves a non-overlapping 12-bar schedule on the complete pre-2024 clock.
`direction_flip` and `signal_delay_1bar` are reported falsification controls.
The score-bearing comparison set is exactly `venue_swap`, `reverse_time`,
`simultaneous_only`, `aggregate_only`, `basis_only`, `asymmetry_only`,
`no_basis_lag`, `no_activity_order`, `stale_1h`, and `stale_24h`.

## Open and sealed windows

The first evaluator run may open only 2020–2022 train, full calendar 2023
selection, and fixed 2023 H1/H2. Calendar 2024 test, calendar 2025 eval, and
2026 YTD remain sealed. A trade is included only when signal, entry, and exit
all lie inside the split. Boundary slicing never rebuilds a schedule.

## Qualification

Train and full 2023 must each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, weekly-cluster one-sided
`p<0.10`, and mean gross underlying move strictly above 12 bp. Each 2023 half
must be positive with at least 200 trades. The primary minimum train/selection
CAGR-to-strict-MDD must strictly beat every frozen score-bearing control.
Failure rejects CATCH-12 v1 without threshold, side, hold, cost, or gate repair
and leaves 2024+ sealed.
