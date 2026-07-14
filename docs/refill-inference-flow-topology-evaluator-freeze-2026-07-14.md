# RIFT-96 evaluator contract freeze — 2026-07-14

## Status

**Evaluator design only; no RIFT return opened.** The source, tests, and this
contract are committed first. A second commit must record the exact evaluator
source hash with `opened_windows: []`; the evaluator refuses to load its price
frame unless that manifest matches.

## Immutable execution and metrics

- selected setup-score quantile: `0.925`;
- primary clock: exact frozen 460-row CSV, replayed field-for-field;
- every control clock: built globally through 2023 before split slicing;
- primary and all non-flip controls: long; exact direction flip: short;
- next-open entry, fixed 96-bar / 8-hour scheduled-open exit;
- leverage `0.5x`, fee `5 bp`, slippage `1 bp` per notional side;
- exact account multiplier `(1-0.0003)*(1+0.5*r)*(1-0.0003)`;
- full split wall-clock CAGR, including idle cash;
- held-path strict MDD with favorable extreme first, adverse extreme second,
  and no later high/low from the scheduled exit bar;
- weekly entry-cluster Rademacher test: 100,000 draws, seed `20260714`;
- mean gross underlying move recovered algebraically from the exact multiplier,
  with a strict `>12 bp` train and selection hurdle.

## Frozen control clocks and source requirements

1. `direction_flip`: exact primary timestamps/holds, action short.
2. `same_bar_static`: setup bar itself becomes the signal; joint quarantine.
3. `no_path_quality`: path fields removed from score, confirmation, and finite
   mask; joint quarantine.
4. `no_derivatives_crowd`: HHI/burstiness removed from score, confirmation, and
   finite mask; joint quarantine.
5. `centroid_free_momentum`: centroid mark removed from score, confirmation,
   and finite mask; joint quarantine.
6. `spot_only`: own sequence and finite mask; Spot quarantine only.
7. `stale_setup_1h` / `stale_setup_24h`: lagged complete setup paired with the
   current complete confirmation; joint current source plus lagged cleanliness
   embedded in the mask.
8. `signal_delay_1bar`: primary signal/action shifted one completed bar; the
   originating signal's joint cleanliness shifts with it.
9. `simple_two_bar_momentum`: only previous/current positive Spot and USD-M
   returns plus joint cleanliness/activity; no centroid, flow, or topology.

All control clocks are independently made non-overlapping before window
slicing. No control may replace a failed primary.

## Open and sealed windows

The first evaluator run may open only 2020–2022 train, full 2023 selection,
and fixed 2023 H1/H2. Calendar 2024 test, calendar 2025 eval, and 2026 YTD
remain sealed. A trade is included only if signal, entry, and exit all lie in
the split; boundary slicing never rebuilds a clock.

## Qualification

Train and full 2023 must each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, weekly-cluster one-sided
`p<0.10`, and mean gross underlying move strictly above 12 bp. Each 2023 half
must be positive with at least 30 trades; full 2023 must contain at least 80
trades. The primary minimum train/selection CAGR-to-strict-MDD must strictly
beat every frozen control. Failure rejects RIFT-96 without repair and leaves
2024+ sealed.
