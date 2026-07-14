# LURI-48 preregistration — 2026-07-14

## Status

**LURI-specific support only; LURI outcomes unopened.** The earlier CATCH-12
clock has already been evaluated through 2023, but no LURI-48 scheduled
post-entry return, held path, win rate, CAGR, or MDD has been calculated. This
document freezes a new episode definition, direction, support-only basis grid,
control family, execution, funding treatment, and return gate. Calendar 2024+
remains sealed.

- name: **LURI-48 — Leveraged USD-M Inventory Release Handoff**
- feature source SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`
- inspected feature horizon: strictly before `2024-01-01`
- formation: 36 completed five-minute bars / three hours, ending at `t-1`
- trigger: completed five-minute bar `t`
- entry: next Binance USD-M five-minute open
- exit: fixed USD-M open 48 bars / four hours later

“Inventory” is an inference from aggressive signed quote flow, not a claim that
exchange account positions are observed. The economic object is a leveraged
auction that accumulated without cash-market confirmation, displaced the
USD-M/Spot basis, and then visibly began releasing from USD-M into Spot.

## Formation state through `t-1`

All 36 formation bars and trigger bar `t` must be clean under the frozen source
and post-defect quarantine. Let:

```text
U = sum(um_signed_quote_notional[t-36:t-1])
    / sum(um_quote_notional[t-36:t-1])
S = sum(spot_signed_quote_notional[t-36:t-1])
    / sum(spot_quote_notional[t-36:t-1])
R_cash = sum(spot_log_return_5m[t-36:t-1])
I = sign(U)
u = -I
delta_basis = close_basis_bp[t-1] - open_basis_bp[t-36]
basis_displacement = I * delta_basis
```

Require:

1. `I != 0`;
2. `basis_displacement > 0`;
3. `-I * S >= 0`: Spot aggressive flow did not confirm the USD-M side;
4. `-I * R_cash >= 0`: Spot price did not confirm the USD-M side;
5. `basis_displacement` is at least its strictly prior rolling percentile.

The percentile is calculated from positive, clean, 36-bar-complete basis
displacements with a one-bar shift, 8,640-bar lookback, and 2,016-row minimum.
The support-only grid is fixed at `0.25`, `0.40`, `0.55`, and `0.70`. Select the
highest row passing every count, balance, ablation, temporal-placebo, and
prior-clock novelty floor. No return is available to this selection.

## Release trigger on completed bar `t`

Define:

```text
forward_handoff = min(
    max(um_to_spot_lagged_directional_alignment, 0),
    max(-lagged_directional_alignment_diff, 0),
)
reverse_handoff = min(
    max(reverse_um_to_spot_lagged_directional_alignment, 0),
    max(-reverse_lagged_directional_alignment_diff, 0),
)
```

Require:

1. `u * um_flow_fraction > 0`;
2. `u * um_log_return_5m > 0`;
3. `u * basis_change_bp > 0`: the prior basis displacement starts compressing;
4. `um_minus_spot_activity_time_centroid < 0`: USD-M activity occurs earlier;
5. `forward_handoff > 0`;
6. `forward_handoff > reverse_handoff`.

Condition 6 is a mandatory temporal falsification inside the primary rule. It
prevents a later-to-earlier reconstruction from receiving the same event label.
The action is `u = -I`: trade against the prior inferred USD-M inventory side.
All trigger features are available only after bar `t` completes, so execution
remains the next five-minute open.

## Frozen controls

Every non-flip control reserves its own global non-overlapping 48-bar clock
before any split or return is opened.

1. `direction_flip`: exact primary clock, side `I` instead of `u`.
2. `no_inventory`: current forward USD-M release only; no 36-bar state.
3. `no_basis_history`: remove only historical basis displacement/percentile.
4. `no_cash_refusal`: remove only Spot flow and Spot price refusal.
5. `spot_confirmed`: replace both refusal signs with same-side Spot confirmation.
6. `basis_only`: keep formation/basis and current USD-M reversal/compression,
   but remove ordered handoff, timing, and forward-over-reverse conditions.
7. `reverse_time`: require `reverse_handoff > forward_handoff` instead.
8. `simultaneous_only`: replace ordered handoff/timing with positive simultaneous
   flow-sign and return-sign agreement.
9. `spot_inventory_swap`: mirror formation and release across Spot/USD-M,
   including the opposite basis sign and Spot-to-USD-M handoff.
10. `stale_24h`: delay the entire selected event and side by 288 bars.
11. `signal_delay_1bar`: delay the entire selected event and side by one bar.
12. frozen CSPR-12, RIFT-96, and CATCH-12 primary scheduled clocks, plus the
    outcome-blind replay of CATCH's selected-quantile `venue_swap` raw and
    independently scheduled control clocks, for novelty overlap.

The score-bearing return-comparison controls are 2–10. Exact direction flip and
one-bar signal delay are reported falsification controls but are not alternative
selection candidates.

## Frozen support and novelty floors

- non-overlapping total at least `420`;
- each calendar year 2020–2023 at least `95`;
- each 2023 half at least `45` and each 2023 quarter at least `20`;
- each side at least 30% overall and at least `35` events per year;
- at least `42` months with at least `5` scheduled events;
- primary/no-inventory raw and scheduled retention at most `0.10`;
- primary/no-basis-history raw and scheduled retention at most `0.55`;
- primary/no-cash-refusal raw and scheduled retention at most `0.20`;
- primary/basis-only raw and scheduled retention at most `0.30`;
- reverse-time, Spot-confirmed, and venue-swap raw/scheduled Jaccard at most
  `0.05` and primary containment at most `0.10`;
- simultaneous-only Jaccard at most `0.15`; primary containment at most `0.80`
  raw and `0.60` scheduled;
- stale-24h and one-bar-delay Jaccard at most `0.01` and containment at most
  `0.02`;
- each frozen prior primary clock has Jaccard and primary containment at most
  `0.01`;
- both CATCH `venue_swap` raw/scheduled comparisons have Jaccard at most `0.02`
  and LURI primary containment at most `0.20`, proving that at least 80% of
  LURI events are outside the structurally nearest CATCH control.

Failure rejects LURI before returns. The basis grid, formation, hold, side,
controls, and floors may not be repaired after support is run.

## Frozen return gate

- train: `2020-01-01 <= t < 2023-01-01`;
- selection: full 2023 and fixed H1/H2;
- 2024 test, 2025 eval, and 2026 YTD remain sealed;
- leverage `0.5x`, fee `5 bp`, slippage `1 bp` per notional side;
- realized Binance USD-M funding is applied at every settlement satisfying
  `entry_time <= funding_time <= exit_time` using
  `funding_factor = product(1 - 0.5 * side * funding_rate)`;
- final multiplier:
  `(1-0.0003) * (1+0.5*r) * funding_factor * (1-0.0003)`;
- full-clock CAGR including idle cash;
- strict held-path MDD applies the favorable price extreme before the adverse
  extreme, excludes the scheduled exit bar's later high/low, applies funding
  debits before the adverse equity, and never lets funding credits raise the
  intratrade peak;
- weekly entry-cluster Rademacher test, 100,000 draws, seed `20260714`.

LURI advances only if train and full 2023 each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, one-sided cluster `p<0.10`,
and mean gross underlying move strictly above 12 bp. Each 2023 half must be
positive with at least 45 trades. The primary minimum train/selection ratio
must strictly beat every frozen score-bearing control. Failure rejects v1
without threshold, direction, formation, hold, funding, or gate repair.
