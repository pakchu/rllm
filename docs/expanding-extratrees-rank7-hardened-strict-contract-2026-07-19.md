# ExtraTrees rank-7 hardened strict audit contract — 2026-07-19

## Purpose

The frozen five-seed `ensemble5_300` rank-7 policy is the strongest existing
standalone Bitcoin candidate in this repository. Its historical metric omitted
a virtual exit cost at the worst held-price mark and treated funding at exact
entry/exit timestamps symmetrically. This audit answers whether the candidate
still clears the target after both accounting rules are made conservative.

This is a **retrospective verification**, not pristine discovery OOS. The 2025+
results were already viewed in the historical study. No result from this audit
may be used to retune the policy.

## Immutable policy

- long only;
- annual expanding refits with cutoff-crossing labels purged;
- five seeds: `7, 71, 715, 2026, 71515`;
- 300 trees per seed; mean ensemble;
- `max_depth=2`, `min_samples_leaf=32`, `max_features=0.8`;
- `score = predicted_net - 0.25 * predicted_adverse`;
- funding/premium score quantiles `0.40/0.55`;
- adverse-risk quantile `0.75`;
- one-hour delayed feature matrix;
- source-owned exits, next-open entry, non-overlap, `0.5x` leverage;
- base execution cost `6 bp/notional/side`.

The independently reconstructed selected-position hash must equal
`8ffbd55f07ceda0e82c270fe4b370fffba44bb3fcfc807368c4385d2ba97f531`.
Every yearly and combined trade-clock hash must exactly match the frozen rank-7
OOS artifact before any hardened result is accepted.

## Only permitted accounting changes

1. Strict MDD uses the global/pre-entry high-water mark, entry cost,
   favorable-before-adverse held 5-minute OHLC, realized funding debit, a
   **virtual liquidation cost at the adverse mark**, and actual exit cost.
2. Funding strictly inside a trade is symmetric. If a funding timestamp exactly
   equals entry or exit, a debit is included and a credit is excluded.

Full-calendar CAGR includes idle time. Absolute return is always reported.

## Frozen gates

- each of 2023, 2024, 2025: positive absolute return, CAGR/strict-MDD `>=3`,
  strict MDD `<=15%`, at least 12 trades;
- 2026H1: same gates with at least 6 trades;
- combined 2025–2026H1 and 2023–2026H1: CAGR/strict-MDD `>=3`, strict MDD
  `<=15%`, at least 18/42 trades respectively;
- both combined windows remain profitable at `10 bp/notional/side`;
- both combined windows have one-sided weekly-cluster sign-flip and stationary
  trade-block-bootstrap p-values `<=0.10`.

Any failed gate downgrades rank-7. There is no threshold, hold, feature, seed,
tree-count, cadence, direction, or schedule repair in this work unit.
