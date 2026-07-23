# FCCM-72 source-support implementation contract

## Scope

This document resolves implementation order and denominator details for the
already-frozen FCCM-72 mechanism. It does not change a source, feature,
threshold, direction, hold, control, gate, or stopping rule.

It is frozen before any FCCM Bitfinex or WBTC value row, FCCM feature/state,
candidate incidence, comparator value row, BTC bar, realized funding row,
future return, PnL, CAGR, or MDD is opened. The sealed preregistration remains
the controlling specification if this implementation contract conflicts with
it.

## Exact state and batch interpretation

- A single-component control's state is that component's exact vote in
  `{-1, 0, +1}`. It uses the primary mechanism's equal-availability batch,
  invalid-reset, greatest-hour-only, first-valid-baseline, transition, WBTC
  suppression, and no-queue rules.
- `majority_without_score` uses `+1` for at least two positive votes, `-1` for
  at least two negative votes, and `0` otherwise, with the same state-machine
  rules.
- Rank warm-up with fewer than all 720 prior valid feature rows cannot
  establish or change any state. It is not an invalid reset. Once available,
  all three component ranks use the same 720-row causal history.
- In an equal-availability batch containing any invalid anchor, every state
  machine resets. Valid feature rows from that batch enter rank history only
  after all batch ranks are computed, but no row in that batch establishes or
  changes state.

## Source-support denominator and split interpretation

- The WBTC active-share denominator is every consensus directional transition
  whose derived entry belongs by entry time to train or selection, before WBTC
  sponsorship, split-containment filtering, or chronological non-overlap.
  Report train and selection independently.
- Component vote-share denominators are accepted primary entries within each
  split after sponsorship, split containment, and global non-overlap.
- A transition whose entry is in a declared split but whose 72-hour exit is
  outside that split remains in the raw WBTC active-share denominator, but it
  cannot become an accepted clock or contribute to accepted-entry gates.
- Maximum entry gap is measured only between consecutive accepted entries
  within the same split; no synthetic gap from a split boundary is added.
- Every source-support ratio is an exact reduced rational. Gate comparisons do
  not use binary floating point.

## Source integrity and post-seal boundary

- A duplicate canonical source identity, malformed permitted value, invalid
  WBTC actor/amount, or impossible timestamp is a whole-run source-integrity
  failure. It is never silently removed or localized to a shorter dependency
  horizon.
- The hash-bound source files may contain confirmation metadata whose
  `available_at` reaches 2024 for a pre-2024 event. The loader first decodes
  only the causal timestamp sentinel. Rows with `available_at >=
  2024-01-01T00:00:00Z` are not decoded into feature values and are reported
  separately as timestamp sentinels. Therefore post-2023 **source value rows
  loaded** remains zero.
- Bitfinex rows are pre-screened by both observation hour and causal
  `available_at`; WBTC rows are pre-screened by `available_at`. These timestamp
  sentinels are decoded before any permitted numeric or actor feature value.
  No post-seal row may enter a rank, state, transition, or clock.
- Any malformed pre-screen timestamp is a whole-run source-integrity failure.

## Controls and placebo boundary

- Each causal control owns an independent chronological `[entry, exit)`
  reservation. A primary reservation never suppresses a control.
- `bitfinex_stale_24h` takes the already-computed consensus transition at exact
  source hour `H-24h`, moves that transition to hour `H`, requires `H` to be the
  greatest eligible row of a valid current causal batch, uses `H`'s unchanged
  causal availability and current WBTC state, and never changes the 72-hour
  hold.
- `wbtc_stale_7d` uses the already-computed WBTC daily state at exact `D-7d`
  while retaining the current Bitfinex signal and execution clock.
- Amount and actor permutations may report only raw transition sponsorship
  incidence and exact shares. They never enter the executable clock CSV,
  chronological scheduler, novelty cohort, or economics.

## Artifact and stopping contract

- The source-support program and its tests must be committed and identical to
  `HEAD` before a real source value row is decoded.
- The program may read only the two frozen FCCM source files and sealed
  preregistration metadata. It cannot open comparator value rows or any market
  outcome container.
- A failed support check writes an auditable source-only clock/report and
  retires FCCM-72 unchanged. It does not authorize a novelty or economic
  evaluator.
