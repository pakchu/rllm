# RITT-1 preregistration: residual inventory-transfer transition

## Narrow hypothesis

Two earlier ideas were individually causal but failed as static policies:

- premium-residualized spot/perpetual basis transitions,
- OI inventory-conservation residual plus carry ownership.

RITT-1 does **not** retune either failed rule. It tests only their interaction:
an abnormal basis residual that is still widening may behave differently when
new leveraged inventory is being created beyond what price and carry explain,
and when completed carry identifies which side likely owns that inventory.

## Exactly four inputs

1. long-window causal basis-residual z-score,
2. short-minus-long basis-residual acceleration,
3. one-bar-delayed 24h OI conservation-residual z-score,
4. carry sign, used only to label excess-inventory ownership.

The current state is `3 basis states × 3 inventory-owner states = 9`. The
learner estimates long and short executable utility for the transition from
the state 12 hours ago to the current state. It is a single empirical-Bayes
transition table, not HGB, HMM, a full feature search, or a Gross9 gate.

## Frozen search

- Physical source cutoff before `2025-01-01`.
- Annual expanding/purged fits for 2023 and 2024.
- Holds `{6h, 12h}` × posterior LCB `{0, 0.5}` = exactly four cells.
- At least 50 purged historical observations per traded transition.
- Next-open execution, 0.5x, 6bp/notional/side, no TP/SL.
- Full-calendar CAGR, non-overlapping sleeve trades, split-contained exits, and
  strict same-BTC OHLC MDD.

A standalone cell must be positive in 2023 and 2024, adequately populated, and
have minimum annual CAGR/strict-MDD at least 1.5. Only a survivor can be added
to frozen Gross9 at `0.25..1.00` gross.

Portfolio admission is intentionally harder than adding leverage: the candidate
must beat a same-gross pro-rata Gross9 leverage control, reduce strict MDD in at
least one pre-2025 window, keep entry Jaccard at most 0.25 versus every sleeve,
and earn positive standalone PnL while Gross9 is flat.

## Future boundary

Only the already frozen top pre-2025 row may open 2025/2026, and those periods
can veto but never rerank or repair it. Because those calendars were inspected
elsewhere, a survivor remains forward-shadow rather than pristine OOS.

The exact contract is machine-frozen in
`results/residual_inventory_transfer_transition_preregistration_2026-07-28.json`.
