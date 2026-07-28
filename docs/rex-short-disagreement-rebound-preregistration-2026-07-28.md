# RSDR-1 preregistration: REX-short disagreement rebound

## Target

Find a **separate long sleeve** that has positive marginal value for the frozen
Gross9 portfolio when a raw REX short setup fails to continue.  The candidate is
not a veto, not a rewrite of REX, and not another weight-only optimization.

## Economic mechanism

1. Recompute the REX HTF pullback/reclaim clock from completed market bars.
2. Fit its q75 activation threshold only on 2020-09 through 2021-12.
3. Keep raw `SHORT` anchors whose one-bar-delayed 4h OI change exceeds the
   corresponding price change by a frozen anchor-relative quantile.
4. Wait 30m or 60m.  Require price reclaim, improving taker flow, retained OI,
   bounded downside, and an upper post-anchor range close.
5. Enter `LONG` at the next 5m open and hold for 3h, 6h, or 12h.

This asks whether leveraged short inventory is trapped after the price stops
confirming the original REX-short thesis.

## Why this is materially different

- It uses an otherwise unused OI-family budget.
- It is intentionally opposite-side during a known REX-short risk cluster.
- It is selected by **marginal Gross9 CAGR/strict-MDD**, not standalone return
  alone.
- It derives the raw REX clock from market bars and never reads the
  future-labelled REX SFT JSONL.

## Frozen search

- Mechanism cells: `2 OI quantiles × 2 waits × 2 OI retention floors ×
  2 recovery locations × 3 holds = 48`.
- Added weight: `0.25, 0.50, 0.75, 1.00`; total gross never exceeds 10.
- Base leverage: 0.5x.
- Cost: 6bp/notional/side; stress: 10bp/notional/side.
- Same-BTC upper-before-lower strict MDD; idle calendar time remains in CAGR.

## Stage gates

1. Fit thresholds on 2020-09 through 2021-12.
2. Use 2022 and 2023 only for support/robustness and freeze at most 12 cells.
3. Open 2024 only for the frozen shortlist and choose a unique portfolio top-1.
4. Open 2025/2026 only as a one-shot veto; never repair or rerank.

The full machine-readable gates and tie-breakers are frozen in
`results/rex_short_disagreement_rebound_preregistration_2026-07-28.json`.

## Interpretation boundary

The programme has already inspected 2025/2026 and Gross9 itself is a research
candidate.  A pass can justify a forward-shadow sleeve, not a pristine-OOS or
automatic-live claim.
