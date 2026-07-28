# AOSP-1 preregistration: annual OI/spot participation path model

## Hypothesis

The current Gross9 book lacks a sleeve whose primary state is **cash-market
participation versus leveraged futures inventory**.  A low-complexity annual
model may combine several weak facts that are individually too small:

- spot/perpetual return leadership,
- spot cash-volume participation versus perpetual quote volume,
- one-bar-delayed OI growth and OI/price divergence,
- OI value relative to rolling spot notional,
- perpetual taker flow and price-risk controls.

The model predicts long and short executable path utility rather than raw price.

## What is deliberately excluded

- future-labelled REX records,
- DXY, USDKRW, Kimchi, funding, and premium-index inputs,
- Rank7 source gates, barrier exits, nested-barrier features, and braid features,
- spot fields that are present in the cache schema but historically all-null.

This keeps the experiment materially different from Rank7 and the earlier
positioning HGB scan.

## Frozen process

- Decision: each completed hourly 5m bar.
- OI delay: one full 5m bar; missing inputs fail closed.
- Models: four fixed low-depth HGB path specifications.
- Policies: confidence q80/q90 × both/long/short = 24 cells.
- Execution: next-open, 0.5x base, 6bp/notional/side, fixed 6h/12h/24h hold.
- Refit: expanding annually with purged targets.
- Selection: annual OOS 2023 and 2024 only.
- Future: 2025/2026 one-shot veto using their past-only annual refits.

## Success condition

A standalone cell must survive both 2023 and 2024.  It is promoted to a
candidate only if adding 0.25–1.00 gross to frozen Gross9 improves the minimum
train/2024 CAGR-to-strict-MDD ratio while respecting the existing strict-MDD and
gross limits.

The exact feature definitions, 24-cell grid, tie-breakers, and future veto are
machine-frozen in
`results/annual_oi_spot_participation_path_preregistration_2026-07-28.json`.

## Claim boundary

The programme has already inspected 2025/2026.  Passing them can veto a broken
idea, but cannot establish pristine OOS evidence.  Any survivor remains
forward-shadow until genuinely new live data accumulates.
