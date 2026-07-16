# NWE-7: Network Weak-Signal Ensemble preregistration

Status: **frozen before constructing any return-labelled NWE-7 sample**.

## Why this is a new candidate

NTB-7 and BFC-3 were sparse threshold-event hypotheses. Both failed their
outcome-blind support gates, so no PnL was opened and no threshold was repaired.
NWE-7 instead tests the user's weak-signal premise directly: eight continuous,
price-independent network features are combined by one fixed online ridge
decoder at a weekly cadence.

Inputs are only:

- fee/issuance level and seven-day change,
- transactions/block level and seven-day change,
- active-addresses/transfer level and seven-day change,
- transfers/transaction level and seven-day change.

Every value is a causal 180-observation z-score computed from already-published
Coin Metrics rows and clipped to ±5. No BTC price, return, volume, derivative,
macro, exchange-tag, or existing-alpha feature enters the model.

## Frozen weekly information clock

- Decision: every Monday 12:00 UTC.
- Snapshot: latest common Coin Metrics observation published by the decision
  and no more than three days old.
- Entry: Monday 12:05 UTC open, after one complete five-minute latency bar.
- Exit: exactly seven days later at the scheduled open.
- Exposure: 0.5x; no stop/take-profit; no overlap.
- Base cost: 6 bp/notional/side; stress: 10 bp/notional/side.

## Frozen online model

At each Monday refit:

1. Admit only historical weekly samples whose feature source and seven-day
   label exit are both already available.
2. Keep the most recent 104; require 52.
3. Standardize features on those samples only.
4. Center the training return target and **do not add the target mean back**.
   This removes the unconditional BTC long drift.
5. Fit closed-form ridge with `alpha=10`, no intercept and no parameter search.
6. Abstain unless the absolute forecast is at least the median absolute
   in-sample fitted centered forecast.
7. Positive forecast is long; negative is short.

This is an expanding/rolling continuous predictor in the causal sense: each
week can learn from newly completed labels, but never from the current week's
future.

## Leakage and revision boundary

- Support construction loads no market rows or return labels.
- Model labels are open-to-open returns and enter training only after exit.
- Market and funding readers must physically stop before 2024 during selection.
- Exchange-tag flow metrics are excluded because no point-in-time tag archive
  is available.
- Raw on-chain files use `AssetEODCompletionTime` and frozen hashes, but are not
  a full historical revision-vintage archive; live promotion requires forward
  vintage parity.

## Gates

Before labels, the finite weekly feature clock must contain at least 90
2021-2022 candidates, 42 in 2021, 50 in 2022, 50 in 2023, and 25 in each 2023
half.

After the evaluator is hash-frozen, both train and 2023 must have positive
absolute return, CAGR/strict-MDD ≥3, strict MDD ≤15%, weekly-cluster
`p<=0.10`, mean gross move ≥20 bp, and positive 10 bp/side stress results.
Both 2023 halves and one-bar-delayed execution must be positive. Train needs 35
trades, 2023 needs 18, each half needs 8, and both long and short must represent
25–75% of trades.

Strict MDD includes the global/pre-entry HWM, favorable-before-adverse held
OHLC, funding, and entry/exit/hypothetical-liquidation costs. CAGR counts the
full split clock including abstained weeks.

Controls are exact side flip, fee-only, topology-only, no abstention, seven-day
stale features, deterministic within-year feature permutation, constant weekly
long, and one-bar-delayed execution. Only a performance pass proceeds to
entry/position/PnL orthogonality and marginal portfolio contribution tests.
2024, 2025, and 2026 YTD remain sealed.
