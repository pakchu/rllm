# NWE-8 preregistration: causal-warmup network weak-signal ensemble

NWE-8 is a singleton, price-independent weekly alpha candidate. It combines
weak Bitcoin blockspace-demand and ledger-topology states in an online ridge
model. The model input excludes BTC price/return, exchange microstructure,
funding, premium, basis, OI, Kimchi, DXY, FX, REX, and Markov features.

## Why this is a new admissible policy

NWE-7 was rejected before any return label or PnL was constructed. Its only
failed support check was mechanical: the first prediction date had 41 fully
available historical labels versus the already frozen minimum of 52. NWE-8
changes only the first prediction date from `2021-03-01` to `2021-06-07`, when
the unchanged 52-label causal warm-up can exist. Features, estimator, training
window, abstention rule, execution, leverage, costs, and seven-day hold remain
unchanged.

This correction is frozen before opening any NWE return. It is not a repair
based on profitability.

## Frozen model

- Decision: Monday 12:00 UTC.
- Entry: Monday 12:05 UTC open.
- Exit: seven days after entry, at the scheduled open.
- Inputs: eight clipped, strictly-prior 180-day z-scores covering fee share,
  transaction density, address breadth, and transaction fanout levels/7-day
  changes.
- Fit: latest 104 fully labelled weekly rows, minimum 52; training-only
  standardization; ridge alpha 10.
- Drift removal: center the training target and never add its mean back.
- Trade: forecast sign determines long/short; abstain below the median absolute
  in-sample centered fitted forecast.
- Exposure: 0.5x; no stop or take profit.
- Costs: 6 bp/notional/side base and 10 bp/notional/side stress; realized
  funding included.

## Selection boundary

- Train: `2021-06-07 <= decision < 2023-01-01`.
- Selection: calendar 2023, with both halves required positive.
- 2024, 2025, and 2026 YTD remain sealed.
- Candidate count: exactly one; no parameter repair.
- Required in both train and 2023: positive absolute return, CAGR/strict-MDD at
  least 3, strict MDD at most 15%, cluster sign-flip p-value at most 0.10,
  average gross underlying edge at least 20 bp, and positive 10 bp/side stress.

The strict-MDD evaluator must use the global/pre-entry high-water mark,
favorable-before-adverse held OHLC, realized funding, entry/exit costs, and
hypothetical liquidation costs. Full wall-clock split duration, including
abstained cash weeks, is used for CAGR.

If the support, performance, mechanism controls, orthogonality, or marginal
portfolio contribution fails, NWE-8 is rejected without changing the frozen
policy.
