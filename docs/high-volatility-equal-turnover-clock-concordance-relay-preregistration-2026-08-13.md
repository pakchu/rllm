# HVETCC-8 preregistration

`HVETCC-8` is a singleton, outcome-blind candidate for July-like volatile BTC markets.

At each exact 02:00, 10:00, and 18:00 UTC decision, the completed prior eight hours of BTCUSDT perpetual one-minute bars are transformed into an intrinsic participation clock. Each indivisible minute is assigned, before adding its turnover, to one of four sequential quartiles of cumulative quote turnover. The return of a segment is the sum of its constituent one-minute `log(close/open)` returns.

The candidate is eligible only when all four segments are nonempty, all four segment returns are strictly nonzero with one common sign, and completed eight-hour realized variation has a causal strict-prior midrank of at least 0.65 over at most 270 earlier source-valid decisions with at least 180 observations. Eligibility must be a new onset. Entry is the exact BTCUSDT perpetual open five minutes after the decision and the frozen hold is eight elapsed hours at 0.5 gross exposure.

This is not dynamic volume-bucket toxicity, a volume-weighted value anchor, equal-variance time, or fixed-duration persistence. It uses the direction of returns accumulated in four endogenous equal-turnover segments and no taker split, funding, OI, premium, fitted outcome, reused event set, or promoted control.

The source-support, Gross9 novelty, sequential economic, stress, split-half, and conditional RV20 gates are inherited unchanged from the frozen high-volatility protocol. Any first failure is terminal; no formula, clock, threshold, side, hold, subset, or control may be repaired or promoted.
