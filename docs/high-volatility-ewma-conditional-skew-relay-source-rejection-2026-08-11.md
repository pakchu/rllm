# HVEWCS-24 source-support rejection — 2026-08-11

The frozen evaluator produced 22/41/40/32 train/test/eval/final events, but its
conditional-skew sign was regime-concentrated: train was 22/0 long/short, eval
4/36, and final 1/31. Train, eval, and final therefore failed fixed side-balance
gates, with additional month-concentration failures.

This first source failure is terminal. No estimator, half-life, rank, variation,
direction, clock, or hold repair is permitted. Gross9 clocks, execution prices,
funding, returns, and PnL remain unopened.
