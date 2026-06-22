# DXY-low Kimchi regime candidate extended eval — 2026-06-23

## Candidate

Family found by bounded regime scan:

- Regime: `dxy_zscore:low` using train-fitted low bucket.
- Signal: `kimchi_premium_zscore` quantile rule.
- Train: 2023-01-01 → 2024-06-30.
- Test/selection: 2024-07-01 → 2025-08-31.
- Extended eval: 2025-09-01 → 2026-05-31.
- Leverage: 0.5.

## Extended result

Gate decision: `NO_GO`, passed `0 / 4`.

Best candidate:

- `dxy_zscore:low -> kimchi_premium_zscore`, horizon 144
  - Test: CAGR 21.02, strict MDD 9.74, ratio 2.16, trades 464.
  - Extended eval: CAGR 33.54, strict MDD 9.04, ratio 3.71, trades 263.
  - Total test+eval trades: 727.
  - Failure: test ratio remains below target 3.0.

Other observations:

- Horizon 288 degrades under extended eval: eval CAGR 20.42, strict MDD 15.53, ratio 1.31.
- `dxy_zscore:high` is consistently bad for the same signal, confirming the regime direction is meaningful rather than arbitrary.

## Interpretation

This is not production-ready, but it is the strongest non-cheating family found in the reset path so far:

- It survives a longer recent eval through 2026-05 for horizon 144.
- It has statistically more meaningful trade count than the earlier tiny-window wins.
- It still fails the user's target because the selection/test window is profitable but not strong enough.

Next step should be an RLLM-shaped refinement around this family, not another broad gate sweep: represent DXY-low/Kimchi context as causal text state and train a compact single policy to decide activation, side, and abstention around this prior, then validate on the same no-leak train/test/eval split.
