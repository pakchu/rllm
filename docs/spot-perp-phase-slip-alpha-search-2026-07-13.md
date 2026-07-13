# Spot–Perp Symbolic Phase-Slip Alpha — Preflight Rejection

Date: 2026-07-13

## Novel hypothesis

Spot and perpetual 5-minute direction symbols normally move in lockstep. The tested
event required this exact path order:

1. a prior synchronized non-flat state,
2. a short interval in which spot advances ahead of the perpetual, and
3. the first completed bar where the perpetual symbol relocks to spot.

The policy traded the relock direction at the next open. This is not a transfer-
entropy tail, basis magnitude, residual-correlation, OI, funding, or liquidation
policy; only the symbolic path order defines an event.

## Causal protocol

- Physical source cutoff: rows strictly before `2024-01-01`.
- Current returns use completed spot/perp bars; volatility uses only prior bars.
- Fixed state deadband: `0.25` prior-volatility units.
- Prior lock and slip windows are disjoint. The current completed bar may only
  establish relock; entry is the next 5-minute open.
- 72 predeclared policies: lock `{12,24}`, slip `{2,3,4}`, excess `{2,3}`,
  relock `{soft,hard}`, hold `{6,12,24}` bars.
- 0.5x exposure, 6 bp/side implementation cost, split-contained exits, and
  favorable-first/adverse-second conservative strict MDD.
- `2020-06..2022` is fit robustness and `2023` is selection robustness. Frozen
  `2024+` OOS remained unopened.

## Strongest adequately populated policy

Parameters: lock `24`, slip `4`, minimum symbolic excess `2`, hard relock,
hold `24` bars.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | -3.89% | -1.52% | 6.01% | -0.25 | 175 |
| Selection 2023 | -5.49% | -5.49% | 5.85% | -0.94 | 84 |
| 2023 H1 | -2.19% | -4.36% | 2.48% | -1.76 | 34 |
| 2023 H2 | -3.37% | -6.59% | 3.72% | -1.77 | 50 |

Only two of seven half-year segments were profitable. Among all 72 policies,
no policy had positive returns in both fit and 2023. Positive maxima came from
one-to-four-trade underpopulated variants and are not evidence.

## Negative controls on the top policy

| Control | Fit return / ratio | 2023 return / ratio | Result |
|---|---:|---:|---|
| Exact direction flip | -15.85% / -0.38 | -4.37% / -0.87 | Also loses. |
| Act without relock | -14.35% / -0.35 | -13.60% / -0.95 | Slip alone is worse. |
| Perp-led role swap | -11.47% / -0.36 | -4.74% / -0.88 | No reverse leadership edge. |
| Spot lagged one bar | 0.00% / 0.00, 0 trades | 0.00% / 0.00, 0 trades | Timing placebo destroys events. |
| Spot lagged twelve bars | +0.42% / +0.09, 3 trades | 0.00% / 0.00, 0 trades | No viable placebo path. |

## Decision

Reject the exact spot-led symbolic slip→relock continuation family before OOS.
Relock is observable and timing-specific, but after it occurs there is no stable
residual move large enough to survive costs. Both the intended direction and exact
flip lose. Do not tune lock/slip/excess/hold values; a new mechanism must encode a
different economic state rather than another synchronization threshold.

Artifacts:

- `training/search_spot_perp_phase_slip_alpha.py`
- `results/spot_perp_phase_slip_alpha_scan_2026-07-13.json`
- `tests/test_search_spot_perp_phase_slip_alpha.py`
