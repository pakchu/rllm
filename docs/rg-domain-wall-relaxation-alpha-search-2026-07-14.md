# RG Domain-Wall Relaxation Alpha Search — 2026-07-14

## Mechanism

This experiment treats price direction as a field across dyadic time scales,
not as a list of independent multi-timeframe momentum indicators. At each
completed bar it constructs four causally smoothed, volatility-normalized
directional fields. A candidate requires all three objects:

1. a stable coarse fixed point: the two longest-scale fields agree and barely
   move under further coarsening;
2. a domain wall: the finest field points against that coarse auction;
3. relaxation curvature: the intermediate fields rotate monotonically back
   toward the coarse fixed point.

The fixed map trades toward the coarse field, interpreting the fine-scale move
as a localized liquidity defect relaxing into a stable larger auction.

## Causal protocol and bounded search

- Physical source cutoff strictly before `2024-01-01`; OOS stayed sealed.
- Causal EWM levels and return variance use history through the completed bar.
- Minute-55 signal enters the following minute-00 open.
- Fit-only score thresholds; 2023 is internal selection.
- 8 final policies: base scale `{24,48}` bars × score tail `{q70,q80}` × hold
  `{6h,12h}`.
- Fixed field floor `0.25`, coarse tolerance `0.10`, `0.5x`, `6bp/side`, strict
  favorable-first/adverse-second OHLC MDD.

A prior 16-policy hard-sign domain-wall propagation/contraction probe was also
weak. Those precursor semantics and the final eight policies are frozen.

## Result

The family failed primarily on support. Base scales 24 and 48 generated only
283 and 243 raw candidates over the entire pre-2024 history. No final policy
reached the required 20 selection trades.

The most-supported policy was base 24, q70 curvature, six-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | -3.49% | -1.59% | 14.85% | -0.11 | 41 | 17/24 |
| 2023 | -2.52% | -2.52% | 3.90% | -0.65 | 16 | 10/6 |
| 2023H1 | -0.89% | -1.78% | 2.65% | -0.67 | 9 | 7/2 |
| 2023H2 | -1.65% | -3.25% | 2.44% | -1.33 | 7 | 3/4 |

The q80 sibling had only 12 selection trades and also lost. The base-48 family
had at most five selection trades at q70. Weakening fixed-point/domain-wall
definitions after observing this scarcity would be post-hoc tuning, so the
experiment stops here.

## Decision

**Rejected; no beta promotion and no OOS opening.** Record the exact field
floors, scale sets, curvature score, tails and holds as gamma failure
provenance. The concept is coherent but the present static definition is too
sparse and has no pre-2024 directional evidence.

Artifacts:

- `training/search_rg_domain_wall_relaxation_alpha.py`
- `tests/test_search_rg_domain_wall_relaxation_alpha.py`
- `results/rg_domain_wall_relaxation_alpha_scan_2026-07-14.json`
