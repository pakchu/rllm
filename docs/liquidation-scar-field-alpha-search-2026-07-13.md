# Causal Liquidation-Scar Price Field — Local-Tail Preflight

Date: 2026-07-13

## Hypothesis

A large delayed-OI contraction with one-sided taker flow may mark a price zone where
leveraged inventory was cleared. Instead of retaining this as another time-domain
OI score, the experiment deposits signed, decaying mass into absolute log-price
bins. When price later approaches an old zone, two opposite mechanisms are tested:

- **permeability:** cleared inventory leaves a low-resistance corridor;
- **fade:** depleted forced inventory makes the old zone support/resistance.

“Liquidation” is an inference, not an observed label. The only novel claim is the
spatial price-bin memory; scalar-collapse and relocated-bin controls test that claim.

## Causal protocol

- Physical source cutoff: rows strictly before `2024-01-01`.
- OI is explicitly delayed one complete 5-minute bar before differencing.
- OI contraction and taker flow are standardized from prior bars only.
- At bar `t`, the field is queried **before** depositing bar `t`; therefore the
  current deposit can first affect an entry after a later completed query.
- Entry is next 5-minute open, 0.5x, 6 bp/side, split-contained exits, and
  favorable-first/adverse-second strict MDD.
- 48 predeclared policies: log bin `{5,10}` bp, half-life `{288,864,2016}` bars,
  contraction fit tail `{q90,q95}`, mapping `{permeability,fade}`, hold `{24,72}`.
- A fixed fit-only q80 local-field threshold and fixed 30 bp query radius are used.
- Frozen `2024+` OOS remained unopened.

## Net result

All 48 policies lost in both the fixed-cost fit and 2023 selection scans. The
strongest adequately populated ranking was a 5 bp-bin, 2016-bar half-life, q90,
fade, 72-bar hold policy:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | -37.72% | -16.74% | 54.80% | -0.31 | 2,015 |
| Selection 2023 | -41.66% | -41.68% | 44.31% | -0.94 | 649 |
| 2023 H1 | -20.09% | -36.40% | 25.50% | -1.43 | 299 |
| 2023 H2 | -26.99% | -46.45% | 29.68% | -1.56 | 350 |

## Controls and cost decomposition

| Variant | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact direction flip | -87.83% / -0.63 | -22.75% / -0.89 |
| Scalar field, no price bins | -37.25% / -0.38 | 0.00% / 0.00, 0 trades |
| Monthly relocated deposit bins | -76.76% / -0.56 | -23.53% / -0.91 |
| Deposit at 12-bar-lagged price | -85.28% / -0.61 | -41.44% / -0.96 |

The top policy has strong direction separation in the old fit period, but its
zero-cost 2023 result is already **-13.88%**. Cost stress is therefore diagnostic,
not a rescue:

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | +108.70% / +1.54 | -13.88% / -0.70 |
| 1 bp | +70.61% / +1.04 | -19.29% / -0.79 |
| 3 bp | +14.02% / +0.17 | -29.12% / -0.88 |
| 6 bp | -37.72% / -0.31 | -41.66% / -0.94 |

## Decision

Reject the exact **repeated local-field q80 onset** usage before OOS. It overtrades,
fails at realistic costs, and the selected mapping is negative in 2023 even with
zero cost. Do not retune the same bin widths, tails, half-lives, or hold lengths.

The broader spatial-memory representation remains only a weak beta hypothesis for
a materially different, predeclared **first-passage/one-touch revisit** event that
consumes each scar once. That retry must beat scalar-collapse and relocated-bin
placebos; otherwise the family is just renamed OI/taker state.

Artifacts:

- `training/search_liquidation_scar_field_alpha.py`
- `results/liquidation_scar_field_alpha_scan_2026-07-13.json`
- `tests/test_search_liquidation_scar_field_alpha.py`
