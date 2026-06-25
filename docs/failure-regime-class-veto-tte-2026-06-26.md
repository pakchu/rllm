# Failure-regime class veto TTE (2026-06-26)

## Purpose

The previous regime ridge gate reduced 2025 damage only weakly and failed 2026H1 eval. This pass tests explicit failure-regime classes as hard veto candidates rather than just numeric ridge features.

Implemented:

- `append_failure_regime_classes()` in `training/sparse_setup_regime_gate_tte.py`
- New TTE sweeper: `training/sparse_setup_failure_veto_tte.py`

Failure classes are fixed-threshold, past-only descriptors:

- `fr__chop_compression`
- `fr__chop_midrange`
- `fr__trend_conflict`
- `fr__volatility_transition`
- `fr__dxy_impulse_up`
- `fr__dxy_impulse_down`
- `fr__usdkrw_impulse`
- `fr__kimchi_shock`
- `fr__flow_dislocation`
- `fr__near_upper_extreme`
- `fr__near_lower_extreme`
- `fr__extreme_overlap_compression`
- `fr__macro_shock_cluster`
- `fr__chop_or_conflict`
- `fr__breakout_failure_risk`

## Protocol

- Candidate pool: train-only sparse setup discovery from `2020H1` through `2024H2`.
- Ridge gate fit: train only.
- Quantile and veto selection: test `2025H1/H2` only.
- Final eval: refit on train+test, apply selected quantile and veto to untouched `2026H1`.

Artifact:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/failure_veto_tte_single.json`

## Result

Single-veto sweep (`max_veto_size=1`) selected **no veto**.

| Period | CAGR | Strict MDD | CAGR/MDD | Trades |
| --- | ---: | ---: | ---: | ---: |
| Train 2020-2024 | 43.42% | 19.24% | 2.26 | 471 |
| Test 2025 | -0.70% | 10.08% | -0.07 | 71 |
| Eval 2026H1 | -34.66% | 14.03% | -2.47 | 36 |

Top test candidates all remained negative. The best single veto was `fr__dxy_impulse_up`, but it only changed test CAGR from `-0.70%` to `-0.16%`, still failing.

## Interpretation

Explicit handcrafted failure classes did not solve generalization. They describe plausible risk states, but they do not align cleanly enough with the realized bad sparse setup trades.

The result argues against continuing manual veto engineering on this candidate pool. The next useful step is to change the target construction:

1. Label failure regimes from realized clusters of bad event outcomes, not from hand-authored classes alone.
2. Mine which feature combinations actually separate 2025 losers from train winners.
3. Use those mined failure descriptors as LLM-readable context and as safety labels.

In short: hand-written `chop/macro shock/extreme` classes are too blunt. The model needs data-mined failure clusters.
