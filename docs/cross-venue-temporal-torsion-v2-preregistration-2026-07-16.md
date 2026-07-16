# CVTT v2 support-only repair preregistration (2026-07-16)

## Why v2 exists

CVTT v1 was rejected without opening any trade return. It required 2,016
eligible route events in a 30-day window even though the observed maxima were
1,028 and 979. It also used a 3% monthly source-quarantine ceiling while the
worst month was 3.3602% and the global fraction was only 0.4391%.

V2 is a single, explicit **support-only feasibility repair**. It does not use
PnL, price after entry, funding after signal, CAGR, MDD, or 2023 data.

## Exactly what changes

1. A threshold needs at least 2,016 clean prior **calendar bars** in its 30-day
   window.
2. It separately needs at least 256 eligible prior route events before its q95
   can be estimated.
3. The monthly unavailable/quarantine ceiling is 5%; the global ceiling remains
   1%, and invalid periods plus the following 24 bars remain quarantined with
   no imputation.

Nothing else changes: crossed-clock formula, side confirmation, four policies,
6/18-bar holds, two-bar execution delay, leverage, costs, funding, strict MDD,
selection/holdout gates, controls, and portfolio orthogonality limits are
byte-for-byte equivalent in the manifest structures tested by
`tests/test_preregister_cross_venue_temporal_torsion_alpha_v2.py`.

## One-way boundary

After any 2020–2022 trade return is opened, no support, feature, threshold,
policy, or execution field may change. If v2 fails economic selection, the
family is rejected; there is no v3 retune on the same historical outcomes.

Parent support artifact:
`results/cross_venue_temporal_torsion_support_v1_2026-07-16.json`.
