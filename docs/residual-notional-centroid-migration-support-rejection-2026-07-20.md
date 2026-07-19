# RNCM-72 source-only support rejection

## Verdict

RNCM-72 was rejected before any OHLC, funding, forward return, PnL, equity,
CAGR, MDD, or post-2023 row was opened. No event clock was admitted and no
strict outcome evaluator may be run for this frozen candidate.

The preregistration was frozen in commit `8cdaa36` before real event incidence
was inspected.

## Mechanical control

All five fixed-absolute-book moving-band nulls produced zero raw and zero
non-overlapping events at every frozen support quantile:

- smooth symmetric constant density;
- tick-rounded anchor;
- stepped asymmetric spread;
- deterministic missing source rows;
- stationary asymmetric discrete tick ladder.

This confirms that the residual and quiet-center controls removed the specific
mechanical percentage-band shadow they were designed to remove. It does not
establish alpha.

## Frozen incidence result

| Quantile | Raw events | Non-overlap | Q1 | Q2 | Q3 | Q4 | Long share | Short share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.995 | 5 | 5 | 1 | 2 | 0 | 2 | 60.00% | 40.00% |
| 0.990 | 16 | 16 | 4 | 3 | 4 | 5 | 43.75% | 56.25% |
| 0.985 | 32 | 31 | 8 | 6 | 6 | 11 | 54.84% | 45.16% |
| 0.975 | 45 | 39 | 10 | 7 | 8 | 14 | 56.41% | 43.59% |

The frozen admission requirements were at least 120 total events, 45 in each
half, and 20 in every quarter. Even the loosest permitted threshold reached
only 39 total, 17 in H1, 22 in H2, and failed every quarter minimum. The
stopping rule therefore selected no quantile.

## Artifact identity

- support artifact:
  `results/residual_notional_centroid_migration_support_2026-07-20.json`
  - SHA-256: `887c532eb3163cfac47eb9fc2956326f02491b2890e4c0231e084807978577dc`
- frozen preregistration source SHA-256:
  `733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f`
- frozen preregistration document SHA-256:
  `fb2ed44cb0eb561c1d436c02c73d4028680b7ba292f67d8bce86ddf3ed23a11f`

## Interpretation

The raw average-quote migration idea was mechanically contaminated; the
deconfounded residual version became too sparse for meaningful inference.
Lowering the threshold, reducing the quiet-center requirement, weakening
coherence, or shrinking the 6-hour hold after seeing these counts would be a
post-support repair and is prohibited under `RNCM-72`.
