# Nonlinear price/flow cross-map alpha — one-shot result

## Verdict

**Rejected as an executable alpha.** The frozen cross-map asymmetry policy had
near-zero fit risk efficiency and lost in the one-shot 2023 inspection. No
2024+ outcome was opened.

The protocol was committed before outcome access in `13e5eb8`. Support-only
preflight and an independent critic cleared one fixed pre-2024 run. No feature,
threshold, direction, entry, hold, cost or admission rule changed after 2023
became visible.

## Fixed primary result

All returns include the complete declared calendar window, including idle time.
Replay uses 0.5x leverage, 6 bp per side, minute-05 next-open entry, a fixed
12-hour hold, and favorable-first/adverse-second strict OHLC MDD.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2020-06..2022 | +4.89% | +1.86% | 19.63% | 0.09 | 348 | 160 / 188 |
| 2023 | -10.22% | -10.23% | 15.06% | -0.68 | 130 | 60 / 70 |
| 2023 H1 | -1.71% | -3.41% | 6.18% | -0.55 | 55 | 25 / 30 |
| 2023 H2 | -7.48% | -14.30% | 11.88% | -1.20 | 74 | 36 / 38 |

The mean-trade test does not support edge: fit mean `+0.0212%`, approximate
`p=0.747`, effect size `d=0.017`; 2023 mean `-0.0806%`, `p=0.172`,
`d=-0.120`.

Zero implementation cost did not rescue generalization:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit, 0 bp/side | +29.25% | +10.44% | 18.81% | 0.55 | 348 |
| 2023, 0 bp/side | -2.93% | -2.94% | 11.32% | -0.26 | 130 |
| 2023 H1, 0 bp/side | +1.59% | +3.24% | 5.10% | 0.63 | 55 |
| 2023 H2, 0 bp/side | -3.28% | -6.40% | 9.55% | -0.67 | 74 |

## Frozen controls and diagnostics

Every fixed structural control lost in 2023:

| Policy | Fit abs. return | Fit ratio | 2023 abs. return | 2023 ratio |
|---|---:|---:|---:|---:|
| Primary | +4.89% | 0.09 | -10.22% | -0.68 |
| Same events, flow follow | -4.12% | -0.07 | -9.52% | -0.66 |
| Same events, flow fade | -34.80% | -0.40 | -6.00% | -0.61 |
| Same events, price follow | -27.99% | -0.36 | -9.62% | -0.66 |
| Same events, price fade | -13.18% | -0.20 | -5.89% | -0.56 |
| Ordinary linear lead/lag | +13.04% | 0.21 | -13.75% | -0.67 |
| Exact direction flip | -40.40% | -0.43 | -5.27% | -0.57 |
| Signal delayed 6 hours | -14.06% | -0.16 | -8.39% | -0.61 |
| Signal delayed 7 days | -39.01% | -0.40 | -7.80% | -0.49 |

Minute-05/10/15 entry diagnostics all produced approximately `-10.2%` to
`-10.7%` in 2023. Fixed 6/12/24-hour holds produced 2023 absolute returns of
`-19.28%`, `-10.22%`, and `-5.67%`; none is a post-hoc replacement candidate.

## Representation audit

The nonlinear state itself passed the preregistered novelty screen:

- Spearman(cross-map dominance, linear lead/lag asymmetry): `-0.067`.
- Spearman(cross-map dominance, same-time price/flow correlation): `-0.118`.
- Primary-versus-linear event Jaccard: `0.143`.

This establishes representational difference, not predictive value or causal
direction. The static map `sign(asymmetry) * sign(current flow)` has no robust
edge. Raw cross-map skills and asymmetry may remain weak research tokens only
under a materially different preregistered learner; this inspected policy
family must not be tuned on the same sample.

## Leakage and source limits

- The returned analysis frame is strictly before `2024-01-01`; a gzip
  cutoff-crossing chunk may be decoded then immediately discarded. Discarded
  rows enter no returned frame, feature, hash, support count or outcome.
- Each state uses exactly the preceding 120 completed six-hour blocks. The
  current `[T-6h,T)` block supplies only completed minute-55 observables.
- The gate is strict `abs(dominance_t) > prior-only rolling q80`, with the
  threshold input shifted by one state.
- Signal at minute 00 enters minute-05 open. Trades are non-overlapping and
  exits remain inside each reported split.
- Full-source hashing was deliberately avoided; the artifact hashes only the
  returned pre-2024 frame.
- Cross-map asymmetry is not proof of causality in noisy or synchronized
  financial data.

## Frozen conclusion

Do not tune embedding dimension, library length, neighbors, Theiler radius,
gate quantile/lookback, sign mapping, delays or holds on this inspected sample.
Retain only the continuous cross-map reconstruction skills/asymmetry as weak
beta representation for a separately preregistered learner or a genuinely
fresh forward shadow period.

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python -m training.search_crossmap_flow_causality_alpha
```
