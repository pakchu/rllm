# Weekend FX reconciliation alpha — one-shot result

## Verdict

**Rejected.** The frozen `fx_event_z - btc_event_z` catch-up direction lost in
fit and in the one-shot 2023 inspection. No 2024+ outcome was opened.

The protocol was committed before outcome access in `7e29426`. Support-only
preflight and an independent critic both cleared the experiment for a single
pre-2024 outcome run. No threshold, direction, entry or hold was changed after
2023 became visible.

## Fixed primary result

All returns include the complete declared calendar window, including idle
time. Replay uses 0.5x leverage, 6 bp per side, next-open entry and conservative
favorable-first/adverse-second strict OHLC MDD.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2020-06..2022 | -45.36% | -20.85% | 46.58% | -0.45 | 112 | 51 / 61 |
| 2023 | -12.18% | -12.19% | 15.89% | -0.77 | 52 | 22 / 30 |
| 2023 H1 | -13.07% | -24.62% | 15.67% | -1.57 | 26 | 9 / 17 |
| 2023 H2 | +1.02% | +2.04% | 11.49% | 0.18 | 26 | 13 / 13 |

Zero implementation cost did not rescue the hypothesis:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD |
|---|---:|---:|---:|---:|
| Fit, 0 bp/side | -41.57% | -18.77% | 43.43% | -0.43 |
| 2023, 0 bp/side | -9.40% | -9.41% | 14.77% | -0.64 |

## Frozen controls

- BTC weekend continuation was the least-bad component: fit
  `+19.99% / 7.31% CAGR / 23.97% MDD / 0.30`, and 2023
  `+10.75% / 10.75% / 13.60% / 0.79`. It is far below promotion strength and
  had negative fit half-years.
- FX reopen alone lost in fit and 2023.
- Exact primary direction flip produced fit ratio `0.88` and 2023 ratio `0.36`;
  2023 H2 was negative. It is not a stable inverse alpha.
- Previous closure's side produced fit ratio `2.24` and 2023 ratio `1.91`, but
  2023 H1 was negative and the fixed threshold was not met. Because this was a
  predeclared placebo observed after outcome access, it cannot be promoted or
  tuned on this sample.
- Minute-05/10/15 entries all lost in fit and 2023.
- Fixed 12/24/48-hour holds all lost in fit and 2023.

## Mechanism audit

- Spearman(`residual`, `btc_event_z`) = `-0.533`.
- Spearman(`residual`, `fx_event_z`) = `+0.688`.
- Spearman(`btc weekend`, later `FX reopen`) = `+0.120`.
- Primary side agreement was `31.7%` with BTC continuation and `73.8%` with
  FX-only direction.

The residual is not an algebraic clone under the fixed 0.85 correlation gate,
but economic direction is decisively wrong and no component policy meets the
admission standard.

## Leakage and data limits

- BTC and FX analysis frames are returned strictly before `2024-01-01`.
- The source loader may physically read and immediately discard rows from a
  cutoff-crossing chunk; discarded rows never enter returned frames or
  computation.
- Each FX event requires all six minute-59-complete pairs after an observed
  45–72-hour Sunday/Monday closure gap.
- Event z-scores use `shift(1)` and only prior closure events. Earlier 2023
  feature states may update later 2023 states, but no outcome updates them.
- Signal at minute 00 enters at minute-05 open; exits are split-contained.
- The historical FX cache lacks bid/ask/mid provenance and publication or
  ingestion timestamps. Five-minute delay is conservative for the replay but
  is not proof of live feed latency.
- Historical perpetual funding is not included, so live promotion was blocked
  even before the statistical rejection.

## Frozen conclusion

Do not tune the FX pair basket, closure range, online normalization, residual
sign, direction, entry delay or hold on this inspected sample. Retain only the
raw closure-event representation as weak research context for a materially
different preregistered learner or genuinely fresh-forward shadow period.

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python -m training.search_weekend_fx_reconciliation_alpha
```
