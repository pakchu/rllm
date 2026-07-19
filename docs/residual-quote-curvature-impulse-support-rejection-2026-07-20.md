# RQCI-24 source-only support rejection

## Verdict

RQCI-24 was rejected before OHLC, funding, entry/later return, PnL, equity,
CAGR, MDD, or any post-2023 row was opened. No event clock was admitted and no
outcome evaluator may be run for this exact candidate.

The source/mechanism decision was frozen in `5e62ab4`; the exact support policy
was frozen in `1feb39c` before real RQCI incidence was read.

## Mechanical gate

All five fixed-absolute-book controls produced zero raw and zero scheduled
events at all five frozen quantiles. The residual, dominance, and quiet-center
contracts therefore passed their intended moving-band null. This is only a
mechanical validity result, not return evidence.

## Frozen support result

| Quantile | Raw | Non-overlap | Q1 | Q2 | Q3 | Q4 | H1 | H2 | Long | Short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .995 | 3 | 3 | 1 | 0 | 0 | 2 | 1 | 2 | 33.33% | 66.67% |
| .990 | 10 | 8 | 1 | 1 | 1 | 5 | 2 | 6 | 50.00% | 50.00% |
| .985 | 29 | 23 | 3 | 4 | 5 | 11 | 7 | 16 | 47.83% | 52.17% |
| .975 | 66 | 50 | 8 | 8 | 17 | 17 | 16 | 34 | 44.00% | 56.00% |
| .950 | 192 | 148 | 14 | 36 | 45 | 53 | 50 | 98 | 47.97% | 52.03% |

Admission required at least 180 non-overlapping events, 70 per half, and 30 per
quarter. The loosest frozen threshold failed total count, H1, and Q1. The
stopping rule therefore selected no quantile.

## Artifact identity

- support artifact:
  `results/residual_quote_curvature_impulse_support_2026-07-20.json`
  - SHA-256: `aef0f375245cd340a29da121551dbf2ebf80ef3e08a4123ad7ef9c5e0c414f68`
- preregistration source SHA-256:
  `1d8e9db1794323e3576ea812673bed7c011ffa2634bc49ebf4d073b9c5ef8468`
- preregistration document SHA-256:
  `48a03266d118d0c68d0b36c84fc2d29cd781b153ec8f3487557aa3f87230f827`

## No repair

Shortening the hold, lowering the minimum counts, adding a lower quantile, or
weakening the quiet-center/dominance filters after seeing this table would be a
post-support repair. Those changes are prohibited under `RQCI-24`. A next
candidate must use a different observable mechanism rather than harvest this
near miss.
