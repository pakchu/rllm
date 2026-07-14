# UMFR-36 support decision — 2026-07-14

## Decision

**PASS support-only preregistration.** UMFR-36 may advance to clock freezing and
pre-2024 outcome evaluation. No UMFR post-entry returns, held-path OHLC, CAGR,
MDD, or 2024+ rows were opened for this decision.

- result artifact: `results/um_forced_flow_reversion_support_2026-07-14.json`
- result SHA-256: `22a6cabe015020fa427a660d611917590a95d1212426be1d391476ddce77d3ba`
- deterministic rerun: same SHA-256 on immediate rerun
- selected quantile: `0.80`

## Selected support summary

| Metric | Value |
| --- | ---: |
| Eligible primary events | 5,267 |
| Raw primary events | 1,070 |
| Non-overlapping scheduled events | 915 |
| 2020 events | 144 |
| 2021 events | 222 |
| 2022 events | 270 |
| 2023 events | 279 |
| 2023 H1 / H2 | 116 / 163 |
| 2023 Q1 / Q2 / Q3 / Q4 | 45 / 71 / 99 / 64 |
| Long / short share | 47.87% / 52.13% |
| Active months with at least five events | 46 |

Per-year side balance:

| Year | Long | Short |
| --- | ---: | ---: |
| 2020 | 68 | 76 |
| 2021 | 98 | 124 |
| 2022 | 136 | 134 |
| 2023 | 136 | 143 |

## Novelty versus prior clocks

| Prior clock | Jaccard | UMFR containment |
| --- | ---: | ---: |
| CSPR-12 | 0.000000 | 0.000000 |
| RIFT-96 | 0.002187 | 0.003279 |
| CATCH-12 | 0.000411 | 0.002186 |
| LURI-48 | 0.000000 | 0.000000 |
| CLASP-24 | 0.000000 | 0.000000 |

## Boundary

This is not a profitability claim. It only says the late USD-M forced-flow
reversion mechanism has enough pre-2024 support, side balance, era balance, and
novelty to justify a frozen clock and then a pre-2024 return evaluation.
