# CLASP-24 support decision — 2026-07-14

## Decision

**PASS support-only preregistration.** CLASP-24 may advance to clock freezing and
pre-2024 outcome evaluation. CLASP post-entry returns, held-path OHLC, CAGR, MDD,
and all calendar 2024+ rows remain unopened at this decision point.

- result artifact: `results/cash_late_arrival_spillover_propagation_support_2026-07-14.json`
- result SHA-256: `bd26905f7c33360a62c9eb14cef23ba917612e64fc5d83e47e25b50b56db8930`
- deterministic rerun: same SHA-256 on immediate rerun
- selected quantile: `0.75`
- selection rule: highest frozen causal event-score quantile passing all count,
  side-balance, era-balance, temporal-placebo, stale-clock, and prior-clock
  novelty floors.

## Selected support summary

At quantile `0.75`:

| Metric | Value |
| --- | ---: |
| Eligible primary events | 2,747 |
| Raw primary events | 649 |
| Non-overlapping scheduled events | 615 |
| 2020 events | 134 |
| 2021 events | 184 |
| 2022 events | 153 |
| 2023 events | 144 |
| 2023 H1 / H2 | 74 / 70 |
| 2023 Q1 / Q2 / Q3 / Q4 | 33 / 41 / 29 / 41 |
| Long / short share | 45.04% / 54.96% |
| Active months with at least five events | 44 |

Per-year side balance:

| Year | Long | Short |
| --- | ---: | ---: |
| 2020 | 53 | 81 |
| 2021 | 82 | 102 |
| 2022 | 79 | 74 |
| 2023 | 63 | 81 |

## Novelty evidence

Scheduled overlap against temporal and stale controls stayed inside frozen
floors:

| Control | Jaccard | CLASP containment |
| --- | ---: | ---: |
| early_cash | 0.000000 | 0.000000 |
| venue_swap | 0.000000 | 0.000000 |
| stale_1h | 0.000000 | 0.000000 |
| stale_24h | 0.001637 | 0.003252 |
| signal_delay_1bar | 0.000000 | 0.000000 |

Scheduled overlap against frozen prior alpha clocks stayed inside frozen floors:

| Prior clock | Jaccard | CLASP containment |
| --- | ---: | ---: |
| CSPR-12 | 0.008259 | 0.019512 |
| RIFT-96 | 0.001864 | 0.003252 |
| CATCH-12 | 0.000000 | 0.000000 |
| LURI-48 | 0.000000 | 0.000000 |

## Boundary

This document does not claim profitability. It only records that the feature
incidence and novelty tests are strong enough to open pre-2024 outcomes under
the already frozen return gate. The next irreversible analytical step is to
freeze the CLASP clock/manifest before any return evaluation.
