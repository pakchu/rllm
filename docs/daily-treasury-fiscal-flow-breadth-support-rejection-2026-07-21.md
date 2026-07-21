# DFFB-601 source-only support rejection

## Verdict

DFFB-601 is permanently rejected before any BTC OHLC, funding, forward return,
PnL, equity, CAGR, or MDD was opened. The frozen primary clock had adequate
incidence and low signed-exposure correlation with prior strategies, but it
failed the preregistered decision-date novelty gate against three source-only
comparators. The stopping rule therefore prohibits an outcome evaluation or an
in-place repair of this candidate.

The support evaluator and its synthetic tests were frozen in commit `e7ffddb`
before real DFFB incidence was computed.

## Frozen support result

The primary 24-hour non-overlapping clock produced 112 events:

| Window | Events | Long | Short |
|---|---:|---:|---:|
| Train 2021 | 36 | - | - |
| Train 2022 | 40 | - | - |
| Train total | 76 | 44 | 32 |
| Selection 2023 H1 | 16 | - | - |
| Selection 2023 H2 | 20 | - | - |
| Selection total | 36 | 17 | 19 |

Every frozen support check passed, including total count, half/year coverage,
both directions, and maximum calendar-month concentration. All six source-only
controls also passed their support floors. Support establishes only that the
clock is testable; it does not establish novelty or alpha.

## Failed novelty gate

The gate required every comparator to satisfy both decision-date Jaccard
`<= 0.30` and the fraction of DFFB dates within plus or minus one frozen U.S.
federal business day `<= 0.50`.

| Comparator | Comparator dates | Jaccard | DFFB dates within +/-1 business day | Result |
|---|---:|---:|---:|---|
| DTS total-net-cash control | 388 | 0.1547 | 102/112 = 91.07% | Fail |
| FLCC primary-clock union | 169 | 0.0604 | 58/112 = 51.79% | Fail |
| Official auction/settlement calendar | 617 | 0.0519 | 80/112 = 71.43% | Fail |

The four individual FLCC candidates and TADI primary clock passed. That does
not rescue DFFB-601: the preregistration requires every non-empty comparator to
pass, and no comparator may be dropped after observing incidence.

## Exposure orthogonality

Signed occupied-exposure correlations on the complete five-minute UTC grid all
passed the absolute `0.40` limit:

| Comparator | Absolute Pearson correlation |
|---|---:|
| FLCC-H4-Q60 | 0.0220 |
| FLCC-H4-Q65 | 0.0153 |
| FLCC-H8-Q60 | 0.0489 |
| FLCC-H8-Q65 | 0.0528 |
| TADI primary | 0.0112 |

This shows that accepted position occupancy was not a close replica of those
prior clocks. It does not override the failed date-novelty gate.

## Outcome boundary

The support run read 205,589 frozen DTS source-value rows, derived 1,255 report
features and 112 primary events, read 3,208 comparator-clock rows, and parsed
the hash-bound 445-row auction panel plus 4,000 raw auction rows. It made no
database, network, or subprocess calls and loaded zero market, funding, return,
or PnL rows. The result records `performance_values_opened = false`.

## Artifact identity

- support decision:
  `results/daily_treasury_fiscal_flow_breadth_support_2026-07-21.json`
  - SHA-256: `a5bf3b15f40f05d876b7603eaa3104cfa21a867fa3dd1aa4681b6b0875c8f549`
  - manifest hash: `05f2de6ab8982671d4adcf44ca7e77a25fd5aa9b0a33e840cce0a34efe2ab36c`
- primary clock:
  `results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz`
  - SHA-256: `df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978`
- control clocks:
  `results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz`
  - SHA-256: `416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312`
- frozen support builder:
  `training/build_daily_treasury_fiscal_flow_breadth_support.py`
  - SHA-256: `7d1eaaf0bcd8159811256c59781c14df08e47376cfc3a362896c416f7b419796`

A clean second source-only run produced byte-identical primary and control
clocks and a semantically identical decision artifact after removing only the
alternate output paths and their path-dependent manifest hash.

## Interpretation and stopping rule

DFFB-601 had enough balanced events and genuinely low position-overlap with
the prior strategies, but its event dates remained too tightly associated with
broad Treasury cash-flow and auction timing. Changing breadth thresholds,
dropping failed comparators, moving the decision date, or altering the hold
after seeing this result would be a post-support repair. Any future fiscal-flow
idea must receive a new candidate identity and a new preregistration; DFFB-601
itself may not proceed to outcomes.
