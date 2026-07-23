# FCCM-72 source-support result — terminal rejection

## Decision

**FCCM-72 is retired unchanged at the source-support stage.** It did not
authorize comparator parsing, novelty evaluation, BTC/funding outcomes, or an
economic evaluator.

The frozen stopping rule forbids threshold, lookback, direction, hold, sponsor,
or scheduler repair after incidence. This result therefore closes the FCCM-72
identity rather than starting a parameter search.

## Reproducible artifacts

- source-only clock:
  `data/funding_currency_custody_mobility_consensus_2021_2023/fccm72_support_clocks_2021_2023.csv.gz`
  - SHA-256:
    `71180862d9dcc4d76e055c52fd72a2424ee12387a6b8062af8a9382675af3810`
  - rows: 760 across the primary and ten causal controls
- report:
  `results/funding_currency_custody_mobility_consensus_support_2026-07-23.json`
  - SHA-256:
    `0cc2f741a9c174f13a050d73df7c668bdb9776c9c431b89bfb88cd814f899266`
  - manifest hash:
    `f88cae078ce090702b3fb874695f3ec50eafef483d076784e882b3dfe56f09c6`

The run completed in 11.22 seconds with a 141,692 KiB maximum resident set.

## Source-only evidence

| Split | Accepted | LONG | SHORT | Max month share | Max gap | WBTC active/raw |
|---|---:|---:|---:|---:|---:|---:|
| Train 2021–2022 | 46 | 21 | 25 | `4/23` | `284/3` days | `145/618` |
| Selection 2023 | 11 | 4 | 7 | `3/11` | `1091/12` days | `53/439` |

Only 14 of 28 frozen support checks passed. Major failures were:

- fewer than 60 train and 24 selection entries;
- 2022 had only 14 accepted entries and 2022H2 had only two;
- selection had only two entries in H1 and no Q1 entry;
- selection WBTC sponsorship was `53/439`, below the frozen `1/5` lower bound;
- maximum month concentration and accepted-entry gaps failed in both splits;
  and
- train contained a nine-trade same-side run, above the maximum eight.

The mechanism did pass exact identity/non-overlap checks, distinct WBTC actor
floors, component-vote contribution floors, and vote-pattern diversity. Those
partial passes do not rescue the failed distribution and sponsorship gates.

## Source and outcome boundary

- Bitfinex value rows read: 70,116 from the frozen 2020–2023 file.
- WBTC value rows read: 993 from the frozen finalized source.
- post-2023 source value rows read: zero.
- comparator value rows read: zero.
- BTC market, realized funding, future-return, PnL, CAGR, and MDD values read:
  zero.
- network calls: zero.

The next alpha search must use a new preregistered identity and economic
mechanism. FCCM-72 cannot be repaired, inverted, or gated into another result.
