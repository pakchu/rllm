# ARCR-864 — outcome-blind support rejection

## Decision

**Reject ARCR-864 before market evaluation.** The frozen Address Reservoir
Capacitance Release singleton failed source-only event support by a wide margin.
No BTC bar, funding row, post-entry return, held path, PnL, CAGR, or MDD was
loaded, and no event-clock artifact was written.

The rejected rule must not be reopened by reducing its thresholds, changing
its side map, repeating persistent states, shortening its hold, loosening its
support minima, or selecting a more favorable calendar. Those would be
post-incidence repairs.

## Frozen evidence chain

- mechanism decision: commit `1dc8b32`;
- source loader: commits `f0b846a` and `efa1035`;
- support preregistration: commit `4506ddc`;
- preregistration artifact hash:
  `8207533dd43f1be29acc4aa70eacc45a95b0ae4c444509cd85b6090d4c32bf66`;
- source gzip SHA-256:
  `15550072f954d29ae4c9ffe16e11f07c492ee5b6b956e54654b14b9a7af5170a`;
- source manifest hash:
  `19d4fcd74fe628c6ecb6d051bc6f0966bcf145263c4c63a4520ceec4f6dc37fe`;
- committed source-manifest file SHA-256:
  `f16827d26a1e095e504623c24f94cfd77af36d4466439cf203c4bf0f72ddad97`;
- support-result file SHA-256:
  `524991e4770719bbbee51c8e9162a335077013258dcb47aa3a7c2ea0e1105e87`.

The local ignored source gzip is 29,396 bytes. Raw API pages were not
persisted.

## Source audit

The source itself passed every frozen quality check:

- exactly 1,826 unique daily observations;
- first 2019-01-01 and last 2023-12-31;
- one response page containing 1,826 rows;
- no missing or duplicate date;
- maximum observation gap one day;
- exact `AdrBalCnt`, `AdrActCnt`, and availability fields; and
- zero 2024+ source, market, funding, return, or PnL rows.

The rejection is therefore candidate sparsity and dispersion, not incomplete
source history.

## Candidate incidence

After current-row freshness, neutral-to-state transition, one completed
five-minute latency bar, full split containment, and greedy three-day
nonoverlap:

| Frozen support item | Observed | Minimum | Result |
|---|---:|---:|---|
| Total 2021H2–2023 | 38 | 90 | fail |
| Train 2021H2–2022 | 13 | 55 | fail |
| Train 2021H2 | 4 | 15 | fail |
| Train 2022 | 9 | 30 | fail |
| Test 2023 | 25 | 30 | fail |
| Test 2023H1 | 14 | 12 | pass |
| Test 2023H2 | 11 | 12 | fail |
| Active months | 19 of 30 | 25 | fail |
| Maximum month share | 13.16% | at most 15% | pass |

Quarter counts were:

```text
2021Q3 3    2021Q4 1
2022Q1 1    2022Q2 1    2022Q3 3    2022Q4 4
2023Q1 8    2023Q2 6    2023Q3 7    2023Q4 4
```

Every quarter required at least five events, so six of ten quarters failed.

Direction shares were:

| Window | Long | Short | Frozen minimum each side |
|---|---:|---:|---:|
| All | 52.63% | 47.37% | 25% |
| Train | 23.08% | 76.92% | 25% |
| Test | 68.00% | 32.00% | 25% |

Train direction balance also failed. Before split containment and nonoverlap,
the source produced 95 nonzero state days and 62 transition onsets; only 38
became accepted executable support events. There were 1,638 rows with all
three finite causal z-scores.

## Closed outcome boundary

The support result records `outcomes_opened=false`, zero market/funding rows,
zero return/PnL fields, and zero post-2023 source rows. Because support failed:

- the event-clock path is null and no event-clock file exists;
- no strict evaluator is authorized;
- no absolute return, CAGR, strict MDD, trade PnL, or control performance can
  be reported; and
- the next candidate must use a genuinely different observable/mechanism,
  rather than repairing ARCR-864 after seeing its incidence.
