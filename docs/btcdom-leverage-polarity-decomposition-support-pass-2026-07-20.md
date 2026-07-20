# DLPD-12 source-only support pass — 2026-07-20

## Verdict

**PASS.**  DLPD-12 passed every frozen annual incidence, side-balance,
quarter-breadth, month-concentration and one-hour clock-novelty gate.  This
authorizes freezing a strict 2022 evaluator; it is not evidence of profitable
BTC trading.

No BTC execution OHLC, return, funding cash flow, label, PnL, equity, CAGR,
MDD, or post-2023 DLPD source row was loaded.

## Primary source incidence

| Year | Events | Long / short | Max month share | Quarter counts |
|---|---:|---:|---:|---|
| 2022 | 237 | 122 / 115 | 12.24% | 54 / 71 / 51 / 61 |
| 2023 | 184 | 122 / 62 | 16.30% | 51 / 59 / 44 / 30 |

Frozen minima were 120 events per year, 25% on each side, 20 events in every
quarter, and a 20% maximum annual month share.  All passed without modifying
the disclosed singleton.

## Source controls

| Clock | 2022 events | 2023 events |
|---|---:|---:|
| primary | 237 | 184 |
| BTC premium tail only | 566 | 499 |
| BTCDOM premium tail only | 530 | 488 |
| same-sign extreme | 240 | 222 |
| stale BTC by one hour | 233 | 188 |
| stale BTCDOM by one hour | 242 | 203 |

The component and staleness clocks remain controls.  Similar incidence does
not prove equivalent outcomes and cannot be used to repair the primary after
market results open.

## Frozen 2023 novelty

| Comparator | Exact Jaccard | Primary near share, ±1h | Comparator near share, ±1h | Max |
|---|---:|---:|---:|---:|
| PSR-30/6 | 0.0000 | 5.98% | 5.19% | 5.98% |
| PCBR-12 | 0.0000 | 1.09% | 3.57% | 3.57% |
| OPDR-24 | 0.0000 | 1.09% | 5.71% | 5.71% |
| CLD-72 | 0.0000 | 2.17% | 3.77% | 3.77% |
| FCIR-12 | 0.0000 | 0.00% | 0.00% | 0.00% |

The preregistered maxima were `0.10` exact Jaccard and `0.35` maximum
bidirectional one-hour containment.  The result is well inside both limits,
including against same-source BTC premium path clocks and alt leadership/flow
clocks.

## Reproducibility and boundary evidence

- Source rows loaded: `21,912`
- Valid source rows: `21,744`
- Comparator clock rows loaded: `2,438`
- All-control DLPD clock rows: `3,832`
- Primary DLPD rows: `421`
- BTC execution rows loaded: `0`
- Funding rows loaded: `0`
- Post-2023 source rows loaded: `0`
- Support builder SHA-256:
  `f662128d5f0a09f8ed80182791df6328ca12efe10e8f8df9682b907d33fb5a23`
- Clock SHA-256:
  `b33990f1629465caa837aa1f6f74430054b7185b68ece47b8c7540f9c11bf0fb`
- Support report SHA-256:
  `1107694d5ff304aabaabbb962e9aeeaa64075001e494a0432dba3261ceace4f6`
- Support manifest hash:
  `037cfba9851a001bbad29fdd9d96bed22354204defa9f2d5788cc51d53528c90`

Two consecutive builds were byte-identical.  The next step is to commit the
strict evaluator before opening 2022 outcomes.  Threshold, side, hold, latency,
support gates and comparators remain frozen.
