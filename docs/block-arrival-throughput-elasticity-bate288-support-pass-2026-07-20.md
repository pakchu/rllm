# BATE-288 source-only support pass — 2026-07-20

## Verdict

**PASS.** BATE-288 satisfies every preregistered source-integrity, side-balance,
calendar-dispersion, and outcome-blind event-count gate. This result authorizes
freezing one strict evaluator. It is not evidence of profitability.

No BTC price, funding, premium, OI, liquidation, order-book, FX, future return,
equity, MDD, or PnL field was loaded or opened in this stage.

## Frozen source evidence

- block interval: `610691..823785` inclusive;
- rows: 213,095;
- chain links checked: 213,094;
- source CSV SHA-256:
  `1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833`;
- source manifest SHA-256:
  `9b1c3a81d607632267fe4c87857b2e80d381d3bab90fca7bb0b7df0061775983`;
- source manifest hash:
  `80491a0688ae49dc5701eab479dba970a3085b50c14c7b1e3d23b25b4f82d2c1`;
- minimum header timestamp: `1577836985` (`2020-01-01T00:03:05Z`);
- maximum header timestamp: `1704066372` (`2023-12-31T23:46:12Z`);
- unique hashes, complete height range, cross-host anchors, and pre-2024
  containment: all passed; and
- market rows, funding rows, return/PnL fields, post-2023 source rows, and
  persisted raw responses: all zero.

The incomplete Blockstream-host checkpoint was not mixed into the completed
Mempool-host source. The completed run used the hash-frozen persistent
transport and the exact host/range contract committed before completion.

## Feature integrity

| check | result |
|---|---:|
| eligible packet endings | 213,083 |
| positive elapsed spans | 213,083 |
| invalid elapsed spans | 0 |
| positive-span ratio | 100.00% |
| maximum invalid run | 0 |
| finite robust-z states | 211,067 |
| HIGH state rows | 18,683 |
| LOW state rows | 9,579 |
| onsets before non-overlap | 11,110 |
| accepted 2021–2023 events | 971 |

The 2,016-packet strictly-prior robust reference, joint ±1.25 threshold,
six-confirmation/two-hour embargo, transition onset, and fixed 24-hour
non-overlap were not repaired after incidence became visible.

## Preregistered support gates

### 2021–2022 train clock

| window | total | HIGH | LOW |
|---|---:|---:|---:|
| combined | 645 | 387 | 258 |
| 2021 | 322 | 193 | 129 |
| 2022 | 323 | 194 | 129 |
| 2021 H1 | 160 | 93 | 67 |
| 2021 H2 | 162 | 100 | 62 |
| 2022 H1 | 159 | 90 | 69 |
| 2022 H2 | 164 | 104 | 60 |

Maximum train single-month share was 4.34%, below the frozen 15% limit.

### 2023 selection clock

| window | total | HIGH | LOW |
|---|---:|---:|---:|
| full year | 326 | 201 | 125 |
| H1 | 161 | 102 | 59 |
| H2 | 165 | 99 | 66 |
| Q1 | 79 | — | — |
| Q2 | 82 | — | — |
| Q3 | 82 | — | — |
| Q4 | 83 | — | — |

Maximum 2023 single-month share was 8.90%, below the frozen 20% limit. Every
support check passed without threshold, side, latency, packet, reference, or
hold repair.

## Frozen clock artifacts

- clock SHA-256:
  `cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02`;
- support JSON SHA-256:
  `42598a24853b1d66f2e91a259b2a23e5939a1d0a640abafb4e087e3f209caefc`;
- support result hash:
  `38fc7aa5435143b62035c0752a43206372feac5f7878e39c19195a611da57dcc`;
- first accepted entry: `2021-01-01T17:30:00Z`; and
- last accepted entry: `2023-12-31T21:35:00Z`.

A second source-only run reproduced the clock and support file hashes exactly.

## Next sealed step

Freeze and test one evaluator that consumes this exact clock, audited Binance
BTCUSDT five-minute next-open/held OHLC, and exact funding. Then open only
2021–2022 train. A train failure rejects BATE-288 before 2023 selection is
opened. Every performance report must include absolute return, full-calendar
CAGR, strict MDD, CAGR/strict-MDD, and trade count.
