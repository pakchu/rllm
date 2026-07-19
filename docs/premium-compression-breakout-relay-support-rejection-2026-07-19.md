# PCBR-12 source-only support rejection — 2026-07-19

## Verdict

`PCBR-12` was rejected before any BTCUSDT execution price, return, funding cash
flow, or strategy PnL was opened. The frozen support contract required every
support and novelty gate to pass. Two gates failed:

1. 2023 test support produced **56 events**, below the preregistered minimum of
   **60**.
2. Within the explicit common observable window for `CMSR-36`
   (`2020-08-01` through `2024-01-01`, end-exclusive), **93 of 283** PCBR events
   occurred within six hours of a CMSR event: **32.86%**, above the frozen
   **25%** containment maximum.

No threshold, direction, latency, holding period, or support minimum was changed
after observing these source-only results. Train, test, and eval outcomes remain
sealed.

## Outcome-blind support summary

| Split | Events | Long | Short | Largest month share | Gate |
|---|---:|---:|---:|---:|---|
| Train, 2020-03 through 2022 | 353 | 192 | 161 | 10.20% | pass |
| Test, 2023 | 56 | 31 | 25 | 14.29% | **fail: 4 events short** |
| Eval, 2024 through 2026-H1 | 163 | 73 | 90 | 13.50% | pass |

All preregistered subperiod, side-balance, and month-concentration checks passed.
The rejection is specifically the 2023 event-count gate plus the corrected
coverage-normalized CMSR near-overlap gate.

## Novelty audit

| Comparator | Common window | Exact Jaccard | Near window | PCBR near share | Gate |
|---|---|---:|---:|---:|---|
| PSR-30/6 | 2020-03 to 2026-07 | 0.0000 | 60m | 1.22% | pass |
| PSI-2016 | 2020-03 to 2026-07 | 0.0014 | 60m | 11.19% | pass |
| PSI-8640 | 2020-03 to 2026-07 | 0.0008 | 60m | 4.90% | pass |
| CMSR-36 | 2020-08 to 2024-01 | 0.0000 | 360m | **32.86%** | **fail** |

The first implementation incorrectly divided CMSR-near events by all 572 PCBR
events, including 2024–2026 events outside CMSR's observable horizon. Independent
review identified the dilution. The final builder uses explicit comparator
coverage and a regression test that prevents recurrence.

## Reproducibility and sealing evidence

- Premium-only source rows loaded: `3,417,120`
- Derived five-minute rows: `683,424`
- BTC execution rows loaded: `0`
- Funding rows loaded: `0`
- Primary clocks: `572`
- Primary clock SHA-256:
  `659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda`
- Support report SHA-256:
  `de41852acb7987685d31a799eddf56a7e59afa756f5435ce46e054ea72f83857`
- Support manifest hash:
  `93a66abd201ecbc244ff3776da03d976758fbb76c14154b000ca32d832e3fe8f`

Two consecutive builds produced byte-identical report, primary clock, and all six
control-clock files. Peak RSS was approximately 1.45 GiB. The next candidate must
be an independently preregistered mechanism; PCBR may not be repaired using these
sealed support observations.
