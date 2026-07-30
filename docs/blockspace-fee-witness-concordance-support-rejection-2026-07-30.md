# BFWC-288 source-support rejection — 2026-07-30

## Decision

`BFWC-288` is **retired unchanged before comparator or market outcomes**.
The exact fee-curve/witness/fullness concordance clock passed source integrity,
join, append-invariance, total selection support, side balance, and most
calendar-dispersion checks, but failed three preregistered source-support
floors. The policy may not be repaired by lowering a count floor, changing the
rank threshold, dropping a sign, widening the month-share limit, changing the
hold, or reopening a component control as the primary.

## Frozen sequence

The singleton policy and write-once preregistration were committed and pushed
before exact candidate incidence:

- policy/test commit: `f1ecc733`;
- canonical preregistration commit: `b2cf4198`;
- source-support evaluator commit: `d779b1f6`;
- preregistration artifact SHA-256:
  `c255cccbda22cdc8c43e35f04f5d1792f0a76f88caa966434b5be79bff1f65f7`;
- preregistration `manifest_hash`:
  `499bdcd199bfe8ae7dad9bf5e51271f8fb1fd762edbaf4a1e0d026708e9fdf9b`.

The committed evaluator was then run once against the two frozen source
artifacts. It exact-joined 2,093 common 12-hour rows and produced 2,091
base-valid rows. There were zero exact join gaps, and both completed-prefix
append-invariance checks passed byte-for-byte.

## Source-only incidence

The exact primary produced 100 raw candidates and 94 accepted,
split-contained, globally non-overlapping entries.

| Window | Total | LONG | SHORT | Maximum month share |
|---|---:|---:|---:|---:|
| Selection | 49 | 27 | 22 | 18.37% |
| Future 2025 | 28 | 11 | 17 | 17.86% |
| Future 2026 | 17 | 12 | 5 | 35.29% |

The failed frozen checks were:

1. selection 2023 November–December: `3 < 6`;
2. future 2025 total: `28 < 30`; and
3. future 2026 maximum month share: `6 / 17 = 35.29% > 30%`.

All other support checks passed, including selection total and both-side
floors, both 2024 half-year floors, both 2025 half-year and side floors, the
2026 total/Q1/April–May/side floors, exact join integrity, and future append
invariance. Near misses do not authorize repair.

## Outcome boundary

Because source support failed, the evaluator stopped before novelty:

- BFRT comparator rows loaded: `0`;
- WCTR comparator rows loaded: `0`;
- Gross9 rows loaded: `0`;
- market rows loaded: `0`;
- funding rows loaded: `0`;
- premium rows loaded: `0`;
- return rows loaded: `0`;
- outcome columns loaded: `0`;
- outcomes computed: `false`.

No exact-entry Jaccard, tolerant containment, signed-exposure correlation,
BTC return, funding cash flow, PnL, CAGR, strict MDD, component-control
performance, or same-gross Gross9 marginal result was opened.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| Support report | `1d7af687d4f0469ff1688d123e9e83ea957a5d9b51fa3617ab16c4c43978e22c` |
| Primary clock | `b125046a1a3defda960e51b42e03ee1c3bb72a0799c646d3ae16a3e692735ed1` |
| Control clocks | `de09979da981c91f91c8c0c57270df72bdc1d5fb6d344b7a783280774b5e3a9d` |
| Support `manifest_hash` | `0557d542597a7dcc5d195c1bd51f8a8ba8828dbce6015d2be7b07542dae9d56d` |

The terminal decision is
`retire_BFWC_288_unchanged`. These artifacts remain useful only as a
source-incidence and non-repair record.
