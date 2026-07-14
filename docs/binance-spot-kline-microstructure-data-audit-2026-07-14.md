# Binance Spot kline microstructure audit — 2026-07-14

## Verdict

**PASS with fail-closed quarantine.** The official Spot source is suitable for
a preregistered pre-2024 alpha experiment, but missing or malformed intervals
must never be filled. No return or future path was opened during this audit.

- builder commit: `2ae4e4b`
- audit artifact:
  `results/binance_spot_kline_microstructure_audit_2026-07-14.json`
- audit SHA-256:
  `2e2faf8d603d84519cd4a335b1c58d7bbe25e2bbeee1de50f725fd8d93288c59`
- source build-manifest SHA-256:
  `69fbce64b4860eecbf1ce414ea719b5c4001852016fe439e61240e050b39b57b`
- combined feature SHA-256:
  `d558239fa7085083aa002b7898b632df0774425719467709680ecb99718035a9`

## Coverage

| year | expected 5m rows | observed | missing | incomplete | clean |
|---:|---:|---:|---:|---:|---:|
| 2020 | 105,408 | 105,159 | 249 | 12 | 105,147 |
| 2021 | 105,120 | 104,923 | 197 | 20 | 104,903 |
| 2022 | 105,120 | 105,120 | 0 | 0 | 105,120 |
| 2023 | 105,120 | 105,104 | 16 | 15 | 105,089 |
| **total** | **420,768** | **420,306** | **462** | **47** | **420,259** |

Clean coverage is `99.8790%` of the complete expected UTC grid.

The 462 absent rows form 15 explicit outage ranges. A further 47 observed
five-minute rows fail closed: four have fewer than five source minutes and 44
contain one or more invalid source minutes; these categories overlap by one
row. Invalid rows include zero-activity exchange placeholders and malformed
`close_time` values. They are preserved for chronology but never admitted as
features.

## Integrity checks

- 48 official monthly ZIP checksums verified;
- raw archives retained only in memory;
- timestamps strictly ordered and duplicate-free inside each archive;
- non-finite or negative volumes rejected;
- taker-buy quantities bounded by total quantities;
- five-minute completeness requires five contiguous minute opens;
- buyer and seller execution centroids require positive denominators and must
  fall inside the five-minute high/low range with only a floating-point
  tolerance far below one tick;
- every invalid source minute, incomplete bucket, missing bucket, and derived
  non-finite centroid is quarantined;
- the artifact stops at `2023-12-31 23:55:00`; no 2024+ source or outcome is
  present.

## Downstream contract

The alpha layer must intersect this clean mask with the existing USD-M
aggTrade quarantine and official USD-M kline grid. A decision may use a
completed spot bucket at `t`; execution may begin only at the USD-M open of
`t+1`. Missing spot observations are not forward-filled. The first experiment
uses a fixed exit so no post-entry spot feature can retroactively change the
event clock.
