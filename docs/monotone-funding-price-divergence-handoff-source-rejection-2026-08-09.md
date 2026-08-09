# MFDH-8 source rejection — 2026-08-09

## Decision

MFDH-8 is terminally rejected unchanged at the source-support gate. Gross9
structural clocks, execution prices, funding PnL, and post-entry outcomes were
not opened.

## Frozen lineage

- Preregistration SHA-256: `5e5b1a98c63c8ea8680f61379d5943dace6ed4b5ec524f1e3471d6cf7447eab1`
- Source evaluator commit: `3a69e322`
- Source-support result SHA-256: `cabe537dc3c13ce5eaf5ea0812880d90d9553b21ad15a52bffda43ef28c5a7e6`
- Primary clock SHA-256: `fe7fc61ba42af2c02d5be9dca4b90498058cae463b2a61ebe45fa59816945771`

## Source-only evidence

| Stage | Events | Long | Short | Minority share | Max month share |
|---|---:|---:|---:|---:|---:|
| train 2023H2 | 0 | 0 | 0 | 0.000 | 0.000 |
| test 2024 | 13 | 0 | 13 | 0.000 | 0.231 |
| eval 2025 | 8 | 1 | 7 | 0.125 | 0.625 |
| final to 2026-08-01 | 1 | 0 | 1 | 0.000 | 1.000 |

The frozen minimum event requirements were `8/12/12/8`, minority-side share
was required to be at least `0.20`, and maximum monthly concentration could
not exceed `0.45`. Train and final event floors failed, all nonempty stages
failed side balance, and eval/final failed monthly concentration.

## Reproducibility and stopping rule

An immediate complete DB rerun reproduced the result, primary clock, feature
source, manifest, and all four diagnostic-control files byte-for-byte. No
funding path, rank threshold, history, side, hold, subset, RV20 definition, or
diagnostic control was changed or promoted. The candidate therefore cannot
advance to Gross9 novelty or economics.
