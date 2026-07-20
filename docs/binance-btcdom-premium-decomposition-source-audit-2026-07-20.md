# Binance BTC/BTCDOM premium decomposition source audit — 2026-07-20

## Verdict

The frozen DLPD predictor prefix was built successfully from official Binance
Vision USD-M monthly `premiumIndexKlines` archives.  All sixty archives matched
the published hashes frozen before the archive build.  Two consecutive network
builds produced byte-identical panel and manifest files.

This artifact is **source-only**.  It contains no BTCUSDT or BTCDOMUSDT contract
OHLC, index level, mark price, return, funding, outcome, label, PnL, equity,
CAGR, MDD, or post-2023 predictor row.

## Physical source

- Symbols: `BTCUSDT`, `BTCDOMUSDT`
- Interval: completed UTC hour
- Grid: `2021-07-02 00:00` through `2023-12-31 23:00` UTC
- Exact grid rows: `21,912`
- Both-leg-valid rows: `21,744`
- BTCUSDT-valid rows: `21,768`
- BTCDOMUSDT-valid rows: `21,768`
- Current-hour values retained: only each leg's premium-index close
- Conservative availability: hour close plus `1.001` seconds

Official source roots and live parity are recorded in
`docs/btcdom-leverage-polarity-decomposition-source-decision-2026-07-20.md`.
The production builder re-fetched every adjacent `.CHECKSUM`, required it to
equal the committed inventory, then required the downloaded ZIP bytes to equal
that hash.

## Missing-row audit

Missing observations were retained on the grid and were never zero-filled,
carried forward, or interpolated.

| Month | BTC valid | BTCDOM valid | Missing hours |
|---|---:|---:|---:|
| 2021-07 | no | no | 96 |
| 2022-10 | no | no | 24 |
| 2023-02 | no | yes | 24 |
| 2023-04 | yes | no | 24 |

The source row is invalid whenever either member is absent.  A present member's
premium close remains available for component controls, but the primary DLPD
clock cannot use a one-sided row.

## Causal and semantic boundary

- Premium rows are keyed by bar-open time but unavailable until after their
  verified close time.
- Rolling normalization and event construction are not performed by this
  source builder; the preregistration must shift all reference distributions.
- The BTCDOM premium is perpetual pressure on an exchange-defined evolving
  relative index.  It is not relabeled as BTC dominance level, historical
  constituent composition, or absolute BTC direction.
- Current composite-index weights are not backfilled into historical rows.
- Archive placeholder volume/count fields and all premium OHLC fields other
  than close are discarded.

## Reproducibility

- Source decision SHA-256:
  `ed402ef2a91e400b29b902154646637987318d57ab0543312f383f1193be3cf6`
- Frozen checksum inventory SHA-256:
  `96240ba01d4cd5720eefdc05aa3a15d94f9c494118815219a9fb3442981f200e`
- Builder SHA-256:
  `9df2a645c96b17acd99477d3f0ad6b4abfbcedc01d8e965cd850717cac6463b9`
- Combined panel SHA-256:
  `75fb36b33810134746515e3ad99234e2a52f6f721551792788f6d3950ff5b1d9`
- Build manifest SHA-256:
  `885014743c299250c85cec42561db0dc99b09a60ecb1adfe893d8cac95651c05`

Artifacts:

- `training/build_binance_btcdom_premium_decomposition_source.py`
- `data/binance_btcdom_premium_decomposition_2021_2023/archive_checksums.json`
- `data/binance_btcdom_premium_decomposition_2021_2023/BTCUSDT_BTCDOMUSDT_premium_close_1h_2021-07-02_2023-12-31.csv.gz`
- `data/binance_btcdom_premium_decomposition_2021_2023/build_manifest.json`

The next authorized step is the already frozen source-only DLPD support and
novelty evaluation.  No profitability result is implied by this source audit.
