# Binance regional fiat cross-rate source audit — 2026-07-20

## Decision

Freeze the complete source-only daily-close panel for **RFXS2-576**. The source
passes its exact byte, calendar, schema, checksum, and causality contract.

This is not alpha evidence. No residual, z-score, event, comparator statistic,
execution return, funding cash flow, PnL, CAGR, or MDD was computed in this work
unit.

## Why this is v2

The first RFXS-576 build failed closed because the BTCBRL October 2020 archive
began on October 13 rather than October 1. That candidate was retired in commit
`8ff99fec6f100537c260df8b1d484c32ebf56d8d` without producing a source panel or
opening an outcome.

RFXS2-576 was separately preregistered in commit
`263426aad67b2ca5fdc408f62a64d970d35fdd43` with the next exact full-month
boundary, 2020-11-01. It inherits the original mechanism without changing the
feature, direction, threshold, support gates, controls, or later performance
gates.

## Frozen source contract

| Item | Frozen value |
|---|---|
| Candidate | `RFXS2-576` |
| Source | official Binance Spot monthly `1d` kline ZIPs and companion checksums |
| Horizon | `[2020-11-01, 2024-01-01)` UTC |
| Symbols | `BTCUSDT`, `BTCEUR`, `BTCTRY`, `BTCBRL` |
| Grid | every UTC date, complete for all four symbols |
| Source archives | 152 ZIPs: 38 months × 4 symbols |
| Timestamp unit | milliseconds in every frozen archive |
| Retained signal values | the four completed daily closes only |
| Validation-only values | open/high/low, volume, trade count, taker fields |
| Missing-data rule | no fill, stale carry, splice, quarantine, or fallback |
| Availability | not before the following UTC day at 00:00 |
| Outcome boundary | no USD-M OHLC, funding, return, PnL, CAGR, or MDD read |

Every checksum record was bound to the exact requested ZIP filename. The
manifest retains the checksum-response hash, published archive hash, local
archive hash, URL, timestamp unit, row count, and first/last day for every ZIP.
All 152 published and locally computed archive hashes matched.

## Frozen artifacts

Builder:

```text
training/build_binance_regional_fiat_cross_rate.py
commit 8259a5121959ea92735a4340d74af19dfd1786a0
SHA-256 2ce3e8f1a0d5c134d120cc1720cd14a81e9c417f79516568c33ebb038a035a87
```

Panel:

```text
data/binance_regional_fiat_cross_rate_btc_2020-11_2023/
  BTC_regional_fiat_cross_rate_1d_2020-11-01_2023-12-31.csv.gz
SHA-256 5dbc697c8299ac892295a01302e9f2d883a6e252c8d3d85a8f60f3a369b533d3
size 25,995 bytes
```

Manifest:

```text
data/binance_regional_fiat_cross_rate_btc_2020-11_2023/build_manifest.json
SHA-256 627fdd8298312ea61c2bfaa14d93d623e61d562d64abea0a3769d79c3a68673c
size 110,170 bytes
```

The manifest also binds the original mechanism, RFXS-576 source rejection, and
RFXS2-576 successor documents by exact commit, path, and SHA-256.

## Integrity result

| Check | Result |
|---|---:|
| Expected source days | 1,156 |
| Observed source days | 1,156 |
| Complete source days | 1,156 |
| Raw validated date-symbol rows | 4,624 |
| Missing date-symbol rows | 0 |
| Duplicate date-symbol rows | 0 |
| Non-finite/non-positive retained closes | 0 |
| Availability-boundary mismatches | 0 |
| Published/local archive hash mismatches | 0 |
| First day | 2020-11-01 |
| Last day | 2023-12-31 |
| Execution/funding/outcome rows opened | 0 |

### Coverage by source book

| Symbol | Months | Days |
|---|---:|---:|
| `BTCUSDT` | 38 | 1,156 |
| `BTCEUR` | 38 | 1,156 |
| `BTCTRY` | 38 | 1,156 |
| `BTCBRL` | 38 | 1,156 |

## Reproducibility

Two complete network builds produced byte-identical panel and manifest files.
The measured second build took 6.10 seconds with 125,908 KiB maximum RSS. The
artifact directory occupies approximately 140 KiB, leaving total WSL disk use
at 292 GiB, below the user's 300 GiB ceiling.

The second build's complete stdout transcript and a machine-readable before /
after hash attestation are frozen as:

```text
results/rfxs2_source_rebuild_2026-07-20.log
SHA-256 98862eec4d7156946ffba2e5e33c9ab91948f4a7a31e126ecbd5acbd6e87440a

results/rfxs2_source_rebuild_attestation_2026-07-20.json
SHA-256 c1505a952487b9c4725f3a04fd8192312fa9011fe2610a6e0703fd61130c70a1
```

These files attest what this run observed; they are not a substitute for the
upstream bytes. Raw ZIP and checksum-response bodies are intentionally not
retained because the frozen panel prohibits non-close source values and the
project has an explicit disk ceiling. Therefore independent revalidation of
upstream provenance and validation-only OHLC/volume checks requires rerunning
the checksum-verifying builder. The retained manifest is sufficient to detect
any changed response or archive hash on that rerun.

Verification commands:

```bash
uv run python -m training.build_binance_regional_fiat_cross_rate

uv run python -m pytest -q \
  tests/test_build_binance_regional_fiat_cross_rate.py \
  tests/test_binance_regional_fiat_cross_rate_source_artifact.py \
  tests/test_build_binance_regional_fiat_flow.py
```

## Next sealed work unit

The next code may compute only the frozen RFXS2 residuals, strictly-prior
180-day robust z-scores, source states, globally reserved clocks, split-contained
support counts, and frozen source-only novelty diagnostics through 2023. It may
not import or open the USD-M execution panel, funding artifact, 2024+ source, or
any strategy outcome. Any failed source-support gate retires RFXS2-576 before a
strict evaluator exists.
