# Binance regional fiat-flow source audit (2026-07-17)

## Decision

Freeze a source-only daily panel for a new, orthogonal alpha family based on
BTC participation in three fiat-quote books. Pair activity does not establish
trader geography or an external fiat deposit:

- `BTCUSDT` — global stablecoin reference market,
- `BTCEUR` — EUR quote book,
- `BTCTRY` — TRY quote book,
- `BTCBRL` — BRL quote book.

This work unit opens **no BTC return, forward label, funding, futures OI, REX,
Kimchi, FX, volatility, or portfolio outcome**. It only establishes whether an
official, reproducible, live-available source exists before a strategy is
specified.

## Official sources

- Binance public-data repository and schema:
  <https://github.com/binance/binance-public-data>
- Official monthly Spot archive root:
  <https://data.binance.vision/?prefix=data/spot/monthly/klines/>
- Per-symbol archive pattern:
  `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1d/{SYMBOL}-1d-{YYYY-MM}.zip`
- Published checksum pattern: the archive URL plus `.CHECKSUM`
- Live public exchange metadata endpoint:
  <https://data-api.binance.vision/api/v3/exchangeInfo>

The upstream README documents the twelve kline fields used for schema
validation and the adjacent `.CHECKSUM` files used for SHA-256 verification. It
also states that Spot timestamps switch to microseconds from 2025-01-01. This
panel ends at 2024-01-01 exclusive, so the frozen parser intentionally accepts
milliseconds only.

## Frozen source contract

| Item | Frozen value |
|---|---|
| Source interval | official Spot monthly `1d` klines |
| Historical interval | `[2021-01-01, 2024-01-01)` UTC |
| Symbols | `BTCUSDT`, `BTCEUR`, `BTCTRY`, `BTCBRL` |
| Expected grid | every UTC date × all four symbols |
| Archives | 144 monthly ZIPs |
| Checksum policy | fetch each official `.CHECKSUM`, verify ZIP SHA-256 before parsing |
| Persisted source observables | BTC base volume, trade count, taker-buy BTC, taker-sell BTC, taker-buy fraction |
| Explicitly discarded | open/high/low/close prices and all quote-currency amounts |
| Missing-data policy | no fill, no stale carry; fail the build |
| Invalid-row policy | no quarantine/fallback; fail the build |
| Outcome policy | no return or trading label is computed or read |

The price and quote fields must be parsed to validate the official archive's
fixed twelve-column schema and volume bounds. They are discarded by
`source_panel()` and do not exist in the frozen artifact.

## Temporal availability contract

The artifact's `date` is the source bar's **UTC open time**, not its decision
availability time. All daily aggregates, including full-day base volume,
taker-buy volume, and trade count, become usable only after `close_time_ms`.
Therefore any future alpha builder must:

1. derive a signal only after the source day has closed;
2. execute no earlier than the next completed trading bucket/open after that
   close;
3. compute every rolling rank or normalizer from strictly prior completed days;
4. fail closed if any of the four markets is missing or incomplete.

Using row `date=d` to trade during day `d` would be future leakage and is
forbidden.

## Integrity result

Builder:
`training/build_binance_regional_fiat_flow.py`

Artifact:
`data/binance_regional_fiat_flow_btc_2021_2023/BTC_regional_fiat_flow_1d_2021-01-01_2023-12-31.csv.gz`

Manifest:
`data/binance_regional_fiat_flow_btc_2021_2023/build_manifest.json`

| Check | Result |
|---|---:|
| Expected rows | 4,380 |
| Observed rows | 4,380 |
| Complete rows | 4,380 |
| Missing date-symbol rows | 0 |
| Incomplete rows | 0 |
| First source date | 2021-01-01 UTC |
| Last source date | 2023-12-31 UTC |
| Panel SHA-256 | `c10ab68e9926c5e30a7ea70d3b54ee7468549b18513b67fc6c86eefa9f0a82c2` |
| Manifest SHA-256 | `772bd166f26f5f0047f71e89a83a65704640b44dbd3648ea0d6dfbb0210ee217` |

The manifest records the official URL and verified SHA-256 for every one of the
144 source archives. A second complete network rebuild produced byte-identical
panel and manifest files.

### Coverage by symbol and year

| Symbol | 2021 | 2022 | 2023 |
|---|---:|---:|---:|
| `BTCUSDT` | 365/365 | 365/365 | 365/365 |
| `BTCEUR` | 365/365 | 365/365 | 365/365 |
| `BTCTRY` | 365/365 | 365/365 | 365/365 |
| `BTCBRL` | 365/365 | 365/365 | 365/365 |

### Live parity check

On 2026-07-17, the official public `exchangeInfo` endpoint returned all four
symbols with:

- `status = TRADING`,
- `baseAsset = BTC`,
- the expected quote asset,
- `isSpotTradingAllowed = true`.

This verifies source availability, not exchange accessibility in any particular
jurisdiction and not alpha profitability.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m training.build_binance_regional_fiat_flow --workers 12

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_build_binance_regional_fiat_flow.py
```

## Scope boundary for the next work unit

The next commit may preregister exactly one fiat-quote breadth hypothesis.
No post-entry return may be opened before that preregistration freezes:

- the strictly-prior normalization window,
- breadth and taker-pressure conditions,
- episode de-duplication,
- availability shift and execution clock,
- side and hold,
- support-only selection rule,
- negative controls and pass/reject gates.

Source completeness alone is not alpha evidence. CAGR, absolute return, strict
MDD, and CAGR/MDD are intentionally **N/A** in this work unit.
