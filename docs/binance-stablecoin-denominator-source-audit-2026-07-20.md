# Binance stablecoin-denominator source audit — 2026-07-20

## Verdict

The initial source-only SDDR prefix passed its integrity gate. Fifteen official
Binance Spot monthly `1h` archives were downloaded, checked against their
published SHA-256 files, replayed against the previously frozen stablecoin-flow
archive manifest, and reduced to 3,592 simultaneous three-book observations.

No BTCUSDT perpetual OHLC, funding, future return, label, PnL, CAGR, MDD, or
post-2023 row was read. Source completeness and cross-price distributions are
not profitability evidence.

## Frozen source

| Item | Value |
|---|---|
| Symbols | `BTCUSDT`, `BTCUSDC`, `BTCFDUSD` |
| Interval | completed UTC `1h` Spot klines |
| Common range | `[2023-08-04 08:00, 2024-01-01)` |
| Official archives | 15 monthly ZIPs plus published checksums |
| Exact common rows | 3,592 |
| Missing/duplicate rows | 0 / 0 |
| Source availability | source hour start plus exactly one hour |
| Persisted raw BTC prices | none |
| Persisted flow/volume/count | none |
| Outcome or funding fields | none |

Official references:

- Binance public-data schema and timestamp policy:
  <https://github.com/binance/binance-public-data>
- Binance Spot monthly kline archive root:
  <https://data.binance.vision/?prefix=data/spot/monthly/klines/>

The builder is
`training/build_binance_stablecoin_denominator_source.py`. It validates the
complete upstream kline schema temporarily, requires exact active hourly grids,
and compares archive hashes and row counts with the frozen manifest
`data/binance_stablecoin_quote_flow_btc_2023_2026/build_manifest.json`.

## Persisted observables

```text
usdc_vs_usdt  = log(BTCUSDC_close / BTCUSDT_close)
fdusd_vs_usdt = log(BTCFDUSD_close / BTCUSDT_close)
alt_consensus = (usdc_vs_usdt + fdusd_vs_usdt) / 2
alt_disagreement = abs(usdc_vs_usdt - fdusd_vs_usdt)
```

These are research proxies for relative quote-denominator value, not official
stablecoin FX rates. The three hourly closes are aligned by exact source hour
and close timestamp before the raw prices are discarded.

## Integrity result

Artifacts:

- panel:
  `data/binance_stablecoin_denominator_btc_2023/BTC_stablecoin_denominator_1h_2023-08-04T08_2023-12-31T23.csv.gz`
  - SHA-256:
    `aab063f0f9d898d5cdafffb57f552244083cd93fe69a3c6ebaf97faf6e27b642`
- manifest:
  `data/binance_stablecoin_denominator_btc_2023/build_manifest.json`
  - SHA-256:
    `863e96b4325d051731c92852c6760986204a9df62f77ff0dd0e01ab08d8a15d3`
- reference stablecoin-flow manifest SHA-256:
  `9e6a82b9747df5c0ba1c9278e436551de03ef6136c0ad3aeb05f0a451ed12134`

Two complete network builds were byte-identical. The measured full build used
about 126 MB maximum resident memory and 1.4 seconds wall time on this host.

### Source-only distributions

| Feature | Min | 1% | Median | 99% | Max |
|---|---:|---:|---:|---:|---:|
| `usdc_vs_usdt` | -0.002115 | -0.001432 | +0.000044 | +0.000887 | +0.001551 |
| `fdusd_vs_usdt` | -0.006564 | -0.002534 | +0.000345 | +0.001639 | +0.002367 |
| `alt_consensus` | -0.003513 | -0.001631 | +0.000199 | +0.000874 | +0.001429 |
| `alt_disagreement` | +0.000001 | +0.000010 | +0.000437 | +0.002898 | +0.006103 |

The wider FDUSD tail and nonzero disagreement make a two-book coherence gate
necessary. They do not authorize an outcome-tuned cutoff.

## Causal boundary

- A source hour `[h,h+1h)` is unavailable before `h+1h`.
- A future SDDR decision may be emitted at `h+1h`, but the conservative
  executable entry is the following five-minute open at `h+1h+5m`.
- Every rolling center, scale, and threshold must exclude the current source
  hour.
- Missing, duplicated, late, non-final, or timestamp-misaligned live rows fail
  closed.
- The source panel alone cannot establish that a ratio move is stablecoin
  pressure rather than asynchronous BTC spot microstructure. Frozen single-book,
  flow, latency, and BTC direction controls are mandatory.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m \
  training.build_binance_stablecoin_denominator_source --workers 8

uv run python -m pytest -q -p no:cacheprovider \
  tests/test_build_binance_stablecoin_denominator_source.py \
  tests/test_build_binance_stablecoin_quote_flow.py
```

The next work unit must freeze the single SDDR support rule before calculating
real event incidence. It still may not open a BTCUSDT perpetual outcome.
