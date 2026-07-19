# RNCM 2023 source-build audit

## Outcome boundary

This audit covers only the official Binance USD-M `BTCUSDT` `bookDepth`
archive and the frozen cumulative depth-weighted average-quote transform.
Price, forward-return, trade-PnL, and post-2023 outcomes remained unopened.

The schema-v2 builder manifest records `outcomes_opened: false`,
`external_market_ohlc_or_return_inputs_opened: false`, and
`average_quote_price_levels_derived: true` under `protocol`. The last field is
explicit because `notional/depth` is itself an average quote, even though no
external OHLC or future outcome is read.

## Reproducible build

```bash
/usr/bin/time -v /home/pakchu/rllm/.venv/bin/python \
  -m training.build_binance_um_book_centroid_2023 --workers 16
```

- elapsed wall time: `4:08.64`
- maximum resident set size: `374,940 KiB`
- output rows: `105,120`, covering every 5-minute slot from
  `2023-01-01 00:00:00` through `2023-12-31 23:55:00`
- source-complete rows: `101,956` (`96.9901%`)
- officially missing archive dates: `2023-02-08`, `2023-02-09`

## Frozen source-quality gate result

The limits were committed in
`docs/rncm-source-quality-gate-2026-07-20.md` before anomaly counts were read.

| Check | Frozen rejection limit | Observed | Result |
|---|---:|---:|---|
| invalid snapshots / verified snapshots | `> 0.0001` | `8 / 1,021,111 = 0.0000078346` | pass |
| quarantined timing-complete bars / timing-complete bars | `> 0.001` | `8 / 101,964 = 0.0000784591` | pass |
| maximum daily invalid-snapshot fraction | `> 0.01` | `0.0003508772` | pass |

All eight affected five-minute bars were fully quarantined. No invalid
snapshot was clipped, repaired, winsorized, imputed, or partially salvaged.

## Artifact identity

The generated data and detailed manifest are intentionally ignored by Git.
Their identities for this research sequence are:

- panel:
  `data/binance_um_book_centroid_btcusdt_2023/BTCUSDT_um_book_centroid_skew_5m_2023.csv.gz`
  - bytes: `13,622,241`
  - SHA-256: `c4053ce27d28bebda4137349192b1a940360231469f63edc32bacabb2ce54131`
- manifest:
  `results/binance_um_book_centroid_btcusdt_2023_manifest.json`
  - bytes: `202,846`
  - SHA-256: `d8237c4562d33c12eff162776f723cc5fc94649b69d26a6230e16fc38c52bba1`

## Verification

```text
11 passed in 0.65s
```

The source passes its preregistered rarity gates. This is not evidence of
alpha. The next admissible operation is a source-only event-incidence and
mechanical-control gate frozen before any return is opened.
