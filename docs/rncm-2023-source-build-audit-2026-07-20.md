# RNCM 2023 source-build audit

## Outcome boundary

This audit covers only the official Binance USD-M `BTCUSDT` `bookDepth`
archive and the frozen cumulative depth-weighted average-quote transform.
Price, forward-return, trade-PnL, and post-2023 outcomes remained unopened.

The builder manifest records both `outcomes_opened: false` and
`price_or_return_inputs_opened: false` under `protocol`.

## Reproducible build

```bash
/usr/bin/time -v /home/pakchu/rllm/.venv/bin/python \
  -m training.build_binance_um_book_centroid_2023 --workers 16
```

- elapsed wall time: `3:46.06`
- maximum resident set size: `371,500 KiB`
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
  - bytes: `12,945,239`
  - SHA-256: `539dd89d0ef555611353378fc1d5f09c0a3ab76167eee16de55e9e23db1c38dd`
- manifest:
  `results/binance_um_book_centroid_btcusdt_2023_manifest.json`
  - bytes: `202,606`
  - SHA-256: `1d39a32471d233d4a593e3a9ffdae88e3beaf2fbdfc8c8898a3c082b2598c552`

## Verification

```text
11 passed in 0.65s
```

The source passes its preregistered rarity gates. This is not evidence of
alpha. The next admissible operation is a source-only event-incidence and
mechanical-control gate frozen before any return is opened.
