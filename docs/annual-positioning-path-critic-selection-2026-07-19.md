# Annual positioning path critic — pre-2024 selection

## Decision

**FREEZE FOR OOS**

The family was searched only on 2022/2023 outcomes. 2024, 2025, and 2026 remained sealed.

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| development_2022 | 65.58% | 65.64% | 10.76% | 6.10 | 143 | 11/132 |
| selection_2023 | 62.68% | 62.73% | 6.88% | 9.12 | 171 | 82/89 |
| selection_2023_h1 | 43.69% | 107.81% | 6.15% | 17.54 | 85 | 36/49 |
| selection_2023_h2 | 13.22% | 27.94% | 6.88% | 4.06 | 86 | 46/40 |

## Frozen policy

```json
{
  "model": "annual_h288_mean",
  "hold_bars": 288,
  "loss": "squared_error",
  "target_quantile": null,
  "score_quantile": 0.8,
  "side": "both",
  "execution_stride_bars": 12,
  "target": "net fixed-hold return minus 0.5 times same-path MAE at 0.5x"
}
```

## Integrity boundary

- Tested cells: `162`; this multiple search is explicitly disclosed.
- Model refits annually using only paths that exited before each January 1 cutoff.
- Score threshold uses the completed prior year's feature-score distribution only.
- Exact realized funding, 6 bp/notional-side base cost, 10 bp stress, next-open fill, full-calendar CAGR, and strict held-OHLC MDD are used.
- OOS may be opened only with the byte-identical source and manifest.
