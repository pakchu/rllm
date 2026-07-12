# Monthly continual positioning HGB alpha (2026-07-13)

## Verdict

Monthly causal refitting repaired the frozen model's negative 2026 return, but
did not produce a policy with `CAGR / strict MDD >= 3` in both 2024 and 2025.
No alpha or live candidate was promoted.

## Protocol

- Features: the same 138 delayed positioning, price, OI, flow, and macro inputs
  used by the fixed HGB path critic.
- Models: 24h and 48h long/short path-utility HGB critics.
- At every month boundary:
  - require each training label's `entry + hold` exit to be strictly before the
    cutoff;
  - refit on either expanding history or the trailing 730 days;
  - calibrate the score threshold only on prior 90/180/365-day feature rows;
  - trade the following month.
- 2023 full/H1/H2 selects and freezes the Top-10.
- Pre-future manifest SHA-256:
  `b19c5f03383a9f41bffc0c1ebae736dd157d1910fd89694d8dae890055aaf4fe`
- Future months are then replayed sequentially through 2024, 2025, and 2026.
- Costs: 0.5x, 5 bp fee + 1 bp slippage per side.
- MDD: conservative worst-order favorable-to-adverse OHLC high-water path
  drawdown.

## Best selected evidence

| Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| 48h expanding, cal90 q80, both, stride24 | 2023 Select | +38.59% | 38.63% | 9.54% | 4.05 | 73 |
| same | 2024 Test | +32.03% | 31.96% | 9.63% | 3.32 | 61 |
| same | 2025 Eval | -7.51% | -7.52% | 18.02% | -0.42 | 63 |
| same | 2026 YTD | +0.44% | 1.05% | 7.81% | 0.13 | 20 |
| 24h expanding, cal90 q80, long, stride12 | 2023 Select | +40.24% | 40.27% | 6.86% | 5.87 | 77 |
| same | 2024 Test | +25.68% | 25.62% | 4.63% | 5.54 | 48 |
| same | 2025 Eval | +3.72% | 3.73% | 10.79% | 0.35 | 68 |
| same | 2026 YTD | +4.79% | 11.91% | 9.56% | 1.25 | 34 |

## Interpretation

Continuous learning is directionally correct: the 24h long critic remained
profitable in every future window and changed 2026 from negative to positive.
Its 2025 edge is too small relative to the path drawdown, so leverage cannot fix
the ratio.  More threshold or update-cadence tuning on this same input set would
be selection overfit.

The next justified experiment adds a genuinely independent volatility state:
Deribit BTC volatility-index history.  That feature can distinguish ordinary
positioning disagreement from option-implied stress and may reduce the 2025
drawdown without retuning the same score gate.

## Artifacts

- Search: `training/search_positioning_continual_hgb_alpha.py`
- Tests: `tests/test_search_positioning_continual_hgb_alpha.py`
- Manifest: `results/positioning_continual_hgb_top10_manifest_2026-07-13.json`
- Result: `results/positioning_continual_hgb_alpha_scan_2026-07-13.json`
