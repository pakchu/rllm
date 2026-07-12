# Continual positioning + Deribit DVOL HGB alpha (2026-07-13)

## Verdict

Adding the causally joined Deribit BTC volatility index improved the continual
critic's 2026 behavior, but did not repair the missing 2025 edge.  None of the
185 policies met the alpha or live-promotion criteria, so this model remains a
research sleeve rather than a standalone alpha.

## Protocol

- Inputs: the delayed Binance USD-M futures positioning feature set plus hourly
  Deribit BTC DVOL state and three fixed DVOL/positioning interactions.
- DVOL rows are joined only when their `close_time` is already observable at
  the market timestamp; a 65-minute tolerance prevents stale option state from
  being carried indefinitely.
- At each month boundary, labels are admitted only when their complete trade
  path exits strictly before the cutoff.  Models and score calibration use only
  those prior rows.
- The Top-10 policy list is selected from 2023 full/H1/H2 results and physically
  written before loading 2024-2026 data.
- Pre-future manifest hash:
  `b0337a4b8f6d6acfc5baafde206aab22990f58c87ad4eb810b2d2fadf8c3256f`
- Costs: 0.5x leverage, 5 bp fee plus 1 bp slippage per side.
- MDD: conservative worst-order favorable-to-adverse OHLC high-water path
  drawdown, including intraposition drawdown.
- CAGR uses each complete calendar evaluation window, including idle time.

## Best selected evidence

| Policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---:|---:|---:|---:|---:|
| 48h expanding, cal180 q90, long, stride24 | 2023 Select | +21.80% | 21.82% | 5.98% | 3.65 | 30 |
| same | 2024 Test | +24.22% | 24.17% | 8.34% | 2.90 | 33 |
| same | 2025 Eval | -1.30% | -1.30% | 12.72% | -0.10 | 40 |
| same | 2026 YTD | +7.53% | 19.06% | 5.65% | 3.37 | 16 |
| 48h expanding, cal90 q90, long, stride12 | 2023 Select | +22.48% | 22.49% | 9.82% | 2.29 | 38 |
| same | 2024 Test | +32.59% | 32.51% | 6.27% | 5.18 | 35 |
| same | 2025 Eval | +1.76% | 1.76% | 12.26% | 0.14 | 39 |
| same | 2026 YTD | +8.87% | 22.65% | 6.39% | 3.54 | 19 |

## Interpretation

DVOL supplies useful independent state: the stronger selected sleeve achieved
`CAGR / strict MDD` above 3 in 2024 and 2026.  Its 2025 return, however, stayed
near zero while strict MDD remained above 12%.  This is not a threshold problem;
the learned directional utility has no durable standalone edge in that regime.

The defensible next test is complementarity with a separately selected bearish
REX sleeve.  Any blend must keep this 2023 manifest frozen and must recompute a
single combined high-water equity path rather than combining standalone MDDs.

## Artifacts

- Search: `training/search_positioning_continual_hgb_alpha.py`
- Tests: `tests/test_search_positioning_continual_hgb_alpha.py`
- Manifest: `results/positioning_dvol_continual_hgb_top10_manifest_2026-07-13.json`
- Result: `results/positioning_dvol_continual_hgb_alpha_scan_2026-07-13.json`
