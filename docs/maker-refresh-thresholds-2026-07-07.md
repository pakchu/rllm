# Maker refresh calibration — 2026-07-07

## Failure / orphan root cause

- The remaining `BTCUSDT LONG 0.001` position was opened by the earlier gross-6 process at `2026-07-07 02:38:23 UTC`, order `1063856762137`, client id `rpf_1783391704_dfbbce_415cb3fe`.
- Its signal was recovered from the client-order digest as `oi_divergence_pullback_range_rsi_h96_s6:2026-07-07T02:30:00`.
- The configured hold was 96 five-minute bars, so the intended exit was `2026-07-07 10:35:00 UTC`.
- The local portfolio state no longer contained that sleeve after the process/restrategy restart, so the loop had no `open_sleeves` entry to evaluate for exit. The exchange position remained live until exchange-position recovery imported it and immediately closed it at `2026-07-07 16:26:01 UTC`.

Fix: `execution/portfolio_live.py` now reconciles live Binance positions into local state using bot-owned `rpf_*` client order ids, even if the position came from a previous live portfolio config.

## 2026-07-07 20:35 KST failed execution

- KST `2026-07-07 20:35` = UTC `2026-07-07 11:35`.
- Signal: `oi_divergence_sma24_highfreq_h30_s6_with_selector:2026-07-07T11:30:00`.
- Order: `1064198865364`, client id `rpf_1783424101_313bb5_eafef251`.
- It was placed once as post-only and left at the original maker price until the 300s timeout. It never refreshed as the book moved, then was cancelled with `executedQty=0.000` at UTC `11:40:02`.

Fix: live post-only orders are now refreshed every 60 seconds when the refreshed maker price remains inside the calibrated deviation band from the signal/exit reference.

## Calibration method

Script: `scripts/calibrate_maker_refresh_thresholds.py`

Input:

- Binance `BTCUSDT` 1m bars from `2024-01-01` through `2026-07-07 16:00 UTC`.
- 264,575 five-minute decision samples.
- Historical L2 is not available locally, so 1m OHLC is used as a conservative proxy.
- For each refresh minute, a post-only price is proxied from the minute open with the live maker offset `0.01%`.
- A fill is counted when that minute's high/low crosses the maker price.
- Selection rule: smallest threshold within 0.5 percentage point of the max fill rate.

Selected thresholds:

| Path | Refresh horizon | Selected max deviation | Mean fill-rate at selected threshold | Max tested fill-rate |
| --- | ---: | ---: | ---: | ---: |
| Open | 5 minutes | `0.003` = `0.30%` | `97.1494%` | `97.3314%` |
| Close | 10 minutes | `0.002` = `0.20%` | `98.9931%` | `99.4877%` |

Live defaults:

- `--maker-refresh-interval-sec 60`
- `--entry-maker-max-deviation-pct 0.003`
- `--exit-maker-max-deviation-pct 0.002`
- `--max-entry-wait-sec 300`
- `--max-exit-wait-sec 600`
