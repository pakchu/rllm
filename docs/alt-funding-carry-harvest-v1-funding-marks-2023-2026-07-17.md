# AFCH v1 causal funding marks — 2026-07-17

> Outcome-blind source freeze only. No position return, PnL, CAGR, MDD, or gate was calculated.

Missing 2023 funding-record marks use the close of the last fully completed
Binance USD-M 5m mark-price interval before the funding event. This is a
causal proxy, not an exact settlement mark. On rows where the exact recorded
mark exists, mark error times absolute funding rate must imply no more than
`0.1 bp/notional` funding-cash error;
exact/proxy counts and worst overlap errors are frozen below. Official endpoint:
<https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data>

| Symbol | Events | Exact recorded marks | Causal proxy marks | Missing non-event 5m bars | Max mark error bp | Max funding-cash error bp/notional |
|---|---:|---:|---:|---:|---:|---:|
| ADAUSDT | 1095 | 185 | 910 | 5 | 7.772021 | 0.001459694 |
| BNBUSDT | 1095 | 185 | 910 | 5 | 7.806802 | 0.001633790 |
| DOGEUSDT | 1095 | 185 | 910 | 5 | 6.485736 | 0.000812636 |
| ETHUSDT | 1095 | 185 | 910 | 5 | 2.892795 | 0.002057674 |
| SOLUSDT | 1095 | 185 | 910 | 5 | 17.218010 | 0.003018989 |
| XRPUSDT | 1095 | 185 | 910 | 5 | 2.717852 | 0.001137693 |

Manifest hash: `344eebba84e747313f062fcda9fa7fbf9960d234e75a2ea66bd507bbe5ad0667`
