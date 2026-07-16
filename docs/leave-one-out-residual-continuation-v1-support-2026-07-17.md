# LORC v1 calendar-2025 support freeze — 2026-07-17

> Outcome-blind event support only. No post-entry 2025 return, PnL, CAGR, or MDD was calculated.

## Support

- events: `99` (H1 `34`, H2 `65`)
- unique ordered pairs: `29`
- maximum pair share: `7.0707%`
- maximum monthly source quarantine: `0.1346%`
- support decision: **PASS**
- clock SHA-256: `b7bdb75831ad597cb23212740e571760de9210a87f709eaddd35cedba5612956`

| Long > short | Events |
|---|---:|
| ADAUSDT>BNBUSDT | 2 |
| ADAUSDT>DOGEUSDT | 2 |
| ADAUSDT>ETHUSDT | 6 |
| ADAUSDT>SOLUSDT | 6 |
| ADAUSDT>XRPUSDT | 1 |
| BNBUSDT>ADAUSDT | 2 |
| BNBUSDT>DOGEUSDT | 4 |
| BNBUSDT>ETHUSDT | 5 |
| BNBUSDT>SOLUSDT | 2 |
| BNBUSDT>XRPUSDT | 2 |
| DOGEUSDT>ADAUSDT | 1 |
| DOGEUSDT>BNBUSDT | 3 |
| DOGEUSDT>ETHUSDT | 7 |
| DOGEUSDT>SOLUSDT | 7 |
| DOGEUSDT>XRPUSDT | 5 |
| ETHUSDT>ADAUSDT | 1 |
| ETHUSDT>BNBUSDT | 2 |
| ETHUSDT>SOLUSDT | 4 |
| ETHUSDT>XRPUSDT | 6 |
| SOLUSDT>ADAUSDT | 6 |
| SOLUSDT>BNBUSDT | 3 |
| SOLUSDT>DOGEUSDT | 3 |
| SOLUSDT>ETHUSDT | 4 |
| SOLUSDT>XRPUSDT | 1 |
| XRPUSDT>ADAUSDT | 2 |
| XRPUSDT>BNBUSDT | 3 |
| XRPUSDT>DOGEUSDT | 1 |
| XRPUSDT>ETHUSDT | 6 |
| XRPUSDT>SOLUSDT | 2 |

Every clock row uses completed data through `signal_time`, enters at +5m, exits
after exactly 12h, is factor-beta neutral, and does not overlap another LORC
position. A 13-completed-hour source-integrity gate covers the 12h factor/flow
window and the residual's `t-12` close.
