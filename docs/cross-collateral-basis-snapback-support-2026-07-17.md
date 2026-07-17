# CCBS-12 support and entry-clock orthogonality — 2026-07-17

## Outcome boundary

This unit loaded only completed close-based CCBS signal columns. It did **not**
open CCBS entry/exit prices, held OHLC paths, returns, PnL, CAGR, or MDD. Known
live-anchor paths were used only to reconstruct the already-live entry clock;
they could veto but could not select or change the CCBS threshold.

## Support-only threshold

The largest 2021-2022-supported threshold is **z=2.0**. It produced
143 pre-2023 events ({'2021': 71, '2022': 72}) and passed every frozen
year/half/quarter/sign floor. Its 2023 count is 58 and is reported
only as a feature-support diagnostic; 2023 was already declared development,
not pristine OOS.

## 2023 live-anchor clock overlap

- CCBS entries: 58; live-anchor unique entries: 135;
- exact 5m intersections: 0 (0.000% of CCBS);
- entry-day Jaccard: 0.1324;
- frozen limits: exact overlap <= 10%, day Jaccard <= 0.20.

Disposition: **PASS_SUPPORT_OPEN_2023_PNL**. PnL may open only when this support
unit passes. Daily-PnL/BTC/portfolio orthogonality remains a later outcome gate,
and live promotion remains blocked by the omitted COIN-M collateral ledger.

Report content hash: `29a002968e784604512e83407facb0d53a0a3c6536d1038af4f9d44adf51d4f1`
