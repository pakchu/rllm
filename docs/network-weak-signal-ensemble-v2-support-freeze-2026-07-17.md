# NWE-8 outcome-blind support freeze

Decision: **support passed; freeze the exact feature clock before labels**.

The support builder loaded only causally available Coin Metrics blockspace and
network observations. It loaded zero market/return rows, calculated no trade
direction or PnL, and kept 2024, 2025, and 2026 YTD sealed.

## Frozen support

| Window | Eligible weeks | Minimum | Result |
|---|---:|---:|---|
| 2021-2022 train | 82 | 80 | pass |
| 2021 train | 30 | 28 | pass |
| 2022 train | 52 | 50 | pass |
| 2023 selection | 51 | 50 | pass |
| 2023 H1 | 26 | 25 | pass |
| 2023 H2 | 25 | 25 | pass |
| Initial causal label history | 55 | 52 | pass |

All 133 prediction-eligible rows have finite feature values. The first decision
is `2021-06-07 12:00 UTC`; the last is `2023-12-18 12:00 UTC` so its seven-day
exit remains before 2024.

## Integrity anchors

- Preregistration manifest hash:
  `0ee4b21197641cb328a10b2fe16ec16d4e570ba502a9c61baeb74e961a354691`
- Support result hash:
  `095a5451b8e5f416223f64df13a5fd334531027119d50e461deea31ac1416152`
- Support JSON SHA-256:
  `f02377d7496751a2243384f73582cc189a4a7c2d5bc4184172424b959af39de7`
- Feature-clock CSV SHA-256:
  `3cc7eaa3b80944580651bf36541f0fde8edf4c66fd881d659f32396d1dda1c36`
- Feature-clock frame hash:
  `c966213175bf908782990e0fe4edb77a173f97d75e7c5dbbcec493557dbdc193`

The next admissible operation is to implement, test, commit, and hash the exact
online-model and strict-accounting evaluator. Return labels may be constructed
only after that evaluator freeze exists.
