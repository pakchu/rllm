# NWE-7 support rejection

Decision: **rejected before any return label or market row was loaded**.

NWE-7 was preregistered as a single online ridge policy in commit `cd9e6c0`.
Its outcome-blind feature clock combines only causally available Coin Metrics
blockspace and network observations. The support artifact reports zero market
or return rows loaded and keeps 2024, 2025, and 2026 YTD sealed.

## Outcome-blind weekly clock

| Window | Eligible weeks | Frozen minimum | Result |
|---|---:|---:|---|
| 2021-2022 train | 96 | 90 | pass |
| 2021 train | 44 | 42 | pass |
| 2022 train | 52 | 50 | pass |
| 2023 selection | 51 | 50 | pass |
| 2023 H1 | 26 | 25 | pass |
| 2023 H2 | 25 | 25 | pass |

All prediction features are finite. However, at the frozen first prediction
decision (`2021-03-01 12:00 UTC`) only **41** weekly samples have both their
feature availability timestamp and seven-day label exit in the past. The
preregistered online ridge requires **52** such samples, so the initial-history
gate fails.

## Integrity anchors

- Preregistration manifest hash:
  `19774e32cef371011e4fd753d92cb97c4464abcd35f196c66f67ef23192667c7`
- Support result hash:
  `d639ce24f8446b1b36748736faa61750885976d4925e7d2dc338c9a96c42bbec`
- Support JSON SHA-256:
  `5d6d0ed961a94fa9583fa282892b9c247834a0f92470cef5722fd76575aa4ef7`
- Clock CSV SHA-256:
  `5e0fc6b99a3fefe5b13a3f6ad66cd40cde6fb4423e4d845a0ee5854496e1af67`
- Clock frame hash:
  `4c559e7f537b78d1fb6a8e9233107b1da675be8915f659c9043a5f51ad12d366`

## Decision boundary

No weekly return label, trade direction, absolute return, CAGR, strict MDD,
control PnL, or 2024+ outcome was constructed. Moving the prediction start,
reducing the 52-sample minimum, or shortening the hold after this failure would
violate the frozen singleton contract.

The eight blockspace/network variables remain admissible weak features for a
new, separately preregistered policy with adequate causal warm-up. This exact
NWE-7 policy is not eligible for alpha or portfolio promotion.
