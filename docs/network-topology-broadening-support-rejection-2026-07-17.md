# NTB-7 support rejection

Decision: **rejected before any post-entry return was loaded**.

NTB-7 was preregistered as a singleton in commit `3077366`. The support
builder reads only Coin Metrics observation/availability timestamps and
`AdrActCnt`, `TxCnt`, and `TxTfrCnt`. It reports zero market or funding rows
loaded and keeps 2024, 2025, and 2026 YTD sealed.

## Outcome-blind clock

| Window | Non-overlapping events | Frozen minimum | Result |
|---|---:|---:|---|
| 2021-2022 train | 30 | 40 | fail |
| 2021 train | 11 | 15 | fail |
| 2022 train | 19 | 15 | pass |
| 2023 selection | 11 | 16 | fail |
| 2023 H1 | 7 | 6 | pass |
| 2023 H2 | 4 | 6 | fail |

The largest month contains 7.32% of events, below the frozen 20% maximum, but
four minimum-count checks fail. The final clock contains 41 events from
2021-03-09 through 2023-12-31.

## Integrity anchors

- Preregistration manifest hash:
  `7cc1eff657eda6b10ff86db6b8e1d0aebfe2e17ceb555473a9fb4d8fd748c9ef`
- Support result hash:
  `fd346b152f198cd5cf12782086774f490f9e74a59f37456a7a50e23d74103194`
- Support JSON SHA-256:
  `31fa9e4bec32986bcc5830ecf4dbbbb0d29c065bfd86fddb7b091f9c9eec87db`
- Clock CSV SHA-256:
  `6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f`
- Clock frame hash:
  `0ffcbe1eae61fbf51cc6076365f5f7af9323ea36fb51e0a54406912545c1b57f`

## Why there is no performance table

The preregistration explicitly requires rejection when support fails. Opening
returns and then loosening `breadth_z`, `fanout_z`, composite, or hold settings
would turn an outcome-blind singleton into an undeclared search. Therefore no
absolute return, CAGR, strict MDD, or direction-flip result was calculated.

The topology ratios remain potentially useful as weak portfolio features, but
this exact sparse seven-day event policy is not eligible for alpha promotion.
