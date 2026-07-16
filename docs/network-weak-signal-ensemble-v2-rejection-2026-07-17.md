# NWE-8 strict pre-2024 rejection

Decision: **rejected before orthogonality and portfolio testing**.

The exact preregistered online ridge, controls, costs, realized funding, and
strict-MDD evaluator were frozen before labels. Only the train window
(`2021-06-07` through 2022) and calendar-2023 selection window were opened.
2024, 2025, and 2026 YTD remain sealed.

## Primary outcome

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| Train | -46.34% | -32.76% | 55.46% | -0.59 | 58 | 23 / 35 |
| 2023 | +10.29% | +10.30% | 10.28% | 1.00 | 22 | 16 / 6 |
| 2023 H1 | +9.51% | +20.13% | 8.55% | 2.35 | 15 | 14 / 1 |
| 2023 H2 | +0.71% | +1.41% | 10.28% | 0.14 | 7 | 2 / 5 |

At 10 bp/notional/side, train absolute return deteriorates to `-47.59%` and
2023 remains positive at `+9.32%`, but the primary policy fails both the train
and 2023 CAGR/strict-MDD requirement. Train mean signed underlying move is
`-180.77 bp`, showing that costs are not the root cause.

The one-bar-delayed control reproduces the failure (`-46.86%` train absolute
return), so the result is not a five-minute entry-timing artifact. The exact
direction flip earns `+55.41%` in train but loses `-13.09%` in 2023; therefore
inversion is not a stable repair and was diagnostic-only by preregistration.

## Interpretation

The network/blockspace relationship changes sign across regimes. Constant
weekly long also loses `-33.25%` in train and earns `+54.97%` in 2023, confirming
that the split contains a broad BTC regime reversal. NWE-8 does not isolate a
stable price-independent alpha from that regime change.

No feature, sign, training window, abstention threshold, or hold is modified
after this outcome. NWE-8 is not eligible for alpha registration or portfolio
promotion.

## Integrity anchors

- Evaluator source SHA-256:
  `ccc04175f853caf4642d6a9d5b8670dc097553d883e02038bd4d32e51191bf64`
- Evaluator-freeze JSON SHA-256:
  `02557349b71a3f2216816d70c08c61e6e3869c0ca269107691e73d7f4c7bb124`
- Selection result SHA-256:
  `64816351ace7af10fd78147018953d1cdda5b25c4dd4c451bfe448cb9b8aca1c`
- Selection result manifest hash:
  `50d152a8ddf3ae0fa97fea070b709da2b0999364aed645e8de44013faf383caa`
