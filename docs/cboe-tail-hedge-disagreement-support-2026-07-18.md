# CTHD-1 source-support freeze — 2026-07-18

Status: **passed source-only support; BTC outcomes remain sealed**.

## Frozen inputs

- Cboe SKEW/VVIX/VIX panel: 1,507 exact-intersection observations,
  `2018-01-02` through `2023-12-29`.
- Panel SHA-256:
  `cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a`.
- Preregistration SHA-256:
  `d9e0e767e293d17c4845d300dad22c113b863796ef309d4d06ec8ecbe7330d0b`.
- No BTC market or funding rows were loaded.

## Primary support

The fixed 22.5% upper-tail short clock passes every preregistered source-only
floor.

| Window | Events | Short | Max month share |
|---|---:|---:|---:|
| 2021 | 123 | 123 | 17.07% |
| 2022 | 34 | 34 | 61.76% |
| Stage 1 | 157 | 157 | 13.38% |
| 2023 H1 | 124 | 124 | 18.55% |
| 2023 H2 | 23 | 23 | 69.57% |
| sealed 2023 | 147 | 147 | 15.65% |

The annual/half-window concentration is intentionally visible. The frozen
support rule limits concentration only over the complete Stage-1 and complete
2023 windows, while performance evaluation separately requires positive 2021,
2022, 2023-H1, and 2023-H2 results and uses weekly cluster inference.

## Frozen source controls

| Clock | Full source events |
|---|---:|
| primary hidden pressure | 426 |
| SKEW only | 415 |
| VVIX/VIX only | 409 |
| low VIX only | 428 |
| tail pair without VIX subtraction | 366 |
| one-release delay | 426 |
| seven-release placebo | 426 |

All clocks are strict-prior, short-only, next-source-session schedules and are
globally non-overlapping. Direction flip is constructed only inside the frozen
evaluator from the exact primary event clock.

## Integrity anchors

- support manifest hash:
  `57b7a756851c6cf86e7948d4a8a2f66f2a528b263587b362d5625bf859651339`
- support JSON SHA-256:
  `2a2f8bca6adc04812949a0af9f005b83e72f97395297c3547d1bd9748466c937`
- complete source-control ledger SHA-256:
  `0e19455e2fb5ab2d36cc996c9adf514adc85c69dd1a325562344a8015464d546`

This freeze authorizes only the exact preregistered Stage-1 evaluator to parse
2021–2022 BTC OHLC/funding. Calendar 2023 remains physically unopened unless
Stage 1 passes without any parameter change.
