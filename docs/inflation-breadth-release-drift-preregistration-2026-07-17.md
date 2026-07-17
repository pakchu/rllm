# IBRD-7 inflation breadth release drift preregistration

## Mechanism

At each archived BLS CPI release, IBRD-7 compares the newly published
not-seasonally-adjusted headline and core 12-month rates with the immediately
previous complete release. Concordant deceleration goes long BTC; concordant
reacceleration goes short. Mixed or unchanged breadth does not trade.

This source contains no BTC price, OI, taker flow, funding, premium, FX/Kimchi,
options, on-chain state, or existing-alpha state. It is therefore source-
orthogonal to the current portfolio. Orthogonal source does not imply positive
alpha; the frozen sequential outcome gates must establish that separately.

## Frozen execution

- availability: official 08:30 America/New_York BLS release timestamp;
- entry: release + 5 minutes;
- hold: 7 calendar days;
- exposure: 0.5x BTCUSDT perpetual;
- cost: 6 bp/notional/side, 10 bp stress;
- exact realized funding on `[entry, exit)`;
- strict intratrade MDD and full-calendar CAGR;
- no grid and no mutable parameter.

## Source-only density

| Window | Trades | Long | Short |
|---|---:|---:|---:|
| 2019_source_history | 4 | 2 | 2 |
| 2020 | 6 | 4 | 2 |
| 2021 | 7 | 1 | 6 |
| 2022 | 7 | 3 | 4 |
| stage1_2020_2022 | 20 | 8 | 12 |
| stage2_2023 | 7 | 7 | 0 |

Stage1 has exactly 20 events (8 long / 12 short). The untouched 2023 Stage2
has 7 source-only events, all long, reflecting a disinflation regime rather
than any inspected BTC outcome.

## Controls

- headline-only and core-only mechanism controls;
- exact direction flip;
- one complete CPI-release delay;
- deterministic hash-random side on the primary clock.

Every control receives the same complete cost, funding, strict-MDD,
subperiod, significance, and trade-count battery.

## Sequential boundary

Stage1 may physically parse only `[2020-01-01, 2023-01-01)`. Stage2 2023 can
open only after a hash-bound exact replay of a passing Stage1 result. Any
Stage1 failure rejects IBRD-7 unchanged and leaves 2023 sealed. 2024+ remains
sealed in both cases.

## Frozen identity

- source commit: `2e98c92`
- source panel SHA-256: `d199f409952d8cb83218864d0a96573bed82b59e649067b22fc97580a06d1059`
- clock SHA-256: `5938d851a17bf4dbd29e8cfecc3053f06bf1c764d11b595081e12ea7ae77a99e`
- preregistration manifest: `b493c5f290aebd31a4e19bd6b8f1e508d669edf28861191f999e8929299ff1e6`
