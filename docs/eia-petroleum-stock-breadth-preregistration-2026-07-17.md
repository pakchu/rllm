# EPSB-1 EIA petroleum stock breadth preregistration

## Mechanism

EPSB-1 combines three weak, source-orthogonal physical inventory signals from
the same point-in-time WPSR issue. Concordant commercial-crude, gasoline, and
distillate builds represent a broad glut/disinflation impulse and go long BTC;
concordant draws represent a scarcity/reflation impulse and go short. Mixed,
zero, or quarantined releases abstain.

## Frozen execution

- source availability: next UTC day 13:00 after the official release date;
- entry: availability + 5 minutes;
- hold: 72 hours;
- exposure: 0.5x BTCUSDT perpetual;
- cost: 6 bp/notional/side, 10 bp stress;
- exact realized funding and strict intratrade MDD;
- no grid and no mutable parameter.

## Source-only density

| Window | Trades | Long | Short |
|---|---:|---:|---:|
| 2019_source_history | 19 | 8 | 11 |
| 2020 | 12 | 6 | 6 |
| 2021 | 11 | 3 | 8 |
| 2022 | 14 | 4 | 10 |
| stage1_2020_2022 | 37 | 13 | 24 |
| stage2_2023 | 13 | 6 | 7 |

## Controls

- commercial-crude-only and refined-products-only mechanism controls;
- exact direction flip;
- one complete WPSR release delay;
- deterministic hash-random side on the primary clock.

Every control receives the same cost, funding, strict-MDD, subperiod,
significance, and trade-count battery.

## Sequential boundary

Stage1 may physically parse only 2020–2022. Stage2 2023 opens only after a
hash-bound exact replay of a passing Stage1 result. Any Stage1 failure rejects
EPSB-1 unchanged and leaves 2023 sealed. 2024+ remains sealed.

## Frozen identity

- source commit: `c45dda1`
- source panel SHA-256: `26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b`
- clock SHA-256: `4f9735a3de7506e85f657217c8358608000d823927b6875c6cc236b94c39cedd`
- preregistration manifest: `050491adcc58735fd13a0ee949f0d79886203e2d3a5569468ce676b9f34a9e30`
