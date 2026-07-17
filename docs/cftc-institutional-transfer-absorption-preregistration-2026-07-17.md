# CITA-1 CFTC institutional transfer absorption preregistration

## Mechanism

CFTC classifies asset managers/institutions and leveraged funds as distinct
buy-side groups. CITA-1 uses only their published weekly net-position changes.
When institutional net exposure rises while leveraged-fund net exposure falls,
the policy treats the transfer as persistent absorption and goes long BTC. The
reverse transfer goes short. Same-sign or zero changes abstain.

This is source-orthogonal to price action, Binance positioning, funding,
premium, FX, network, and existing portfolio scores.

## Frozen execution

- conservative CFTC availability, including the 2023 ION backlog;
- entry: availability + 5 minutes;
- hold: 7 days; source-time overlapping catch-up reports are skipped;
- exposure: 0.5x BTCUSDT perpetual;
- cost: 6 bp/notional/side, 10 bp stress;
- exact realized funding and strict intratrade MDD;
- no threshold, z-score, fitted direction, or mutable parameter.

## Source-only density

| Window | Trades | Long | Short | ION overrides |
|---|---:|---:|---:|---:|
| 2019_source_history | 31 | 17 | 14 | 0 |
| 2020 | 34 | 17 | 17 | 0 |
| 2021 | 36 | 20 | 16 | 0 |
| 2022 | 28 | 14 | 14 | 0 |
| stage1_2020_2022 | 98 | 51 | 47 | 0 |
| 2023_h1 | 12 | 7 | 5 | 2 |
| 2023_h2 | 11 | 9 | 2 | 0 |
| stage2_2023 | 23 | 16 | 7 | 2 |

## Controls

- asset-manager-only and leveraged-money-contrarian-only mechanisms;
- exact direction flip;
- one complete report delay;
- deterministic hash-random side on the primary clock.

Every control receives the same cost, funding, strict-MDD, subperiod,
significance, and trade-count battery.

## Sequential boundary

Stage1 may physically parse only 2020–2022. Stage2 2023 opens only after a
hash-bound exact replay of a passing Stage1 result. Any Stage1 failure rejects
CITA-1 unchanged and leaves 2023 sealed. 2024+ remains sealed.

## Frozen identity

- source commit: 27823e7
- source panel SHA-256: 064eed3fa340b1701f4686d1176de2a10f39128abc5ebf846e8b6319b8144ee6
- clock SHA-256: 65f64544ff8d45ab20882b770907cc958d96c5bd8982f731e09941de2e1a28d2
- preregistration manifest: 4c5e9911524125f960d3da06bd6c6caa7e83b124d77ab643e2614751d7573038
