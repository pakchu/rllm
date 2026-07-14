# Binance aggTrade source-integrity audit — 2026-07-14

## Verdict

**PASS with fail-closed quarantine.** The feature artifact is suitable for
pre-2024 alpha research only when the five source-gap UTC days below, every
missing 5m slot, and the next 24 bars are excluded from sequential features and
trade entry/exit. No return, label, or 2024+ outcome was opened by this audit.

## Built artifacts

- aggTrade features: 420,732 observed 5m rows over
  `[2020-01-01, 2024-01-01)`;
- official daily-kline reference: all 420,768 expected 5m rows;
- 48 monthly aggTrade artifacts and 1,461 source archive hashes verified;
- disk footprint: about 157 MB for aggTrade artifacts and 34 MB for klines.

The 36-row difference is not silently imputed:

- 26 slots have zero volume and zero trades in the official daily kline;
- 10 slots have nonzero kline activity but fall inside confirmed aggTrade
  source omissions.

## Confirmed source-gap days

| UTC day | Missing aggregate IDs inside daily archive |
|---|---:|
| 2020-04-15 | 46 |
| 2021-02-09 | 22,793 |
| 2021-02-24 | 13 |
| 2021-05-19 | 557,026 |
| 2022-09-06 | 31,656 |

Cross-day aggregate IDs have zero gaps and zero overlaps. The omissions are
inside otherwise checksummed daily files. The two largest gaps were also
independently reproduced in official monthly aggTrade archives:

- 2021-02 monthly SHA-256
  `6f8fe48bc9b634ab364163b321a456b71286d7816957de7d9aa38a0721335cd1`;
- 2021-05 monthly SHA-256
  `11d546ec4752291f425d3bdeac0df1af5a7f48a08a700ba33a571bcbe590330b`.

The monthly files contain the same 27.11-minute and 24.61-minute holes, so
replacing daily archives with monthly ones would not repair the data.

## Kline source-version finding

Official monthly and daily kline archives are not always identical. At
`2023-11-10 15:05 UTC`, the monthly archive and old wave cache report
`high=37118.7`, `close=37118.4`, `volume=206.313`, while the current official
daily kline reports `high=37168.7`, `close=37145.6`, `volume=1252.754` and
matches the daily aggTrade archive. Reconciliation therefore uses checksummed
**daily kline vs daily aggTrade** sources. The older wave cache is not the
authoritative OHLCV reference for this experiment.

## Reconciliation after quarantining source-gap days

Per-5m differences remain expected because a Binance aggregate event can span
an underlying kline boundary but is assigned wholly by its transaction time.
Those local differences cancel at longer horizons:

| Field | 5m p99 rel. error | Daily max rel. error | Monthly max rel. error | Full-period rel. error |
|---|---:|---:|---:|---:|
| base volume | 0.1056% | 0.0528% | 0.00237% | 0.000076% |
| quote notional | 0.1056% | 0.0532% | 0.00236% | 0.000075% |
| taker-buy quote | 0.0984% | 0.0523% | 0.00222% | 0.000086% |
| underlying trade count | 0.1242% | 0.5005% | 0.0551% | 0.0131% |

Additional hard checks passed:

- direct UTC join dominates both `-9h` and `+9h` shift controls;
- every available first/last aggTrade price lies inside the daily kline
  high/low envelope;
- last aggTrade price equals kline close in 99.999% of clean rows;
- all accounting identities, finite-value checks, hashes, archive coverage and
  daily aggregate-event row counts pass.

## Mandatory downstream rule

Reindex to the exact official 5m grid and retain an availability mask. A signal
is invalid when its lookback, entry, hold, or exit intersects:

1. any missing aggTrade slot;
2. any source-gap UTC day;
3. the following 24 bars (two hours), which is the maximum preregistered
   metaorder lookback.

The machine-readable evidence is
`results/binance_aggtrade_microstructure_audit_2026-07-14.json`.
