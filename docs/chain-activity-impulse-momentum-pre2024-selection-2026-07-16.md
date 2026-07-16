# Chain activity impulse momentum — pre-2024 selection

## Frozen mechanism

A completed BTC network day is converted into the mean one-day log change of active addresses, transactions and transfers. When its 180-day rolling z-score enters the fitted upper tail, the policy trades in the direction of the already-completed prior 24-hour BTC move for seven days.

- event: `activity_shock_1d`
- fit tail: `0.1`; threshold `1.494385582429`
- direction: `momentum`
- hold: `7` days
- source availability: Coin Metrics `AssetEODCompletionTime`; signal uses a completed 5-minute bar after availability; fill is the next 5-minute open.
- execution: 0.5x, 6 bp of notional per side, realized funding, full-calendar CAGR, intratrade strict MDD.

## Selection evidence

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit_2021 | 18.6852% | 22.6880% | 27.0114% | 0.8399 | 21 | 9/12 |
| fit_2022 | 32.1347% | 32.1599% | 15.9271% | 2.0192 | 24 | 14/10 |
| select_2023_h1 | 22.3284% | 50.1849% | 10.1482% | 4.9452 | 9 | 8/1 |
| select_2023_h2 | 9.0091% | 18.6768% | 7.4796% | 2.4970 | 11 | 5/6 |
| select_2023 | 30.3548% | 30.3785% | 10.1482% | 2.9935 | 21 | 15/6 |

## Falsification controls

- price-only seven-day momentum is reported separately and does not reproduce the candidate's all-block result.
- ±7/14/28-day event-clock shifts are reported; none is used to alter the frozen rule.
- `2000` year-stratified matched random clocks: all-block-positive fraction `0.0535`; q99 minimum-block ratio `0.6664`; empirical p(minimum-block ratio) `0.007996`; empirical p(positive blocks and summed return) `0.004998`.

## Integrity boundary

- No exchange-address-labelled metric is used.
- The formal search opened 180 pre-2024 cells, so the family still has multiple-testing risk. Earlier exploratory prototypes also inspected other network transforms; random-clock p-values are diagnostics, not a full family-wise correction.
- 2024+ BTC outcomes are globally research-seen in this repository; they are not pristine human holdout. The manifest still prevents this new data family from being re-ranked after its 2024+ replay.
- Promotion requires positive test/eval/holdout performance, doubled-cost survival, and measured trade/PnL orthogonality.
- OOS replay intentionally uses a separate full-horizon network file produced by the same downloader; the pre-2024 prefix must reproduce the frozen network hash.

Official source: https://gitbook-docs.coinmetrics.io/access-our-data/api
