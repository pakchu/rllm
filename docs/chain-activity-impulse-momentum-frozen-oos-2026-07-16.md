# Chain activity impulse momentum — frozen OOS result

The pre-2024 manifest `87c5361ab8669d1140aa3fed90322b33413bc84566bb0e34d2ecd2e56462dcc6`
was committed before this replay opened the Coin Metrics 2024+ network rows.
The pre-2024 network prefix, feature frame, schedules, execution economics and
selection statistics all reproduced before future performance was accepted.

## Frozen policy

- independent event clock: upper-10% 180-day z-score of the mean one-day log
  change in BTC active addresses, transactions and transfers;
- side: direction of the completed prior 24-hour BTC move;
- entry: one completed 5-minute signal bar after Coin Metrics
  `AssetEODCompletionTime`, then the next 5-minute open;
- hold: seven days, no overlapping position;
- accounting: 0.5x, 6 bp of notional per side, realized funding, full-calendar
  CAGR and intratrade strict MDD.

## Frozen performance

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | -13.6930% | -13.6670% | 22.1017% | -0.6184 | 13 | 1/12 |
| eval 2025 | 12.3305% | 12.3395% | 9.2386% | 1.3356 | 16 | 11/5 |
| holdout 2026H1 | -13.8344% | -30.0787% | 19.5502% | -1.5385 | 9 | 4/5 |
| OOS 2024–2026H1 | -17.2684% | -7.5419% | 24.4536% | -0.3084 | 38 | 15/23 |
| all 2021–2026H1 | 69.9154% | 10.6168% | 27.0114% | 0.3930 | 104 | 53/51 |

At doubled transaction cost the combined OOS result worsens to `-19.1342%`
absolute return, `-8.4102%` CAGR and `26.0242%` strict MDD.

## Statistical check

- OOS mean trade return: `-0.4551%`;
- approximate t statistic: `-0.9496`;
- approximate two-sided p value: `0.3423`;
- 95% mean-trade interval: `[-1.3943%, 0.4842%]`.

## Verdict

**Rejected.** The new event clock beat price-only and shifted-clock controls in
the pre-2024 selection, but its relationship to seven-day price continuation
did not survive 2024 and 2026. Do not invert or retune the same family on those
consumed windows. Retain the causal source adapter and availability tests for a
new mechanism, not this mapping.

The reported future is mechanically non-reranked but not globally pristine:
the repository had already inspected 2024–2026 BTC outcomes in other research.

Official source: <https://gitbook-docs.coinmetrics.io/access-our-data/api>
