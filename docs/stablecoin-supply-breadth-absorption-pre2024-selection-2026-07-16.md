# Stablecoin supply breadth absorption — pre-2024 selection

## Frozen mechanism

The feature counts how many members of a fixed chain-specific stablecoin basket increased supply over seven days. A prior-only 180-day z-score converts that breadth into a sparse event clock. A broad expansion after a completed BTC decline is interpreted as cash absorption and goes long; a broad contraction after a completed BTC rally is interpreted as fragile liquidity and goes short.

- feature: `breadth_7d_z`
- fitted tails: `0.3`; lower `-1.072519767925`, upper `0.142734947593`
- direction rule: `absorb`
- hold: `7` days, no overlap
- fixed basket: `usdt_eth, usdt_trx, usdt_omni, usdc_eth, dai, busd, gusd, pax`; composite `usdt`/`usdc` excluded
- availability: all component rows must complete 1–3 days after observation; first 5-minute bar at/after the latest completion is the signal bar; next open fills.
- execution: 0.5x, 6 bp of notional per side, realized funding, full-calendar CAGR, intratrade strict MDD.

## Selection evidence

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit_2021h2 | 12.9402% | 23.0836% | 10.0056% | 2.3071 | 6 | 3/3 |
| fit_2022 | 15.9052% | 15.9169% | 14.3998% | 1.1054 | 15 | 8/7 |
| select_2023_h1 | 17.8259% | 39.2389% | 14.7443% | 2.6613 | 10 | 7/3 |
| select_2023_h2 | 11.7457% | 24.6639% | 12.1151% | 2.0358 | 8 | 5/3 |
| select_2023 | 31.6653% | 31.6901% | 14.7443% | 2.1493 | 18 | 12/6 |

## Falsification controls

- Direct, inverse, confirmation, unconditional price-momentum/reversal, and one-sided mappings are reported without altering the frozen choice.
- An every-valid-day price-reversal control and ±7/14/28-day event-clock shifts are reported.
- `2000` year- and event-sign-matched random clocks: all-block-positive fraction `0.0570`; q99 minimum-block ratio `0.6958`; empirical p(minimum-block ratio) `0.004998`; empirical p(positive blocks and summed return) `0.011494`.

## Integrity boundary

- The exploratory family opened `672` pre-2024 cells. Random-clock p-values are diagnostics, not family-wise correction.
- The trade-return p-value and random-clock p-values are descriptive post-selection diagnostics; none is selection-adjusted across the 672-cell family.
- Coin Metrics `SplyCur` can be revised. Composite assets and rows completed more than three days late are excluded from event generation, but `AssetEODCompletionTime` is not a value-vintage archive. A committed prefix hash detects later changes and prevents reranking; it does not prove the latest snapshot equals the value published historically.
- 2024+ BTC outcomes are globally research-seen in this repository. The freeze prevents reranking this new family after replay, but the future is not a pristine human holdout.
- This candidate is therefore a non-promotable historical hypothesis. Promotion requires immutable block-height reconstruction or forward-only versioned snapshots, positive test/eval/holdout performance, doubled-cost survival, and actual trade/PnL orthogonality.

Official sources:
- https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/supply/current-supply
- https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/availability/asseteodcompletiontime
- https://gitbook-docs.coinmetrics.io/access-our-data/api
