# Aggressor execution-centroid inversion alpha search (2026-07-14)

## Decision

**Reject the static alpha; preserve only the observable execution-centroid
representation as weak beta; keep 2024+ sealed.**

Binance completed klines provide taker-buy volume in both base and quote units.
Their ratio reconstructs the average execution price of buyer-initiated trades.
Subtracting taker-buy quantities from total base and quote volume reconstructs
the seller-initiated average execution price. Binance documents these raw kline
fields as taker-buy base and quote asset volume in its
[official kline API documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data).

For each completed hour, define buyer centroid `B`, seller centroid `S`, and
hour-end close `C`. The one-cell primary rule was:

- long when `B < S < C`;
- short when `C < B < S`.

This is threshold-free. In both cases the ultimately winning aggressor side
transacted at the better average price and the hour settled beyond both
centroids. The economic interpretation was a continuation after the opposite
aggressor side became marked against the close. “Trapped” or “informed” traders
are hypotheses, not observed account labels.

## Causal protocol

- The returned analysis frame is hard-filtered before `2024-01-01`. The shared
  parser may read and immediately discard later rows in the cutoff-crossing
  chunk; no such row enters the returned frame, features, signals, or outcomes.
- Each decision uses exactly the 12 completed 5-minute rows ending at minute
  55 and enters the next minute-00 open.
- One fixed 12-hour hold, 0.5x exposure, 6 bp/side cost, split-contained
  non-overlapping trades, and favorable-first/adverse-second OHLC strict MDD.
- One primary mapping only. The exact direction flip is a falsification control,
  not a selectable second policy.
- Direct controls include ordinary centroid order, centroid-order removal,
  terminal-settlement removal, `B<S` plus VWAP/hourly-return direction, raw
  VWAP/return/taker direction, and 5m/1h/24h/7d delays.
- 2023 is inspected internal selection. No 2024+ outcome was computed.

## Accounting and support-only preflight

All checks below were completed before returns were opened.

- Completed-hour decisions: 35,073.
- Valid buyer/seller centroid pairs: 35,073; invalid: 0.
- Maximum quote-volume reconstruction relative error: `2.38e-16`.
- Centroids outside the completed-hour high/low range: 0.
- Total raw primary events: 4,835, balanced 2,547 long / 2,288 short.

| Split | Raw (L/S) | Strict executable (L/S) |
|---|---:|---:|
| Fit | 2,287 (1,206/1,081) | 914 (499/415) |
| 2020Q4 | 203 (120/83) | 87 (45/42) |
| 2021H1 | 378 (224/154) | 188 (121/67) |
| 2021H2 | 507 (256/251) | 207 (113/94) |
| 2022H1 | 439 (225/214) | 186 (98/88) |
| 2022H2 | 760 (381/379) | 244 (121/123) |
| 2023 | 1,498 (774/724) | 472 (237/235) |
| 2023H1 | 653 (340/313) | 226 (117/109) |
| 2023H2 | 845 (434/411) | 246 (120/126) |

Support was therefore not the failure.

## Primary results

| Split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| Fit | **-50.05%** | -26.93% | 54.86% | -0.49 | 914 (499/415) |
| 2020Q4 | -13.34% | -48.86% | 19.47% | -2.51 | 87 (45/42) |
| 2021H1 | +8.41% | +17.70% | 21.32% | 0.83 | 188 (121/67) |
| 2021H2 | -21.26% | -37.79% | 30.09% | -1.26 | 207 (113/94) |
| 2022H1 | -25.14% | -44.25% | 33.83% | -1.31 | 186 (98/88) |
| 2022H2 | -6.83% | -13.10% | 16.33% | -0.80 | 244 (121/123) |
| 2023 | **-10.75%** | -10.76% | 24.96% | -0.43 | 472 (237/235) |
| 2023H1 | -9.26% | -17.80% | 16.94% | -1.05 | 226 (117/109) |
| 2023H2 | -1.64% | -3.24% | 14.45% | -0.22 | 246 (120/126) |

Approximate mean-trade p-values were 0.132 in fit and 0.581 in 2023, with
negative effect sizes. Only 2021H1 was positive. The static rule is not close to
the required CAGR/strict-MDD ratio 3.

## Structural and timing controls

| Control | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact direction flip | -43.30% / -0.45 | -38.39% / -0.91 |
| Ordinary centroid terminal order | -53.57% / -0.51 | -38.16% / -0.82 |
| Terminal settlement, centroid order removed | -57.76% / -0.46 | -32.72% / -0.87 |
| `B<S` plus centroid midpoint direction | -41.76% / -0.45 | -7.94% / -0.33 |
| `B<S` plus market-VWAP direction | -41.76% / -0.45 | -12.36% / -0.51 |
| `B<S` plus hourly-return direction | -53.84% / -0.45 | -5.01% / -0.19 |
| Market-VWAP direction only | -29.38% / -0.27 | -20.96% / -0.67 |
| Hourly-return direction only | -64.44% / -0.50 | -3.61% / -0.15 |
| Taker-flow direction only | -65.94% / -0.54 | -21.20% / -0.70 |
| Signal delayed 5m | -44.22% / -0.42 | -13.05% / -0.49 |
| Signal delayed 1h | -61.31% / -0.50 | -6.38% / -0.24 |
| Signal delayed 24h | -50.31% / -0.46 | -30.03% / -0.92 |
| Signal delayed 7d | -55.94% / -0.51 | -9.57% / -0.35 |

The exact flip also loses, so there is no stable opposite direction to promote.
The primary event Jaccard with the three direct `B<S` momentum controls is
`0.976`. Requiring the close to settle beyond both centroids removes only about
2.4% of those events. Consequently, the proposed topology does not establish a
distinct executable transition beyond `B<S` plus simple price direction.

The continuous signed centroid feature remains less linearly redundant:
Spearman correlations were 0.251 with hourly return, 0.293 with close-minus-
VWAP, and 0.213 with taker imbalance. That supports retaining raw centroids as
an observable representation, not retaining this static signal.

## Cost stress

| Cost/side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | -13.56% / -0.15 | +18.47% / 1.42 |
| 1 bp | -21.11% / -0.23 | +13.01% / 0.88 |
| 3 bp | -34.29% / -0.36 | +2.83% / 0.15 |
| 6 bp | -50.05% / -0.49 | -10.75% / -0.43 |
| 10 bp | -65.35% / -0.56 | -26.11% / -0.80 |
| 15 bp | -78.07% / -0.63 | -41.65% / -0.98 |

Turnover makes the 2023 effect uneconomic, but costs are not the root failure:
fit loses even at zero cost and no cost level reaches ratio 3 in either period.

## Conclusion

The raw buyer/seller execution centroids are exact, causal, live-computable,
and absent from the existing alpha families. They may be useful tokens for a
materially different preregistered learner, including an RLLM that reasons over
auction ordering. However, all inspected static orderings, directions, matched
momentum controls, hold, and timing variants are now frozen as gamma failure
provenance. Do not rescue this sample with centroid-gap thresholds, OI gates,
alternate holds, or topology transitions.

## Artifacts

- `training/search_aggressor_centroid_inversion_alpha.py`
- `tests/test_search_aggressor_centroid_inversion_alpha.py`
- `results/aggressor_centroid_inversion_alpha_scan_2026-07-14.json`
