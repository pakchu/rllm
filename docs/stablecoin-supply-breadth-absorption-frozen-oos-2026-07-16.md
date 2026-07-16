# Stablecoin supply breadth absorption — frozen OOS result

The pre-2024 policy and manifest
`481275ac302630cfd3c1ff1c5672f3431806fc10c49d592aa0906bb4e4ea34d6`
were committed before the reviewed Coin Metrics 2024+ supply rows were opened.
The pre-2024 market, funding, stablecoin and feature prefixes, schedules,
execution economics and statistics reproduced exactly.

## Frozen policy

- event: upper/lower 30% tails of a prior-only 180-day z-score of the number of
  fixed-basket stablecoins whose supply increased over seven days;
- long: broad supply expansion after a completed negative BTC 24-hour move;
- short: broad supply contraction after a completed positive BTC 24-hour move;
- hold: seven days without overlap;
- execution: one completed 5-minute signal bar after the latest component
  completion, next-open fill, 0.5x, 6 bp per side, realized funding,
  full-calendar CAGR and intratrade strict MDD.

## Frozen performance

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | 0.8525% | 0.8507% | 8.4606% | 0.1006 | 1 | 1/0 |
| eval 2025 | -6.4526% | -6.4568% | 8.7664% | -0.7365 | 7 | 5/2 |
| holdout 2026H1 | 5.3405% | 13.3172% | 9.1394% | 1.4571 | 6 | 3/3 |
| OOS 2024–2026H1 | -0.6166% | -0.2555% | 11.3042% | -0.0226 | 14 | 9/5 |
| all 2021H2–2026H1 | 71.9099% | 11.4399% | 14.7443% | 0.7759 | 54 | 32/22 |

At doubled transaction cost the combined OOS result is `-1.4483%` absolute
return, `-0.6016%` CAGR, `11.7291%` strict MDD and `-0.0513` CAGR/MDD.
The 14 OOS trades have mean net return `-0.0141%`, approximate p value
`0.9837`, and a 95% mean interval of `[-1.3658%, 1.3377%]`.

## Verdict

**Rejected.** The supply-breadth/price-divergence relation did not survive the
frozen future, and 2024 supplied only one trade. Do not invert or retune the
same family on the consumed windows.

Independently, this history was never promotion-eligible: reviewed Coin Metrics
`SplyCur` is a latest snapshot, while `AssetEODCompletionTime` is not a value
vintage archive. Prefix hashes prevent reranking and detect later changes but
cannot prove that a historical value equals the value published at that time.

Official sources:

- <https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/supply/current-supply>
- <https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/availability/asseteodcompletiontime>
- <https://gitbook-docs.coinmetrics.io/access-our-data/api>
