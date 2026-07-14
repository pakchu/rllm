# Binance cross-collateral radial book-shell audit — 2026-07-14

## Purpose and evidence boundary

The official `bookDepth` files report cumulative depth at ±1…5% from the
moving book reference. Treating those five bands as independent duplicates
nearer liquidity. This outcome-blind builder instead converts every complete
raw snapshot into non-overlapping radial shells before any five-minute
aggregation:

```text
shell_1 = cumulative_1
shell_k = cumulative_k - cumulative_(k-1), k=2..5
```

The result covers `0–1, 1–2, 2–3, 3–4, 4–5%` on both sides of USD-M
`BTCUSDT` and COIN-M `BTCUSD_PERP`. It does not load market prices, returns,
labels, PnL, CAGR, MDD, or 2024+ data.

Shell decomposition and a later outer-to-inner wavefront hypothesis are
feature engineering, not exchange-native concepts or established alpha.

## Sources and research limits

Official source semantics:

- Binance public data: <https://github.com/binance/binance-public-data>
- maintainer confirmation of nominal 30-second `bookDepth` sampling:
  <https://github.com/binance/binance-public-data/issues/437>
- maintainer confirmation of ±1…5 bands and depth/notional fields:
  <https://github.com/binance/binance-public-data/issues/447>
- live update-ID/order-book semantics absent from the archive:
  <https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

Primary literature supports non-flat book shape, deeper-level relevance, and
resiliency, but does not establish a universal radial wavefront:

- hump-shaped book depth: <https://arxiv.org/abs/cond-mat/0203511>
- empirical shape away from the touch: <https://arxiv.org/abs/0801.3712>
- meso-scale flow/resiliency and deeper shape:
  <https://arxiv.org/abs/1708.02715>
- state-dependent and sometimes opposite-side resiliency:
  <https://arxiv.org/abs/1602.00731>
- bilateral replenishment evidence: <https://arxiv.org/pdf/1003.3796>

Therefore “outer-to-inner cancellation wave” remains a falsifiable inference,
not a cited market law.

## Physical boundary and replay

- inclusive start: `2023-01-01 00:00:00` UTC
- exclusive end: `2024-01-01 00:00:00` UTC
- output grid: 105,120 five-minute rows
- source-complete rows: 101,649
- archives requested: 730 venue-days
- archives reproduced: 727
- USD-M gaps: 2023-02-08 and 2023-02-09
- COIN-M gap: 2023-09-25
- post-2023 rows requested: false
- outcomes opened: false

Each available ZIP was downloaded into memory, verified against the official
checksum, and compared with the frozen archive SHA/raw-row/snapshot record.
No ZIP was retained. The original 28-column cumulative-depth/timing panel was
replayed exactly, with only the already-audited `1e-10` decimal CSV tolerance.

## Snapshot-to-bar statistics

For side-total cumulative 5% depth `T_i`, shell depth `H_(k,i)`, and adjacent
snapshots `i-1,i` inside one accepted five-minute bar:

```text
denom_i = 0.5 * (T_i + T_(i-1))
flow_(k,i) = (H_(k,i) - H_(k,i-1)) / denom_i
```

The panel stores, per venue/side/shell:

- `share_median`: median `H_k / T`;
- `flow_net`: sum of signed normalized flow;
- `flow_add`: sum of positive flow;
- `flow_withdraw`: sum of negative-flow magnitudes;
- `flow_churn = flow_add + flow_withdraw`;
- `flow_efficiency = abs(flow_net) / flow_churn`, or zero at zero churn.

Bars retain the frozen coverage rule: at least eight complete snapshots, first
offset no later than 60 seconds, last offset no earlier than 240 seconds. There
is no fill, interpolation, or nearest-time join.

Across all complete rows, venues, sides, and shells:

| statistic | minimum | p01 | median | p99 | maximum |
|---|---:|---:|---:|---:|---:|
| `share_median` | 0.000000 | 0.026924 | 0.182654 | 0.462159 | 0.890936 |
| `flow_net` | -0.943341 | -0.052735 | 0.000062 | 0.052146 | 0.802729 |
| `flow_add` | 0.000000 | 0.000001 | 0.014015 | 0.111921 | 1.043254 |
| `flow_withdraw` | 0.000000 | 0.000000 | 0.013560 | 0.113297 | 1.036609 |
| `flow_churn` | 0.000000 | 0.000115 | 0.029188 | 0.212136 | 1.978619 |
| `flow_efficiency` | 0.000000 | 0.001321 | 0.235135 | 1.000000 | 1.000000 |

The maximum numerical errors in `net = add - withdraw` and
`churn = add + withdraw` were below `5.1e-12`.

## Frozen output

- builder commit: `ca14ea3`
- builder SHA256:
  `8d343830e4d51596e7b369f303a7ba3fc807dbecb5f19193028dcf43c8c67a1c`
- panel:
  `data/binance_cross_collateral_book_shells_btc_2023/BTC_cross_collateral_book_shells_5m_2023.csv.gz`
- panel SHA256:
  `ead931ec8ce2bbd73c946b8660e16d7750ce73051e60ce4989467a7c5bc68342`
- panel size: 92 MiB
- rows / columns: 105,120 / 148
- manifest:
  `results/binance_cross_collateral_book_shells_btc_2023_manifest.json`
- manifest SHA256:
  `1b5519143d58f62ef3e8b6d9e22f012f80197a59903509041aca24252ed04521`
- build wall time / max RSS: 3:00.06 / 964,376 KiB

WSL remained at 294 GB used, below the user's 300 GB ceiling.

## Limitations and safe next use

- Percent bands move with the book reference and are not fixed price levels.
- The archive is a 30-second snapshot sample, not order-event L2; flow is net
  visible shell-mass change, not literal additions/cancellations.
- Hidden/RPI liquidity and sub-30-second activity are unobserved.
- Median shell shares need not sum exactly to one because each shell is
  median-aggregated separately.
- UM and CM raw units must not be compared before product-local causal
  normalization.
- Any wavefront rule must be frozen from shell incidence before reading returns
  and must be rejected if it merely relabels CLV, CCLH, or PDF-10 clocks.
