# OPDR-24 feature-source freeze — 2026-07-19

## Decision

The post-preregistration feature sources are acquired, validated, and hash
frozen. This stage opened no BTCUSDT execution OHLC, future return, funding cash
flow, strategy PnL, CAGR, or drawdown.

The exact preregistration commit is
`c4b9c4f22d24783f8176897ec4159a5ae1f6e68c`. Future source acquisition occurred
only after that commit.

## Binance BTCBVOL

- official source: Binance Vision daily `BTCBVOLUSDT` one-second BVOLIndex
  archives;
- requested range: `[2023-06-20, 2026-07-01)`;
- complete hourly grid: **26,568 rows**;
- valid checksum-backed hours: **23,771**;
- invalid hours: **2,797**, retained as unavailable with no fill;
- verified daily archives: **1,081**;
- unavailable daily archives: **26**, retained as invalid days;
- combined SHA-256:
  `40c0d1aecb15119e7fab31aae4108c632d25de136401a6896896852c7f4032b1`;
- acquisition manifest SHA-256:
  `6c62a389cbc8d6524444f5e5fe1d2945c20bafa9fa707b7f2a4801c74221a7e4`.

Valid coverage by calendar year:

| Year | Rows | Valid | Valid share |
|---:|---:|---:|---:|
| 2023 partial | 4,680 | 4,537 | 96.94% |
| 2024 | 8,784 | 7,987 | 90.93% |
| 2025 | 8,760 | 7,329 | 83.66% |
| 2026H1 | 4,344 | 3,918 | 90.19% |

Every valid hour has 3,600 source seconds and a valid OHLC envelope. Invalid
hours have all BVOL values cleared. The completed-hour value is available only
at hour end.

## Deribit DVOL

- official endpoint: `public/get_volatility_index_data`;
- requested range: `2023-06-20` through `2026-07-01`;
- hourly rows: **26,569**;
- no duplicate or missing hourly timestamp;
- candle values join on `close_time`, never on candle open time;
- DVOL SHA-256:
  `26b768f81c2fa49fd59d9f1a173a829329a7ed5bb94c2d71af7c33b46f4f02cf`;
- acquisition summary SHA-256:
  `22e0a6e311fcad34a51f5b0844b7807e7c851eecc4a367f89b7a7d6ce438bf74`.

The downloader includes the boundary candle whose close exceeds the declared
end. The support builder must use `close_time < 2026-07-01`, so that extra row
cannot enter a signal.

## Premium and sealing

The already frozen premium-only source remains bound by SHA-256
`7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9`.
This source-freeze stage verified its contract hash but did not read its values;
the source-only support builder is the first OPDR stage allowed to aggregate
them.

- BTC execution rows loaded: `0`
- funding rows loaded: `0`
- source-freeze report SHA-256:
  `5801b8b819f4951a141700a0249c9cd421ab88922931dc1336ec15de8d1c7883`
- source-freeze manifest hash:
  `43aa11881204627e779ae5e1e562f9e9ab50485a89f4b0eaa6337b934a07741c`

The next stage may calculate only causal feature clocks and support/novelty
statistics. Candidate returns remain sealed.
