# BTCUSDT funding settlement marks, 2020-2023

Outcome-blind execution-source freeze only. No position return, PnL, CAGR,
drawdown, gate, or 2024+ row was calculated.

- funding events: 4383
- exact funding-record mark overlap: 185
- mark-open backfills: 4198
- maximum funding timestamp jitter: 47 ms
- maximum overlap mark error: 13.484319911 bp
- maximum implied funding-cash error: 0.001348431991 bp/notional

All settlement marks use one uniform rule: the open of the official Binance
BTCUSDT USD-M 8h mark-price kline whose canonical boundary contains the
returned funding timestamp. The 185 non-empty historical
`fundingRate.markPrice` values are validation overlaps only. Returned funding
timestamps are retained exactly for settlement inclusion.

Official endpoint: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data>

Manifest hash: `3b447e94d9dbb6ba4994713df565b7d6ec5b38c26c4b568ad7f4e102fefc299c`
