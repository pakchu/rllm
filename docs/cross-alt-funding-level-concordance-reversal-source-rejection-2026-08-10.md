# CAFLCR-8 source-support rejection

CAFLCR-8 is terminally rejected before candidate incidence, Gross9, or market
outcomes. The read-only `funding_rates_binance` query returned `4,203` rows for
the frozen seven-symbol universe over 2023-01-01 through 2026-08-01, but every
row belonged to `BTCUSDT`:

```text
BTCUSDT 4203
ETHUSDT/SOLUSDT/BNBUSDT/XRPUSDT/DOGEUSDT/ADAUSDT 0
```

Therefore no exact common seven-symbol funding settlement exists. All four
calendar source clocks contain zero events, and source support fails closed.
Execution prices, post-entry returns, PnL, Gross9 rows, and RV20 were not
opened.

The universe and funding-level concordance mechanism are frozen. Substituting
symbols, venues, funding sources, price/flow proxies, or a BTC-only diagnostic
would create a repaired successor and is forbidden.
