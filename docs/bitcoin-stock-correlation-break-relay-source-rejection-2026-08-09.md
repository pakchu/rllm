# BSCBR-24 source-support rejection — 2026-08-09

BSCBR-24 was preregistered and its source evaluator was code-reviewed, tested,
committed, and pushed before candidate incidence was computed. The frozen
candidate uses the direction reported by Yae and Tian (2024): a decrease in
the causal BTC-stock conditional correlation maps to long BTC and an increase
maps to short BTC. The implementation uses a pre-2023 fitted, recursively
filtered bivariate GARCH(1,1)-DCC(1,1), an SPY proxy, and the user-required
high-volatility gate. It is an implementation adaptation, not a paper
replication.

## Frozen source-support result

| stage | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 11 | 6 | 5 | 0.4545 | 0.3636 |
| test | 27 | 13 | 14 | 0.4815 | 0.1852 |
| eval | 23 | 6 | 17 | 0.2609 | 0.2609 |
| final | 8 | 2 | 6 | 0.2500 | **0.5000** |

Every minimum-event and side-balance gate passed. Every month-concentration
gate passed except final: `0.50` exceeded the frozen `0.45` maximum. Therefore
`results/bitcoin_stock_correlation_break_relay_support_2026-08-09.json`
records `terminal_source_support_reject`.

No Gross9 comparator rows, execution prices, post-entry returns, funding PnL,
or economic outcomes were opened. The threshold, side, volatility gate,
calendar, hold, source proxy, and subset will not be repaired. Diagnostic
controls remain non-promotable.

External mechanism reference: [Volatile safe-haven asset: Evidence from
Bitcoin](https://www.sciencedirect.com/science/article/pii/S1572308924000706).
