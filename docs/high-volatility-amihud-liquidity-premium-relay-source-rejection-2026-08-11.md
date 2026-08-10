# HVALPR-24 source rejection — 2026-08-11

The frozen Amihud liquidity-premium candidate failed its first scientific gate.

- Train: `22` events, `2` long and `20` short; minority share `0.091 < 0.20`.
- Test: `53` events; eval: `40`; final: `33`.
- Every event-count and month-concentration check passed, as did side balance outside train.
- The sole failure was train side balance, showing that elevated 2023H2 variation coincided overwhelmingly with lower-tail rather than upper-tail illiquidity innovations under the frozen definition.
- Gross9 clocks, execution prices, funding values, returns, PnL, CAGR, and MDD remained unopened.
- Terminal action: reject unchanged. No tail, variation, reference window, side, hold, or clock repair is authorized.
