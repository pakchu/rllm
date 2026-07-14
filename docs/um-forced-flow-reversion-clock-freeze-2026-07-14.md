# UMFR-36 clock freeze — 2026-07-14

Freeze selected q=0.80 UMFR-36 support clock before any UMFR outcome
calculation.

- support artifact: `results/um_forced_flow_reversion_support_2026-07-14.json`
- support SHA-256: `22a6cabe015020fa427a660d611917590a95d1212426be1d391476ddce77d3ba`
- selected quantile: `0.80`
- entry: next USD-M five-minute open
- exit: fixed 36 bars after entry

The emitted schedule may contain only the frozen clock schema and must not
contain OHLC, future return, PnL, funding, CAGR, or MDD columns.
