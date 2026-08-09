# CAFACR-8 source rejection — 2026-08-09

CAFACR-8 is terminally rejected unchanged at source support. The configured
Postgres table contained `3,750` BTCUSDT funding rows but zero rows for each of
ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, and DOGEUSDT over the frozen
source window. Consequently there were no valid seven-symbol common
settlements and the candidate produced zero events in train, test, eval, and
final.

- Preregistration SHA-256: `b80da6fd1df1bd2148340261b87d322151e248af8bd7a5c077751493b699c96c`
- Source evaluator commit: `ebff6544`
- Source result SHA-256: `a5bca7bcc2f647588cafb229239cb5e5ec361d52b06aa0dc1f2b3244ee2e6b16`
- Primary empty-clock SHA-256: `dbe18974be8eb7e2a36600be924e50ccc3ecdab0a00afb56b66413ec0394a12b`

The complete source evaluator was rerun and every result, feature, manifest,
clock, and diagnostic-control artifact was byte-identical. BTC price rows,
Gross9 clocks, execution prices, RV20, funding PnL, and post-entry outcomes
were not opened. No substitute source, universe reduction, majority change,
control promotion, or gate repair is permitted.
