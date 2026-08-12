# HVCAPCR-6 source rejection — 2026-08-12

The frozen metadata-only source evaluator found that `bars_binance_premium`
contains only `BTCUSDT` one-minute rows during the preregistered 2023-01-01
through 2026-08-01 window. All six required alt symbols (`ADAUSDT`, `BNBUSDT`,
`DOGEUSDT`, `ETHUSDT`, `SOLUSDT`, and `XRPUSDT`) are absent.

No premium values, candidate incidence, Gross9 rows, execution prices, funding,
post-entry returns, or PnL were opened. The frozen evaluator reproduced the
result byte-for-byte with SHA-256
`250b49efed5ce3525d67a54db57d1b8eccaa4d156abad6ff6a928ad26d48c55f`.

HVCAPCR-6 is therefore terminally rejected for physical source-axis absence.
Substituting BTC premium, funding, price basis, a smaller symbol set, or another
venue would change the preregistered mechanism and is prohibited.
