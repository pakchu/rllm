# ORPLHC-6 economic evaluator freeze — 2026-08-11

The sequential evaluator and outcome-blind `load_clock_allow_empty` behavior are frozen before opening execution prices, funding PnL, or post-entry returns. Stage order is train, test, eval, final. Every stage uses fixed 0.5 gross, exact held funding, 6bp/10bp per notional side, full-calendar CAGR, and strict held-five-minute favorable-then-adverse MDD. First failure is terminal.
