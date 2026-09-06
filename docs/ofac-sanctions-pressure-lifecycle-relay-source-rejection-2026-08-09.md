# OSPLR-24 terminal source rejection

OSPLR-24 passed event-count and monthly-concentration gates in every split, but
failed the frozen TEST side-balance gate. Counts were `9/49/21/20`; 2024 TEST
contained 6 LONG and 43 SHORT events, a minority share of `0.1224` versus the
required `0.20`.

No Gross9 clock, execution price, funding row, post-entry return, PnL, CAGR, or
MDD was opened. The pressure/relief taxonomy, ambiguity rule, volatility gate,
availability time, hold, and source subset are not changed or rerun.
