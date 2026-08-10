# High-volatility intraday hour reversal boundary — 2026-08-11

- Candidate: `HVIHR-1`, one immutable policy and no grid.
- Published basis: Wen, Bouri, Xu, and Zhao (2022), Table 2, reports Bitcoin `r3 -> r19` with coefficient `-0.16` and Newey-West `t=-3.50`.
- Frozen mapping: `r3 = 02:00-03:00 UTC`; `r19 = 18:00-19:00 UTC`.
- Frozen adaptation: causal upper-tail prior-24h variation, upper-tail predictor magnitude, fade side, entry at `18:05 UTC`, exit at `19:00 UTC`.
- The repository-wide slug/history/mechanism search found no prior candidate using this exact hour pair.
- No candidate incidence, Gross9 row, post-entry price, funding value, return, PnL, CAGR, or MDD was opened before this preregistration.
- First scientific failure is terminal. Diagnostic controls cannot be promoted.
