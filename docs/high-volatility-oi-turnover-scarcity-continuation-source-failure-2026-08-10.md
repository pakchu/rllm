# HVOTSC-8 source execution failure — 2026-08-10

HVOTSC-8 was rejected at its first source/scientific failure. The frozen source
evaluator raised `RuntimeError: HVOTSC invalid OI key` before producing a
candidate clock or source-support artifact.

The preregistered OI input requires a finite causal `observed_at` for every
5-minute observation. In the frozen 2023-01-01 through 2026-08-01 source
window, all 376,570 matching `open_interest_binance` rows have
`observed_at IS NULL` (null share 1.0). Substituting `ts` would change the
preregistered causal-availability rule and is therefore forbidden.

No execution prices, post-entry returns, PnL, funding values, or Gross9 rows
were opened. Novelty and economic evaluation were not started. The terminal
decision is `terminal_source_execution_reject_no_repair`.

Machine-readable evidence is in
`results/high_volatility_oi_turnover_scarcity_continuation_source_execution_failure_2026-08-10.json`.
