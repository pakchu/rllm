# UMFR-36 evaluator freeze — 2026-07-14

Pre-outcome evaluator boundary for UMFR-36. The evaluator source must be
committed and hashed before loading USD-M execution OHLC or realized funding.
Only 2020–2023 train/selection windows may be opened; 2024, 2025, and 2026 YTD
remain sealed unless the frozen pre-2024 gate passes.

Execution assumptions: next-open entry, fixed 36-bar exit, 0.5x leverage,
5 bp fee + 1 bp slippage per notional side, realized funding in
`entry_time <= funding_time <= exit_time`, full-clock CAGR, and strict MDD with
favorable-first then adverse held path excluding the exit bar's later high/low.
