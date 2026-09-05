# Meaningful alpha combination study

## Scope
New research branch, not a continuation of the rejected PPOSM action router.
Discover dense BTCUSDT trend, pullback, breakout, flow-exhaustion, carry, and
regime-reversion strategies; compare fixed Ridge/HGB/ExtraTrees forecasts and
netted portfolios. Do not deploy or change live configuration.

## Selection and evidence
- Fit ML only on 2020-03-01 through 2022; purge labels that mature in 2023.
- Select only using 2023 full year and half-year stability.
- Freeze five finalists before computing their 2024, 2025, and 2026 reports.
- The historical reports are previously exposed data, NOT clean new OOS.
- Report 0, 6, and 10 bp/side plus funding. No fee-ratio or high-frequency gate.
- Exact next five-minute open after completed hourly observation. Same-symbol
  long/short sleeve positions offset BEFORE orders, fees, and net risk limits.
- Hourly conservative envelope used for screening; finalists replay intrabar
  five-minute envelopes. Peak-before-trough is a conservative assumption where
  within-bar ordering is unavailable.
- Funding mark missing: settlement five-minute open proxy, separately disclosed.
- No liquidation, tick-level slippage/capacity, or external fresh-forward proof.

## Completion
Emit full candidate inventory, fixed finalist reports (CAGR, MDD, net return,
entry episodes, rebalances, turnover, fees, funding), reproducible code/tests,
and a research-only config. A negative result is reported, not silently tuned
against the report years. Promising candidates are not automatically live alphas.
