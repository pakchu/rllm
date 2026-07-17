# FADC-21 preregistration — 2026-07-17

## Hypothesis

FADC-21 is a same-collateral, first-order delta-matched relative-value sleeve:
it trades Binance USD-M BTCUSDT perpetual against the USD-M current-quarter
future. It does not use BTC direction, REX, OI, kimchi/FX, Markov, tree, LLM or
a manual regime. The expected edge is delivery basis minus a strictly prior
21-settlement funding forecast.

This is not the rejected CCBS twelve-hour cross-collateral snapback and not the
rejected spot/perpetual z-score compression. Both legs are linear USD-M; the
economic anchor is delivery convergence plus actual perpetual funding.

## Frozen causal policy

- At funding time `t`, wait for both complete `[t,t+5m)` candles; decide at
  `t+5m` and execute both legs at the `t+10m` open.
- `funding_ann = mean(last 21 settled rates including t) * 3 * 365`.
- `basis_ann = log(quarter_close/perp_close) * 365 / DTE`.
- `gap = basis_ann - funding_ann`.
- Enter only for DTE 14–80 days and expected edge through `delivery-24h` of at
  least **30 bp**. Positive gap means long perp/short quarter; negative means
  short perp/long quarter.
- After 24 hours, exit when expected edge is at most 5 bp or the gap changes
  sign. Always exit at least 24 hours before delivery and wait 24 hours before
  re-entry.

The 30 bp gate is economic, not outcome-fit: at gross 1x the two-leg round trip
cost is 12 bp at base costs and 20 bp under the frozen 10 bp/notional-side
stress.

## Strict ledger

Both legs receive the same frozen BTC quantity, so entry gross is exactly 1x.
Funding uses the frozen settlement-mark proxy for
`entry_time <= funding_time < exit_time`. Strict MDD uses the global/pre-entry
HWM, favorable-before-adverse independent leg extrema, funding in timestamp
order, and hypothetical two-leg liquidation costs. CAGR covers the full
wall-clock period; absolute return is mandatory in every result table.

## Outcome boundary and gates

The preregistration reads no price outcome or PnL. All three bound inputs end
before 2024. Outcome-blind support must first find at least
24 pre-2023 entries with year/half/direction and
month-concentration floors. Only then may 2021–2022 PnL open. A failure rejects
the candidate without opening 2023. A stage-1 pass may open 2023 exactly once;
2024 remains sealed until the 2023 and executed-PnL orthogonality gates pass.

Passing history is still not live-ready: continuous-contract symbol resolution,
tick/step rounding, atomic two-leg execution, margin/liquidation accounting and
forward source/slippage parity remain hard blockers.

Protocol hash: `39dc96a719c4fe001eb547f1a3c8be8134a391934997efae049cf06bb487b2f0`
