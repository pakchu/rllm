# HVQHPS-12 preregistration — 2026-08-13

`HVQHPS-12` tests the public price-volume-state component of scheduled BTC
quarter-hour opening flow. A strictly-prior rolling OLS separates the contribution
of the paper's fixed TI28 block from twelve opening-flow lags; only an extreme
public-component onset in elevated completed variation can signal. The current
opening flow and all post-entry information are excluded.

The external basis is Kim and Hansen (2026), *The Quarter-Hour Effect: Periodic
Algorithmic Trading and Return Predictability in Cryptocurrency Futures*
([arXiv 2607.09426](https://arxiv.org/abs/2607.09426)). The paper reports that the
public-signal component dominates at twelve hours across all six contracts. Its
TI28 appendix fixes the momentum, trend, volume, and volatility menu. This
experiment additionally freezes candle aggregation, exact formulas, a 15-minute
information exclusion, causal estimation, and tail/variation onset gates.

This component and horizon were identified before the terminal economics of
`HVQHLF-4`; this is not a repair of that lagged-flow candidate. Source incidence,
Gross9 rows, execution prices, returns, and PnL remain unopened. The first failed
gate is terminal and diagnostic controls cannot be promoted.
