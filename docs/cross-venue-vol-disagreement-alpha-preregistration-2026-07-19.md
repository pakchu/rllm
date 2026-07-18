# Cross-venue volatility disagreement alpha preregistration — 2026-07-19

## Scope and evidence status

This is an outcome-blind BTC alpha battery. The source ends before 2024 and no
future trade return, fee/funding PnL, CAGR, or MDD is opened at this stage. The
economic labels below are hypotheses, not claims about trader identity.

## Frozen hypotheses

1. **BVOL-rich move fade**: when Binance BVOL is unusually high relative to
   Deribit DVOL and BTC has already made an unusually large completed 4h move,
   trade against that move. The proposed mechanism is faster venue-local
   volatility repricing followed by price-impact exhaustion.
2. **DVOL-rich move follow**: when Deribit DVOL is unusually high relative to
   Binance BVOL and BTC has made an unusually large completed 4h move, trade
   with that move. The proposed mechanism is slower convergence toward a
   broader derivatives-volatility repricing.

Neither hypothesis uses options skew, contract-level options flow, investor
identity, funding, premium, OI, Kimchi, FX, REX, an LLM, or another alpha gate.

## Frozen clock

- signal frequency: hourly, after both volatility candles and BTC price history
  have closed;
- relative-volatility feature: `log(Binance BVOL / Deribit DVOL)`;
- price feature: completed BTCUSDT 4h close-to-close return;
- thresholds: 30d (720h) rolling quantiles with the current hour excluded and a
  7d (168h) minimum history;
- relative-volatility tails: 80th/20th and 90th/10th percentiles;
- absolute-price-move tails: 80th and 90th percentiles;
- holds: 12h, 24h, 48h;
- execution: five minutes after signal availability; fixed elapsed-time exit;
- overlap: globally non-overlapping within each candidate.

The grid has exactly 24 cells. Support requires at least 24 scheduled trades,
at least 8 in both 2023Q3 and 2023Q4, at least 25% on each side, and no month
containing more than 35% of a candidate's trades. Support pruning may remove
structurally sparse clocks; return-based direction or parameter repair is
forbidden after this freeze.

## Forward protocol

Only support-passing clocks enter the sealed evaluator. Selection is confined to
2023H2. One final policy must be frozen before 2024 is built or opened. The
ordered windows are 2024 test, 2025 evaluation, and 2026 report-only holdout.
All performance stages must use next-executable 5m prices, two-sided costs,
realized funding, full-calendar CAGR, held-path high/low strict MDD, trade counts,
and weekly cluster inference.
