# CRRC-72 preregistration — 2026-07-17

## Mechanism

CRRC-72 looks for a contemporaneous radial compression of the Binance BTC
order book. On one side, both USD-M and COIN-M must show strong inner-shell
adds, outer-shell withdrawals, positive inner net depth, and non-extreme
inner flicker. Bid agreement is long, ask agreement is short, and a two-sided
conflict is flat.

Every rolling threshold uses the previous 8,640 rows only, requires 6,912
finite prior rows, and excludes the current row. A completed signal bar is
available at `t+5m`; the trade enters USD-M BTCUSDT at `t+10m`, holds six
hours, and is greedily non-overlapping within each calendar quarter.

## Frozen support choice

Fourteen incidence-only cells were inspected without any price or return.
Among cells passing >=150 events, >=50 per half, >=25 per quarter, balanced
sides, and concentration gates, the deterministic rule maximized add,
withdrawal, and net quantiles in order, then minimized the flicker quantile.
It selected `(0.85, 0.75, 0.55, 0.85)` with 156 scheduled events. No outcome
was used.

Before any outcome was opened, a canonical replay corrected eight scratch
incidence counts for looser cells whose earlier temporary scheduler was not
the frozen `t+10m`, quarter-contained clock. The selected cell and its 156
events were unchanged.

## Novelty boundary

This is distinct from REX, OI/funding/premium/Kimchi signals and differs from
PDF-10, CCLH, RLWC-144, and the signed near-pressure score. It is not claimed
to be a globally new family because adjacent book-depth experiments already
exist. Their causal clocks must be replayed and pass frozen overlap gates
before 2023 returns can be opened; PnL correlation is tested only after a
standalone pass.

## Evaluation and stop rule

Strict MDD uses the global/pre-entry HWM, held OHLC with
favorable-before-adverse ordering, exact funding cash, all entry/exit costs,
and hypothetical liquidation costs. CAGR spans the full calendar including
warm-up and idle periods. The singleton must be positive in every 2023
quarter, in both side sleeves, under 10bp stress and a +5m delay, while
reaching CAGR/strict-MDD >=3 and strict MDD <=15%. The first failed gate
retires it without sign, threshold, hold, or feature repair and keeps later
years sealed.

## Live limitation

Binance Vision archives are not a live feed. Promotion requires a live UM/CM
local-order-book collector that reproduces cumulative +/-1..5% depth at
nominal 30-second snapshots and the exact 5m transforms, completeness rules,
quantiles, and clock. Official references:

- https://github.com/binance/binance-public-data
- https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly

Protocol hash: `205e0df485b202ea9ad67fee677c4796e9a65a6fbee04fd87b9ba8a4ba4321b0`
