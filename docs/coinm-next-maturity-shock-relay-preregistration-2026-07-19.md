# CMSR-36 preregistration — COIN-M next-maturity shock relay

## Outcome boundary

CMSR-36 is frozen before any post-entry `BTCUSDT` OHLC, return, funding cash,
PnL, CAGR, or strict MDD is opened. The disclosed 27-cell preflight used only
completed COIN-M front/next source paths, strictly prior feature distributions,
clock counts, sides, months, and contract-pair concentration.

## New mechanism

The rejected roll-migration rule used one five-minute next-contract share/flow
bar and a one-hour continuation on that same COIN-M contract. CMSR instead asks
whether a **two-hour state transition** in the next maturity relays into the
common BTC perpetual:

1. next-contract volume share rises from the first 30 minutes to the last;
2. two-hour next-contract aggressor flow is extreme;
3. next-contract price accepts that flow;
4. next price leads the front contract in the same direction;
5. the front move is at most 60% of the next move.

For each current path, pair-local 30-day thresholds are shifted by 24 bars, so
no prior feature path overlaps the current 24-bar path. The frozen ranks are
q90 share-slope, q80 absolute next flow, and q80 absolute lead shock. Only a
false-to-true transition signals. Side is the accepted next-flow direction.

The source-only grid selected q90/q80/q80 as the lexicographically strictest
cell passing all power and dispersion gates. It has 93 fit clocks, half-year
counts `19/16/21/12/25`, and 65 source-only 2023 clocks split `35/30`.
Every q90/q85 flow cell failed the fixed 2022H1 minimum; no return chose the
q80 fallback.

## Causal execution

- feature source: checksum-bound official Binance Vision COIN-M quarterly
  front/next strip, ending before 2024;
- all 24 rows valid, exact-grid, and same contract pair; no fill or promotion;
- feature available at signal open `t+5m`; leave one full bucket empty;
- enter USD-M `BTCUSDT` at `t+10m`;
- hold exactly 36 five-minute bars / three hours;
- both source contracts retain the 12-hour delivery buffer plus hold and
  latency;
- fixed 0.5x, globally non-overlapping, no stop, target, regime, model, LLM,
  or secondary gate.

## Staged validation

Support must first reproduce the frozen incidence and pass novelty against the
prior single-bar roll clock and independent BTC clocks. Failure opens no
BTCUSDT execution outcomes.

Train `[2020-08-01, 2023-01-01)` requires positive absolute return,
full-calendar CAGR/strict-MDD at least 3, strict MDD at most 15%, at least 90
trades, every half-year positive, 10bp-side stress positive with stress ratio
at least 2.5, weekly-cluster sign-flip `p <= 0.10`, and a 0.25 ratio margin over
each mechanism-removal control.

Only a complete train pass may open 2023. Test requires positive absolute
return, ratio at least 3, MDD at most 15%, at least 60 trades, positive H1/H2
and stress results, and `p <= 0.10`. 2024+ remains sealed. Failure cannot be
repaired by direction, threshold, feature, timing, hold, or regime changes.

Every eventual report includes absolute return, full-clock CAGR, strict MDD,
CAGR/MDD, and trades. Strict MDD includes the global/pre-entry HWM, entry cost,
every held favorable-then-adverse OHLC path, conservative funding boundaries,
virtual adverse-mark exit cost, and actual exit cost.
