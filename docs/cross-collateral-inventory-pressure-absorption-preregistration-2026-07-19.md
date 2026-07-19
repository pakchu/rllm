# CIPA-48 preregistration — cross-collateral inventory-pressure absorption

## Outcome boundary

This work unit freezes one source-only BTC alpha before any CIPA post-entry
price, return, funding cash flow, PnL, CAGR, or drawdown is opened. The only
preflight evidence inspected was causal Binance USD-M/COIN-M positioning-source
incidence and event-clock distribution.

## Hypothesis

At an hourly completed metrics anchor, define:

- `R`: six-hour USD-M OI-value growth minus COIN-M OI-contract growth;
- `T`: one-hour median log USD-M taker ratio minus log COIN-M taker ratio;
- `A`, `G`: strict-prior seven-day hourly mid-ranks of `abs(R)`, `abs(T)`.

CIPA activates when `A >= 0.80`, `G >= 0.60`, and `sign(R) = -sign(T) != 0`.
Relative inventory has migrated toward one collateral venue while relative
aggressive flow points against it. The frozen interpretation is passive
absorption: trade `side = sign(R) = -sign(T)`.

This is not a direction repair of failed CCPR. CCPR traded the disjoint
**concordant** quadrant `sign(R)=sign(T)` and faded it. CIPA trades only the
**opposed** quadrant; no CCPR entry is reused or flipped.

## Causal execution

- source: checksum-bound official Binance Vision five-minute USD-M `BTCUSDT`
  and COIN-M `BTCUSD_PERP` positioning metrics;
- anchor: hourly row at UTC `:55`;
- every current 73-row OI path and every one of 168 strict-prior hourly anchors
  must be complete; no fill or stale carry;
- only false-to-true transitions create an episode;
- wait one empty five-minute availability bucket and enter at `t+10m`;
- hold exactly 48 five-minute bars / four hours;
- globally non-overlapping, fixed 0.5x, no stop, target, regime, model, LLM, or
  secondary gate.

## Source-only support

The disclosed incidence-only scan selected the lowest preregistered extreme
rank `0.80` because stricter ranks could not support a statistically useful
test. At the frozen 48-bar hold, the clock has 100 non-overlapping train trades
(45 in 2021 partial, 55 in 2022) and 65 source-only 2023 clocks (44 H1, 21 H2).
Returns did not select the threshold.

Support still must independently verify side balance, month concentration,
exact causality, and low overlap with CCPR. Failure retires CIPA without opening
execution outcomes.

## Sequential performance gates

Train is `[2021-07-08, 2023-01-01)` and must have positive absolute return,
full-clock CAGR/strict-MDD at least 3, strict MDD at most 15%, at least 90
trades, positive return in 2021 partial/2022H1/2022H2, positive 10bp-side stress
return with stress ratio at least 2.5, and weekly-cluster sign-flip `p <= 0.10`.

Only a complete train pass may open 2023. The unchanged test requires positive
absolute return, ratio at least 3, MDD at most 15%, at least 60 trades, positive
H1/H2 and stress results, and `p <= 0.10`. Any failure forbids threshold,
direction, timing, hold, feature, or regime repair. 2024+ remains sealed until
the unchanged train and test both pass.

Every performance report must include absolute return, full-calendar CAGR,
strict MDD, CAGR/MDD, and trade count. Strict MDD includes the global/pre-entry
HWM, entry cost, every held favorable-then-adverse OHLC path, conservative
funding boundaries, virtual adverse-mark exit cost, and actual exit cost.
