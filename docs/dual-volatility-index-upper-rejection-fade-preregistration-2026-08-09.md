# DVURF-6 preregistration

DVURF-6 is frozen before paired source incidence, Gross9 rows, BTC execution
prices, funding, or post-entry outcomes are opened. At every completed UTC hour
it requires both Binance BVOL and Deribit DVOL to show positive upper-minus-
lower wick rejection. The weaker of the two rejections must rank in its causal
upper quartile, their joint normalized range must rank above its causal median,
and the exact completed BTC hour's absolute return must rank in its causal
upper quartile.

The policy enters at T+5m opposite the completed BTC return and holds six
elapsed hours at fixed 0.5 gross. This is paired within-candle options-index
rejection geometry, not OIFAR's OI flush, cross-venue level/body disagreement,
or BTC wick imbalance. No prior event set or control is reused.

RV20 q90 is not an entry filter and remains a post-stage audit only. Source,
novelty, and strict sequential economics stop at first failure with no wick,
rank, side, clock, hold, subset, or diagnostic-control repair.
