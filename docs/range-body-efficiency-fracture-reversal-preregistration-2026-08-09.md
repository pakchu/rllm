# RBEFR-8 preregistration

RBEFR-8 is frozen before incidence, Gross9 clocks, execution prices, funding,
or post-entry outcomes are opened. At every exact two-hour UTC boundary it
uses the preceding 96 completed five-minute bars. It divides summed
Parkinson-style high-low variance by summed candle-body variance and enters
only when that ratio crosses into its strictly-prior top 15%. The policy fades
the final completed two-hour impulse from decision+5m for eight elapsed hours
at fixed 0.5 gross.

The mechanism targets volatile auctions with large intrabar travel but weak
directional body efficiency. It is distinct from wick-side imbalance,
semivariance, bipower jump size, variance-ratio dependence, close location,
funding, and flow. RV20 q90 is not an entry filter and remains sealed until all
four full-calendar economic stages pass.

Every gate is terminal at first failure. No estimator, rank, history, onset,
impulse, side, entry, hold, subset, or diagnostic control may be repaired or
promoted.
