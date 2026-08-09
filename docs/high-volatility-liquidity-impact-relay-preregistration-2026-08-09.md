# HVLIR-8 preregistration

HVLIR-8 is a singleton high-volatility BTC liquidity-impact continuation relay.
It is frozen before source incidence, Gross9 rows, execution prices, funding,
or post-entry outcomes are opened.

At each completed UTC hour it measures the absolute one-hour return per unit of
quote turnover, with turnover normalized by its own strictly prior seven-day
median.  The current value must newly cross into its strictly prior top 20%
while prior-24-hour realized variation is in its strictly prior top 35%.
HVLIR-8 follows the completed hour's strict return sign for eight elapsed
hours, entering five minutes after the decision boundary at fixed 0.5 gross.

The source gate requires `8/12/12/8` events, minority side share at least
`0.20`, and maximum monthly concentration at most `0.45`.  Gross9 novelty and
the unchanged strict economic contract are evaluated only after a source
pass.  The first failed gate retires HVLIR-8 unchanged; no threshold, baseline,
rank, direction, hold, clock, or subset repair is allowed.
