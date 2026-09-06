# HVSPSR-12 preregistration — 2026-08-09

HVSPSR-12 tests cash-sponsored high-volatility price discovery. For each
completed UTC day it computes Binance BTCUSDT spot quote turnover as a share of
combined spot and perpetual quote turnover. The share must rank in the causal
upper 25%, perpetual realized variation in the upper 35%, and the completed
spot/perpetual day returns must have one common strict sign. The position follows
that sign from the exact next-day 00:05 UTC perpetual open for twelve hours.

This is not spot/perpetual price leadership, Upbit participation rotation,
premium-index activity, or implied-volatility ignition. Source, Gross9, and
economic gates stop at the first failure; controls cannot be promoted and RV20
q90 remains a post-stage audit only.
