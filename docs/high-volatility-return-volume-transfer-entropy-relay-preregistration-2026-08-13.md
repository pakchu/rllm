# HVRVTE-8 preregistration

`HVRVTE-8` is a singleton, outcome-blind candidate for July-like volatile BTC markets.

At each exact 01:00, 09:00, and 17:00 UTC decision, the completed prior eight hours are aggregated into 96 five-minute BTCUSDT perpetual auctions. Turnover is discretized as above or below the within-block median, and each strictly nonzero return becomes a binary sign. The frozen statistic is empirical conditional mutual information `I(turnover_state[t-1]; return_sign[t] | return_sign[t-1])`. Every turnover/previous-sign conditioning cell must have at least five samples.

The signal direction is the strict sign of the previous-sign-frequency-weighted difference between next-up probabilities following high and low turnover. The information statistic must rank in its causal upper quartile and completed variation in its causal upper 35 percent using 270/180 strict-prior histories. Only a fresh eligibility onset enters at `D+5m` for an eight-hour hold and 0.5 gross exposure.

Unlike linear turnover-to-next-return correlation, same-bar covariance, low-frequency coherence, or continuous temporary-impact regression, this candidate measures nonlinear categorical information transfer conditional on return-sign persistence. It uses no taker split, funding, OI, premium, fitted outcome, reused event set, or promoted control.

Source support, Gross9 novelty, sequential economics, stress, split-half, and conditional RV20 gates remain unchanged. The first failed gate is terminal and no formula, discretization, threshold, side, clock, hold, subset, or control may be repaired.
