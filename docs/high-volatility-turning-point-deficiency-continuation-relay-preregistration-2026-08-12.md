# HVTPDCR-8 preregistration

`HVTPDCR-8` tests whether a July-like high-variation BTC auction with unusually
few strict ordinal turning points continues in the direction shared by its
completed eight-hour and final-two-hour returns.

The signal object is only the order relation among each triple of adjacent
five-minute closes. It does not use return magnitudes. This separates it from
volatility-threshold directional changes, absolute return curvature, sign-run
lengths, recurrence geometry, and Lempel-Ziv complexity. No previous event set
or diagnostic control is reused.

The policy is fixed at three eight-hour UTC boundaries, entry five minutes
later, an eight-hour hold, 0.5 gross, causal lower-20% turning-point rank, and
causal upper-35% realized-variation rank. Source support, Gross9 novelty, and
strict sequential economics are terminal gates. RV20 q90 is only an audit
after an unchanged all-stage pass.
