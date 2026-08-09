# PPCSR-6 preregistration

PPCSR-6 is frozen before exact incidence, Gross9 clocks, BTC execution prices,
funding, or post-entry outcomes are opened. At exact five-minute UTC boundaries
D, the current completed premium-index minute must have the opposite strict
sign from all 60 immediately preceding completed premium minutes. No premium
magnitude threshold, fitted rank, BTC price, funding, OI, flow, or block signal
is used.

The final premium row is conservatively available at D+1 second. Its
availability is ceiled to G=D+5m, the complete bucket [D,D+5m) is left empty,
and BTC entry occurs at G+5m=D+10m for a six-hour hold at fixed 0.5 gross.
Controls cannot be promoted. RV20 q90 remains a later audit and every gate is
terminal.
