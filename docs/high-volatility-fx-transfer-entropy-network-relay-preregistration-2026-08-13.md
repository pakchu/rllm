# HVFXTE-12 preregistration

`HVFXTE-12` is an outcome-blind singleton for July-like volatile BTC markets.
For each completed weekday 13:00-21:00 UTC session, six major FX pairs are
encoded into canonical US-dollar return signs on exact common one-minute
timestamps. Every ordered edge uses empirical transfer entropy
`I(sign_i[t-1]; sign_j[t] | sign_j[t-1])`; the antisymmetric edge is forward
minus reverse entropy.

The unique node with the largest strictly positive net outgoing score must lead
at least four peers, its score must rank in its causal upper quartile, and
completed BTC 24-hour variation must rank in its causal upper 35 percent. At a
fresh onset, BTC trades opposite the source node's completed canonical-dollar
direction from 21:05 UTC for twelve elapsed hours at 0.5 gross exposure.

This differs from linear lagged FX correlation, contemporaneous synchronization,
and endpoint breadth. No BTC return direction, fitted outcome, prior event set,
funding, OI, premium, or promoted control enters the signal. Source support,
Gross9 novelty, sequential economics, stress, split-half, and conditional RV20
gates remain unchanged; the first failure is terminal without repair.
