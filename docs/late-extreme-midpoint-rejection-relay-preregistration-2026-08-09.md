# LEMRR-16 preregistration

LEMRR-16 is frozen before candidate incidence, Gross9 clocks, execution prices,
funding, or post-entry outcomes are opened. At Tuesday and Friday 00:00 UTC it
reads the preceding exact 1,440 one-minute bars and records the **last** minute
attaining the absolute high and low. A later high rejected to a close below the
full-range midpoint is short; a later low rejected to a close above the
midpoint is long. Entry is 00:05 UTC and hold is sixteen elapsed hours at fixed
0.5 gross.

This measures temporal ordering and failure of the last range extreme, not
ordinary return direction, wick magnitude, Donchian reclaim, VWAP displacement,
run persistence, flow, funding, or a fitted model. The exact weekdays, last-tie
rule, midpoint condition, side, entry, and hold are immutable. Diagnostic
controls cannot be promoted.

The mechanism is intended to concentrate economic value in two-sided volatile
range traversal, but RV20 q90 is not an entry filter. It is opened only after
all four full-calendar economic stages pass, when the unchanged candidate must
have at least eight q90 trades, positive q90 return, and positive residual
against an identical-clock, identical-gross forced-long comparator. Every
gate is terminal at first failure with no repair.
