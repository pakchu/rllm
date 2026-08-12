# HVRNAR-8 Gross9 novelty rejection

HVRNAR-8 passed frozen source support but failed the mandatory structural
novelty gate before any execution price, return, funding, PnL, or economic
metric was opened.

The one-to-one six-hour matched share exceeded the frozen 0.35 maximum for:

- `fresh_kimchi_fx`: 0.4324
- `markov_transition_long`: 0.3824

All exact-entry Jaccards were zero and occupied-bar/exposure metrics passed,
but every Gross9 sleeve had to pass every metric. The candidate is therefore
terminally rejected unchanged. The lattice increment, acceptance window,
variation gate, direction, clock, and hold may not be repaired or retested.
