# DVURF-6 terminal Gross9 novelty rejection

DVURF-6 passed source support with `124 / 230 / 224 / 161` events, then failed
the preregistered Gross9 novelty gate before any execution price, funding row,
post-entry outcome, or economic metric was opened.

Exact-entry Jaccard, occupied-bar Jaccard, and absolute signed-exposure
correlation passed against every sleeve. The one-to-one ±6-hour matched share
failed against `frozen_annual_rank7` at `0.413793` and
`markov_transition_long` at `0.411765`, versus the fixed `0.35` ceiling.

An immediate rerun reproduced result SHA-256
`faed62f66d3a60bcf2468efc3d4ecfb6ecf6c91608520f56371d7dad61a640f1`.
No rank, rejection geometry, opportunity clock, reservation, side, hold,
subset, or diagnostic control is repaired or promoted.
