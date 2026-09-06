# CLDCR-6 terminal source-support rejection

CLDCR-6 was frozen before exact incidence and produced `7/16/13/4`
train/test/eval/final events. Test and eval passed all source gates, but train
missed the immutable minimum by one event (`7 < 8`), final had only four
events (`4 < 8`), and final maximum-month share was `0.50 > 0.45`.
Side balance passed in every nonempty stage.

Two executions reproduced the same artifacts:

- clock SHA-256: `7d419b4db2269c4a9cd08d05a9bb79afd835daa738b832557b2177ffccab71d2`
- result SHA-256: `cb703bc92629d01896c8eaaa18e1b9440a5770cbc513d2c88539241884df91e7`

CLDCR-6 is rejected unchanged. Changing height modulo, q90, prior history,
five-of-six coherence, onset, confirmation, embargo, side, or hold would be a
forbidden repair. Gross9 rows, execution prices, funding, outcomes, economics,
controls, and RV20 remain unopened.
