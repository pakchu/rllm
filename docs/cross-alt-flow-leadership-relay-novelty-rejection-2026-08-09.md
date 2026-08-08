# CAFLR-6 terminal Gross9 novelty rejection

CAFLR-6 passed source support with 185/377/370/191 events, then failed the
frozen Gross9 novelty gate before any execution prices, funding rows, returns,
or PnL were opened.

Exact-entry Jaccard, occupied-bar Jaccard, and absolute signed-exposure
correlation passed against every Gross9 sleeve.  The one-to-one six-hour
matched share failed against all five sleeves, ranging from 0.4375 to 0.6757
versus the 0.35 ceiling.  The hourly flow-disagreement clock is structurally
too close in time to the dense Gross9 roster even though its held exposure is
otherwise weakly correlated.

Changing the clock, hold, latency, event subset, or threshold would repair the
frozen candidate.  CAFLR-6 is rejected unchanged and economics remain sealed.
