# CAFAER-12 terminal Gross9 novelty rejection

CAFAER-12 passed source support with 35/109/57/60 events, then failed Gross9
novelty before execution prices, funding, returns, or PnL were opened.

All four novelty metrics passed against four of five Gross9 sleeves.  Against
`fresh_kimchi_fx`, exact-entry Jaccard was 0, occupied-bar Jaccard 0.0786, and
absolute signed-exposure correlation 0.0672, but the one-to-one six-hour
matched share was 0.4324 versus the 0.35 ceiling.  The sparse daily mechanism
is still too close in clock time to that existing sleeve.

Retiming, changing the daily session, filtering events, or altering the hold
would repair the frozen candidate.  CAFAER-12 is rejected unchanged and
economics remain sealed.
