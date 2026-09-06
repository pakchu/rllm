# HVBFRR-12 terminal source-support rejection

HVBFRR-12 produced `0/0/0/0` train/test/eval/final events. The frozen source
SQL assigned all timestamps from 18:00 through the following 17:59 to one
shifted UTC-day group, while the source-valid contract required exactly the
preregistered 480-row 18:00–02:00 block. Consequently all 1,308 candidate
days were invalid and no causal fit or rank became available.

The failure was reproduced without modifying the evaluator:

- source-feature SHA-256: `081075bde09f83a0eb1bcbd390a8b678e3c58f19363e6e2c6273ef45cf7d81a7`
- empty primary-clock SHA-256: `4699f3e44593d60224951b872c503cb4b4a128dd88b89d7bfac29749d4a3e3c0`
- result SHA-256: `30d52bbd875b76768466fc48f9e32f8b5b36c40bf08777e8ae4e5bd624f6533a`
- result manifest hash: `51d49ec9ad5f2eb4ac68386728a759bdcb86ffb46e917eebff6a278710164bcf`

Because source incidence was already opened, adding an hour selector would be
a prohibited source/block repair. HVBFRR-12 is rejected unchanged. Gross9
rows, execution prices, funding, post-entry outcomes, economics, and RV20
remain unopened; no symbol panel, block, factor, fit, scale, threshold, side,
clock, hold, subset, comparator, or control may be repaired or promoted.
