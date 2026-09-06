# STCSR-12 terminal source-support rejection

STCSR-12 was rejected before Gross9 rows, execution prices, or post-entry
returns were opened.

The perpetual source supplied all 1,440 required minute rows on every daily
block, but the spot source had only 26 fully populated daily blocks across the
1,307-row decision panel. Most historical spot rows lack the preregistered
`number_of_trades` field, so neither the spot-count-share rank nor the realized
variation rank reached the required 126 valid prior observations. The frozen
candidate consequently produced 0/0/0/0 train/test/eval/final events.

Backfilling trade counts, replacing transaction count with volume, weakening
the exact-path requirement, or shortening rank history would alter the frozen
rule. The candidate is terminally rejected unchanged and its diagnostic
controls remain non-promotable.
