# HVSPSR-12 terminal source-support rejection

HVSPSR-12 produced `0/0/0/0` train/test/eval/final events and is terminally
rejected at its first gate. The merged 2023-01-01 through 2026-08-01 daily
source frame contained 1,308 dates, but only 28 dates had exact, coherent
1,440-minute Binance spot and perpetual paths. Consequently neither frozen
causal rank reached its 180-day minimum history requirement.

Two executions reproduced the same terminal result:

- valid exact joint source days: `28`
- finite spot-participation ranks: `0`
- finite perpetual-variation ranks: `0`
- source-feature SHA-256: `9449f4bcdde3be8c2790d64a9122c55de058203bdd46c30fa7bd339cb4b87551`
- empty primary-clock SHA-256: `ac370f03f9940433a2bd98c7a5817225a84469306a622c26e437cdbfbc8c68b0`
- result SHA-256: `55135f6e802f5ae2e47efc3b84a4531c82f744830c8af7101ac1f1b4d9cbec3d`
- result manifest hash: `cdb99f2b48e3297c8dd0e49aad7ce3bceda1ce1acbcd4e58dce5b5f8027dfa28`

HVSPSR-12 remains unchanged. Gross9 rows, execution prices, funding,
post-entry outcomes, economics, and RV20 remain unopened. No source rule,
rank history, threshold, direction agreement, side, timing, hold, subset, or
diagnostic control may be repaired or promoted.
