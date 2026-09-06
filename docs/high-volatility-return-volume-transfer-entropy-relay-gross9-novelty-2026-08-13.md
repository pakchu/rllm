# HVRVTE-8 Gross9 novelty

HVRVTE-8 passed every frozen structural-novelty comparison against the complete
Gross9 sleeve roster. Worst-case metrics were:

- exact-entry Jaccard: `0.007407` versus `0.10` maximum;
- one-to-one six-hour matched share: `0.145833` versus `0.35` maximum;
- occupied five-minute-bar Jaccard: `0.045818` versus `0.25` maximum;
- absolute signed-exposure Pearson correlation: `0.037947` versus `0.35` maximum.

Two independent evaluator runs reproduced the canonical report byte-for-byte.
Its SHA-256 is
`8c1e2c0151c40b9290a6062b4891d9e5314448e016d7b737c9e090b44f1ad3c7`.
The evaluator opened 213 candidate clock rows and 1,064 authoritative Gross9
clock rows, but no BTC execution price, return, funding, PnL, or economic row.
The unchanged candidate advances to sequential economic evaluation.
