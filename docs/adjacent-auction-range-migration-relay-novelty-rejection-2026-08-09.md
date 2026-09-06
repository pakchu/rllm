# AARMR-8 terminal Gross9 novelty rejection

AARMR-8 passed source support with `168/340/280/163` events, then failed the
pre-registered Gross9 novelty gate before any execution prices, funding, or
post-entry economic outcomes were opened.

Exact-entry Jaccard, occupied-bar Jaccard, and absolute signed-exposure
correlation all passed. The one-to-one ±6h matched share failed against every
Gross9 sleeve, ranging from `0.5135` to `0.5862`, above the immutable `0.35`
ceiling. Two executions produced result SHA-256
`186e2d5f28ac3bdff44a94f566561bb1d359bde0935f65af2d10b12afcf7e241`.

AARMR-8 is rejected unchanged. Sparsifying the hourly clock, tightening ranks,
altering onset, shifting entry, changing hold, or selecting a subset after
seeing overlap would repair the frozen candidate and is forbidden. Economics,
controls, funding, RV20, and post-entry prices remain unopened.
