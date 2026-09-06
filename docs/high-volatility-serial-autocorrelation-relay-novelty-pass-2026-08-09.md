# HVSAR-12 Gross9 novelty pass — 2026-08-09

HVSAR-12 passed all four frozen structural-overlap limits against every
authoritative Gross9 sleeve. No BTC execution rows, funding rows, post-entry
returns, PnL, or economic metrics were opened.

Maximum observed overlap across the complete Gross9 roster:

- exact-entry Jaccard: `0.009174` (limit `0.10`)
- one-to-one entry share within 6h: `0.123457` (limit `0.35`)
- occupied 5m-bar Jaccard: `0.055864` (limit `0.25`)
- absolute signed-exposure Pearson: `0.062875` (limit `0.35`)

The unchanged candidate is authorized to open only its frozen train economics.

- novelty result SHA256: `de9c8afc7e96e7aaf4c3187e543974c2b82ff44c3b88088acf57f9256f73ad7c`
- novelty manifest hash: `3a7a2254d8991f88d7464981511c61c58326185b0b96818f77d818b641d0b5a2`
