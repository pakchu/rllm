# HVPASR-12 Gross9 novelty pass — 2026-08-09

HVPASR-12 passed all frozen structural-overlap limits against every
authoritative Gross9 sleeve without opening BTC execution rows, funding rows,
post-entry returns, PnL, or economic metrics.

Maximum overlap across the complete roster:

- exact-entry Jaccard: `0.000000` (limit `0.10`)
- one-to-one entry share within 6h: `0.243243` (limit `0.35`)
- occupied 5m-bar Jaccard: `0.094980` (limit `0.25`)
- absolute signed-exposure Pearson: `0.139339` (limit `0.35`)

The unchanged singleton is authorized to open only frozen train economics.

- novelty result SHA256: `209225b36565239501e99f0cc7bb573b3f939ac6ba8650006320eba5bd063848`
- novelty manifest hash: `6f91aea787b9b84977aaec50f40af1fbea16965c4d1afc5bfbbbb245e52fd299`
