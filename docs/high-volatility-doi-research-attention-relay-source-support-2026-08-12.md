# HVDRA-24 source support

Date: 2026-08-12

## Decision

The unchanged preregistered HVDRA-24 DOI research-attention clock passes every
source-support gate and may advance only to Gross9 novelty. No execution price,
funding settlement, post-entry return, PnL, Gross9 row, CAGR, or drawdown was
opened.

## Causal source

- Authority: Crossref REST `/works` DOI registry.
- Retrieval: fixed `query.title` searches for `bitcoin`, `cryptocurrency`, and
  `cryptoasset`, cursor pagination to the declared total, then exact local title
  grammar and DOI deduplication.
- Retrieved rows: 13,545 including query overlap; 13,279 unique DOI identities.
- Eligible records: exactly one title, one of three registered research types,
  and latest `deposited` UTC day equal to the original `created` UTC day.
- Eligible DOI deposits: 5,577.
- Signal: daily eligible count minus the count seven days earlier.
- Decision: D+2 at 12:00 UTC; entry five minutes later; 24-hour hold.
- Volatility gate: strict-prior 270-day BTC variation midrank, minimum 180,
  current excluded, rank at least 0.65.

## Gate result

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| 2023H2 train | 43 | 25 | 18 | 0.4186 | 0.4186 |
| 2024 test | 124 | 62 | 62 | 0.5000 | 0.1694 |
| 2025 eval | 88 | 42 | 46 | 0.4773 | 0.2045 |
| 2026 through July final | 72 | 40 | 32 | 0.4444 | 0.3194 |

Required minima are 8/12/12/8, minority share at least 0.20, and maximum month
share at most 0.45. Every check passes.

After the network snapshot was fixed, two complete reruns produced identical
source manifest, feature panel, clock, and support artifact bytes. The final
support artifact SHA-256 is
`f0e03ed636f984253bb11e2bcd0b93dd4731b3ad7128c3585437a73cf13c82ea`.

Economic outcomes remain sealed. The next authorized action is the frozen
Gross9 structural novelty comparison.
