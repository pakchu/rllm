# CVVH-432 source-support rejection — 2026-07-30

## Terminal decision

`CVVH-432` is retired unchanged before novelty. The single authoritative
source-support attempt wrote its claim first, decoded only the retained
hash-bound BVOL/DVOL compressed snapshots, and produced a valid terminal bundle
with `passed=false`.

- attempt-claim SHA-256:
  `5b9a6c1a8aa29172485b95f8c3717138a6926de5061a38f2ae3aa0da438dde08`
- claim hash:
  `dc5b6e10bdeab08048004608ed968b4d71e98e9dc32d1033c3fe9a3f03c5aefb`
- support manifest:
  `2e32d1a75dd249827c86f5445091f63f5fca23a95c3ccdcf922aa3bd0b2a2c16`
- bundle manifest:
  `1e8096f6a3cc733d29bc4785169ed8017f71c4fbcf1e62611c4fa8a96f3ddb1e`
- sealed evaluator commit:
  `c48daaa052c873743491d3e273bc4d9f3a90d45a`

There is no retry, resume, verification replay, threshold repair, delayed-entry
substitution, or control replacement.

## What passed

The primary clock had ample and balanced support:

| Window | Events | LONG | SHORT | Maximum month share |
|---|---:|---:|---:|---:|
| Selection | 249 | 118 | 131 | 7.23% |
| 2023H2 | 82 | 39 | 43 | 19.51% |
| 2024H1 | 88 | 40 | 48 | 20.45% |
| 2024H2 | 78 | 39 | 39 | 19.23% |
| Future25 | 125 | 57 | 68 | 11.20% |
| Future26 | 59 | 29 | 30 | 23.73% |
| Full | 433 | 204 | 229 | 4.16% |

The maximum accepted-entry gap was 23.75 days, the maximum same-side run was
10, and the selection-prefix trace was byte-identical after future append.
Exact joining reported zero missing rows, fills, imputations, tolerance joins,
or nearest matches.

## Why it failed

The preregistration required **every** independent control to have:

- exact-entry Jaccard `< 9/10`; and
- deterministic one-to-one 24-hour maximum matched share `< 19/20`.

All exact-entry Jaccards passed. The Deribit-led control also passed the
24-hour condition. Three controls were nevertheless too temporally close:

| Independent control | Exact Jaccard | 24h matches / denominator | Matched share | Result |
|---|---:|---:|---:|---|
| Deribit-led | 0.00% | 311 / 429 | 72.49% | pass |
| Body-lead-only | 40.15% | 423 / 433 | 97.69% | fail |
| Range-lead-only | 36.64% | 419 / 433 | 96.77% | fail |
| Stale-Deribit | 1.91% | 414 / 433 | 95.61% | fail |

Thus the only failed aggregate check was
`all_four_independent_controls_distinct`. Low exact overlap was not sufficient:
the controls still generated almost the same event opportunities within one
day. This is precisely the source-specificity veto frozen before incidence.

## Research boundary

No prior-volatility comparator rows, Gross9 clock rows, BTC execution prices,
funding rows, returns, PnL, CAGR, or drawdown were opened. Novelty and economic
evaluation are prohibited for this candidate.
