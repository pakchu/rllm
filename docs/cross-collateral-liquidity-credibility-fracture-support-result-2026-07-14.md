# PDF-10 outcome-blind support result — 2026-07-14

## Verdict

**PASS support and independence; returns remain unopened.**

The preregistered PDF-10 source and document were committed in `2737ab2`
before this run. The support program loaded only the frozen calendar-2023
book-credibility panel. It did not load BTC prices, future returns, labels,
PnL, CAGR, MDD, or any 2024+ row.

## Frozen artifacts

- support result:
  `results/cross_collateral_liquidity_credibility_fracture_support_2026-07-14.json`
- support result SHA256:
  `9a3001db640ec8041d885645d33f11dd6075276685eb22f8ae3c618363d3099a`
- preregistration source SHA256:
  `8947050c990b5638f6d8b2e952f252289ddef6c92f85fb13f75001fe721e6e28`
- preregistration document SHA256:
  `e7bf6dc9b2c7bf1ec2d560ea4e1dff8018cb6c28177fa012b729d2e0a2ca1dfe`
- credibility panel SHA256:
  `45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429`
- credibility manifest SHA256:
  `f530f472765c8cb56bf564efd346c734e2404e072b87e2ff8dc3b84e303c30f7`

Runtime was 10.26 seconds with maximum RSS 363,100 KiB. WSL usage remained
294 GB, below the 300 GB ceiling.

## Support incidence

The frozen clock contained 682 same-side two-bar confirmations before
non-overlap scheduling: 334 bullish and 348 bearish confirmed rows.

| period | scheduled trades |
|---|---:|
| 2023 Q1 | 96 |
| 2023 Q2 | 122 |
| 2023 Q3 | 145 |
| 2023 Q4 | 228 |
| H1 | 218 |
| H2 | 373 |
| total | 591 |

- long share: 48.9002%
- short share: 51.0998%
- largest-quarter share: 38.5787%
- all preregistered support floors: **passed**

These are event counts only, not profitability statistics.

## Independence from failed CCLH geometry

The credibility panel replayed CCLH's frozen 167-event positions and sides
exactly:

`e90079d95b111f95ce64459c42d17e4286636a1a2854ed948e8ada497a13dfa7`

| diagnostic | observed | maximum | verdict |
|---|---:|---:|---|
| event Jaccard within ±2 bars | 0.01337 | 0.15 | pass |
| event Jaccard within ±12 bars | 0.04264 | 0.30 | pass |
| maximum absolute feature Spearman | 0.46475 | 0.60 | pass |

PDF credibility itself had only `-0.02868` correlation with CCLH pressure and
`-0.03638` with CCLH elasticity. The larger `-0.46475` value came from the
display component versus CCLH pressure, which is expected because both include
displayed depth; the PDF action additionally requires opposite firmness.

## Locked next step

Support does not establish alpha. The next step is to implement the exact
fee/slippage, full-clock CAGR, held-path strict-MDD, weekly-cluster test, and
same-clock controls specified in the preregistration. That evaluator must be
committed and separately hash-frozen before any PDF-10 return outcome is read.
Calendar 2024+ remains sealed.
