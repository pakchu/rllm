# RCR-144 support result — 2026-07-14

## Frozen decision

RCR-144 was preregistered in commit `889fd5d` before the formal support run.
The run loaded only frozen calendar-2023 shell/credibility panels for feature
availability and prior-clock independence. It loaded no BTC price, return, PnL,
CAGR, MDD, label, or 2024+ row.

- support artifact:
  `results/radial_composition_rotation_support_2026-07-14.json`
- artifact SHA256:
  `0e801542c29a964ea969ac4cc4317f98f89d95639683687fb605b9799fcd2d2e`
- preregistration source SHA256:
  `6ae77c65150c05b99782fb7ca376ed905424767366a4906c2bca690a094b3b65`
- preregistration document SHA256:
  `4e462518bacd2ea7e8464bcff9a968c6f906642907d92acd839aef7898d10c43`
- wall time / maximum RSS: `20.81 s / 558,464 KiB`

**Decision: RCR-144 passes the outcome-blind support gate.** This authorizes
writing and separately freezing one exact 2023 return evaluator. It is not a
profitability result and does not authorize 2024+ access.

## Availability and incidence

| statistic | result | gate |
|---|---:|---:|
| finite score rows | 97,414 | at least 90,000 |
| finite Q1 / Q2 / Q3 / Q4 | 19,073 / 25,700 / 26,179 / 26,462 | at least 15,000 each |
| strong bars | 18,884 | informational |
| strong Q1 / Q2 / Q3 / Q4 | 4,254 / 4,204 / 5,131 / 5,295 | at least 500 each |
| strong long / short share | 49.54% / 50.46% | at least 35% each |
| scheduled trades | 646 | at least 120 |
| scheduled H1 / H2 | 300 / 346 | at least 45 each |
| scheduled Q1 / Q2 / Q3 / Q4 | 132 / 168 / 170 / 176 | at least 20 each |
| scheduled long / short share | 49.69% / 50.31% | at least 35% each |
| largest quarter share | 27.24% | at most 40% |

## Independence

| comparator | ±12-bar Jaccard | RCR-event match share |
|---|---:|---:|
| CCLH | 0.03567 | 4.33% |
| PDF-10 | 0.07472 | 13.31% |

Feature Spearman correlations were:

- CCLH cross-pressure: `-0.01606`;
- CCLH cross-elasticity: `-0.01439`;
- PDF-10 credibility: `+0.49939`;
- PDF-10 display: `+0.05964`.

The maximum absolute correlation, `0.49939`, remains below the frozen `0.60`
limit. All event and feature independence gates passed.

## Locked next step

The evaluator must reproduce this exact 646-trade support clock, use the frozen
0.5x/cost/full-clock-CAGR/held-path-strict-MDD contract, and be committed and
hash-frozen before any outcome is read. No RCR threshold, direction, hold,
feature, schedule, or event may change. Calendar 2024–2026 remains sealed until
all 2023 return gates pass.
