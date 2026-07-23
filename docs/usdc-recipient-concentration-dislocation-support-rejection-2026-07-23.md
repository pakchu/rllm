# URCD-72 source-support result — terminal rejection

## Decision

**URCD-72 is retired unchanged at the source-support stage.** The frozen
amount-weighted USDC mint-recipient concentration transition did not have
enough pre-2024 incidence to form a train/selection candidate.

The run did not authorize comparator timestamps, novelty evaluation, BTC or
funding outcomes, or an economic evaluator. Threshold, lookback, direction,
hold, materiality, and scheduler repair are forbidden after incidence, so this
result closes URCD-72 rather than beginning a parameter search.

## Reproducible artifacts

- source-only clocks:
  `data/usdc_recipient_concentration_dislocation_2021_2023/urcd72_support_clocks_2021_2023.csv.gz`
  - SHA-256:
    `ad9617ec5af0c0189aa384a49ab9244e957758f7c8abe71b6b61e911b7663ea1`
  - rows: 186 across the primary and seven causal controls
- report:
  `results/usdc_recipient_concentration_dislocation_support_2026-07-23.json`
  - SHA-256:
    `648825052812a8f436b8e7743973f3d6edcf4e013a767082103c98707e66f998`
  - manifest hash:
    `1ad500d2c60a18ce12345804beb8465bfdf00febd60e86d21f1eeedc94d2d685`

The run completed in 14.88 seconds with a 203,620 KiB maximum resident set.

## Source-only evidence

| Split | Accepted | LONG | SHORT | Max month share | Max quarter share | Max gap |
|---|---:|---:|---:|---:|---:|---:|
| Train 2021–2022 | 0 | 0 | 0 | `1/1` | `1/1` | N/A |
| Selection 2023 | 14 | 2 | 12 | `2/7` | `5/7` | 29.5 days |

Only 5 of 21 frozen source-support checks passed. The primary's first accepted
decision was `2023-08-21T18:00:00Z`; at that point it had only 124 valid prior
same-hour daily windows. The exact 180-day reference rule requires at least 120
valid windows, while earlier source history was too sparse under the frozen
minimum of four mints and three recipients per 24-hour window. Consequently:

- train incidence was zero rather than at least 80;
- selection incidence was 14 rather than at least 30 and occurred entirely in
  2023H2;
- selection was highly one-sided at 2 LONG versus 12 SHORT;
- month and quarter concentration exceeded the frozen limits; and
- train permutation-selectivity ratios were the empty-set fallback `1/1`.

Selection-period recipient and amount permutation controls were sufficiently
different from the primary, but those partial passes cannot rescue missing
train support.

## Source and outcome boundary

- promoted source physical rows scanned: 266,362;
- pre-seal rows timestamp-screened: 266,360;
- eligible pre-2024 USDC mint rows decoded: 99,033;
- post-2023 source value rows decoded: zero;
- comparator four-field rows decoded: zero;
- BTC market, funding, future-return, PnL, absolute-return, CAGR, and strict-MDD
  values decoded: zero;
- network calls: zero.

Absolute return, CAGR, strict MDD, and CAGR/MDD are therefore **N/A**, not zero.
The next search must use a new preregistered mechanism identity. URCD-72 cannot
be repaired, inverted, or gated into another result.
