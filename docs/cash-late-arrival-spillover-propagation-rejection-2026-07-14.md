# CLASP-24 rejection — 2026-07-14

## Decision

**REJECT before any 2024+ evaluation.** CLASP-24 passed outcome-blind support,
but failed the frozen pre-2024 return gate after execution OHLC and realized
funding were opened for 2020–2023 only.

- selection artifact: `results/cash_late_arrival_spillover_propagation_selection_2026-07-14.json`
- selection SHA-256: `33800f7bd1fd53533cabaaf7125b6299d9cb74743c0101bcf8883a440423e446`
- deterministic rerun: same SHA-256 on immediate rerun
- 2024, 2025, and 2026 YTD: not opened for CLASP

## Primary outcome

| Window | Abs return | CAGR | Strict MDD | CAGR/MDD | Gross bp | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2020–2022 | -33.26% | -12.61% | 34.90% | -0.36 | -4.86 | 471 |
| select 2023 | -9.38% | -9.38% | 9.95% | -0.94 | -1.58 | 144 |
| 2023 H1 | -5.50% | -10.78% | 6.07% | -1.78 | -3.22 | 74 |
| 2023 H2 | -4.10% | -7.98% | 6.78% | -1.18 | 0.16 | 70 |

## Interpretation

The creative thesis was coherent at the feature-incidence level: late efficient
Spot impulses with remaining USD-M propagation debt were frequent, balanced,
and largely independent of prior alpha clocks. The market outcome did not
support the directional continuation claim. Both train and 2023 were negative,
mean gross underlying move was below the 12 bp gate, and weekly-cluster
significance failed.

Direction flip was not a rescue: it improved gross sign but remained net
negative after costs and funding. Therefore CLASP should not be retuned into a
new gate; doing so would risk turning a failed directional hypothesis into gate
optimization.

## Carry-forward lesson

A support-rich cross-venue shape is insufficient. The next candidate should
avoid assuming venue-spillover continuation from within-bar descriptors alone.
Look instead for a mechanism where the execution leg is explicitly advantaged
by a forced-liquidity, constraint-release, or inventory-repricing effect that
can survive costs without needing a post-hoc gate.
