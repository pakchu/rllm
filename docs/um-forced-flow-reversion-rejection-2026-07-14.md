# UMFR-36 rejection — 2026-07-14

## Decision

**REJECT before any 2024+ evaluation.** UMFR-36 passed support and novelty, but
failed the frozen pre-2024 return gate after 2020–2023 execution OHLC and
realized funding were opened.

- selection artifact: `results/um_forced_flow_reversion_selection_2026-07-14.json`
- selection SHA-256: `54306d1f2391e95c9bd1cda3573d11252c1f261a2e73ab0fb130f67ecb813c73`
- deterministic rerun: same SHA-256 on immediate rerun
- 2024, 2025, and 2026 YTD: not opened for UMFR

## Primary outcome

| Window | Abs return | CAGR | Strict MDD | CAGR/MDD | Gross bp | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train 2020–2022 | -16.09% | -5.68% | 24.21% | -0.23 | 6.82 | 636 |
| select 2023 | -10.20% | -10.21% | 11.16% | -0.91 | 4.35 | 279 |
| 2023 H1 | -4.12% | -8.13% | 4.81% | -1.69 | 4.79 | 116 |
| 2023 H2 | -6.34% | -12.20% | 7.51% | -1.62 | 4.03 | 163 |

## Interpretation

The forced USD-M-flow reversion thesis had better directional sign than CLASP:
primary gross underlying move was positive in train and selection, while the
exact direction flip was worse. However, the edge was only about 4–7 bp before
costs/funding, below the frozen 12 bp mean-gross gate, and strict MDD exceeded
15% on train. This is not enough for a deployable alpha.

The lesson is sharper than CLASP: cross-venue microstructure can identify
something real, but the magnitude is too small at this bar/hold level. The next
search should either find a larger forced-flow event class or use this weak
signal only as one component inside a higher-magnitude trigger, not as a
standalone alpha.
