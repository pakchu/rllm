# CSPR-12 frozen pre-2024 result — 2026-07-14

## Verdict

**Rejected without repair.** The exact evaluator frozen in commit
`04baee67f25c8d26301e5b50ec646444261ba3e9` opened only 2020–2023 and CSPR-12
failed every economic qualification family. Calendar 2024, calendar 2025, and
2026 YTD remain sealed.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly-cluster p |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 train | -48.72% | -19.95% | 48.78% | -0.41 | 763 | 1.00000 |
| 2023 selection | -5.74% | -5.75% | 5.74% | -1.00 | 87 | 0.99999 |
| 2023 H1 | -1.81% | -3.62% | 2.05% | -1.77 | 41 | — |
| 2023 H2 | -4.00% | -7.79% | 4.14% | -1.88 | 46 | — |

Both 2023 halves were negative. The primary also failed to beat every frozen
control on minimum train/selection CAGR-to-strict-MDD.

## Falsification result

The exact direction flip was also net negative, but was less bad than the
primary:

| split | flip absolute return | flip CAGR/MDD | flip trades |
|---|---:|---:|---:|
| 2020–2022 train | -22.54% | -0.28 | 763 |
| 2023 selection | -4.44% | -0.87 | 87 |

Solving the frozen cost multiplier algebraically after rejection gives the
mean **gross underlying** move on the exact paired clock:

- train primary / flip: `-5.40 bp / +5.40 bp`;
- 2023 primary / flip: `-1.57 bp / +1.57 bp`;
- 2023 H1 primary: `+3.12 bp`;
- 2023 H2 primary: `-5.75 bp`.

At `0.5x`, the frozen round-trip execution charge requires roughly 12 bp of
underlying move merely to break even. The flip therefore does not constitute a
tradable reverse alpha, and its sign also changed between the 2023 halves.

## Root cause and stop decision

The Spot/perpetual execution-centroid relation selected a real, repeatable
microstructure event clock, but not a persistent one-hour directional edge.
The signal's gross move was small relative to execution cost, direction was
unstable across regimes, and broader component clocks decayed further under
turnover. This is a mechanism failure, not a threshold shortage.

No quantile, hold, side, centroid rule, stop, or cost assumption will be
repaired on CSPR-12. The next candidate must use the new Spot source for a
different economic object and must be preregistered before its returns are
opened.

## Artifact

- result SHA-256:
  `4aa08bda84bd7e2c58f15046a4c126c5e16b79294f991730207d7a85f2e89b44`
- evaluator-freeze manifest SHA-256:
  `3fbe0ed6e5eb5c4f473575ef4affb64a658177ae42053f6669891bb535a712ff`
