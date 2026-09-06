# Macro-flow regime-switch alpha — 2026-09-06

## Decision

**SHADOW CANDIDATE; live disabled.** The candidate was fixed before its recent DB replay and passed that one-shot report, but earlier historical periods were already exposed.

## Formula

- 75%: six-hour aggressive futures flow when its direction opposes the six-hour dollar move. Volatility-targeted and refreshed every 24 hours.
- 25%: long-only 720-hour regime switch. Follow positive trend only with aligned flow; otherwise buy a 24-hour downside displacement in a non-trending regime.
- Sum the signed sleeves, then cap absolute net exposure at 1x. Overlap is allowed; opposing positions offset before costs and risk.

## Evidence at 6 bp/side

| Window | Return | CAGR | strict MDD | CAGR/MDD | Entry episodes | Rebalance orders | Fees / initial |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 15.48% | 15.45% | 9.49% | 1.63 | 100 | 3343 | 4.30% |
| 2025 | 10.40% | 10.41% | 6.57% | 1.58 | 79 | 2871 | 4.53% |
| 2026 H1 | -1.56% | -3.74% | 7.04% | -0.53 | 40 | 1399 | 1.77% |
| 2024–2026 H1 | 25.50% | 9.87% | 10.07% | 0.98 | 219 | 7613 | 11.78% |
| Recent Jun–Sep 5 | 5.63% | 23.05% | 3.98% | 5.80 | 29 | 792 | 1.41% |

Recent 10 bp/side stress return: **4.66%**. Historical combined stress return: **17.56%**.

## Interpretation

The edge is strongest when dollar direction and actual aggressive crypto flow disagree, while the smaller sleeve supplies long exposure only in an established flow-confirmed regime or a non-trending downside displacement. Fixed ML candidates were tested, but formulaic mixtures were more stable and easier to audit.

## Risks

- Historical report periods are not pristine OOS.
- 2026 H1 was negative before recent recovery.
- Recent external-source availability is partial; live publication-time parity is not yet proven.
- Keep this configuration disabled until a shadow/live-parity audit and additional forward data pass.
