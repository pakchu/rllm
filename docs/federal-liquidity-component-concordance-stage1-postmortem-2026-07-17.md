# FLCC-1 Stage1 rejection postmortem — 2026-07-17

## Verdict

FLCC-1 is rejected at the preregistered 2020–2022 Stage1 gate. No candidate
qualified, no winner was selected, and the 2023 BTC outcome window remains
physically unopened.

| Candidate | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Weekly p |
|---|---:|---:|---:|---:|---:|---:|
| FLCC-H4-Q60 | 33.97% | 10.24% | 28.20% | 0.36 | 108 | 0.1729 |
| FLCC-H4-Q65 | 23.65% | 7.33% | 28.20% | 0.26 | 99 | 0.2283 |
| FLCC-H8-Q60 | -2.82% | -0.95% | 33.77% | -0.03 | 97 | 0.4532 |
| FLCC-H8-Q65 | 18.61% | 5.85% | 25.95% | 0.23 | 94 | 0.2575 |

The best headline candidate, H4-Q60, also remained profitable at the frozen
10 bp/notional/side stress cost (+28.32% absolute return), so transaction cost
was not the main failure. Its risk-adjusted and statistical evidence was far
below the frozen requirements: strict MDD 28.20% versus 15%, CAGR/MDD 0.36
versus 3, and weekly-cluster p=0.173 versus 0.025.

## Temporal instability

| Candidate | 2020 absolute return | 2021 | 2022 |
|---|---:|---:|---:|
| FLCC-H4-Q60 | -7.76% | 23.06% | 16.69% |
| FLCC-H4-Q65 | -9.41% | 23.78% | 9.03% |
| FLCC-H8-Q60 | 8.36% | -23.06% | 6.36% |
| FLCC-H8-Q65 | 8.47% | -12.17% | 13.61% |

The sign of the useful horizon flips by year: H4 fails in 2020 while H8 fails
in 2021. That is regime dependence, not a stable unconditional liquidity
impulse. Adding a BTC-price gate after seeing these outcomes is forbidden and
would turn this family into a post-hoc repair.

## Mechanism check

H4-Q60 also failed the preregistered component-dominance check. Its CAGR/MDD
was 0.36, below both `net_only` (0.52) and
`component_concordance_only` (0.48). The conjunction removed useful events
rather than strengthening the mechanism. This means the specific “net tail +
2-of-3 component agreement” construction is not supported as a new alpha.

All direction-flip, one-release-delay, and hash-random controls failed their
complete qualification batteries. That avoids a false interpretation, but it
does not rescue the primary.

## Leakage and isolation evidence

- Physical outcome window parsed: `[2020-01-01, 2023-01-01)` only.
- Market rows parsed: 315,648.
- Funding rows parsed: 3,288.
- 2023 execution rows parsed: 0.
- 2023 funding rows parsed: 0.
- 2023, 2024, 2025, and 2026 YTD remain sealed.
- Stage1 result manifest: `9ae2641cd4b9fa63a4f6ba6328181e84e4c72d812263f5266b1c102cc2cb9847`.
- Stage1 JSON SHA-256: `10dc911ad06c7e523d612ff34675421388fefb94fa93e157bfac7e93bd1d82a6`.

## Research decision

Do not tune thresholds, horizon, side mapping, hold, or add gates inside
FLCC-1. Archive it as a clean negative result. The next candidate should use a
different causal clock and economic mechanism rather than another threshold
variant of aggregate Fed net liquidity.
