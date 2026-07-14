# PDF-10 frozen 2023 selection result — 2026-07-14

## Verdict

**REJECT PDF-10 v1. Do not open 2024+.**

The evaluator source was committed at `4afbf15`, then separately frozen at
`cfcb988` with source SHA256
`513570e06529bd65966e505a2fc005f160417992fa52d36122401419cad9c252`.
Only the preregistered calendar-2023 H1/H2 and quarter windows were opened.

Frozen selection artifact:

- `results/cross_collateral_liquidity_credibility_fracture_selection_2026-07-14.json`
- SHA256:
  `663d1b4a832fd87cdc92de8569915d5441a6880736b6a46092615eea03822f24`

## PDF-10 results

All statistics use 0.5x, 5 bp fee plus 1 bp slippage per notional side,
next-five-minute-open entry, two held bars, scheduled-open exit, complete split
clock CAGR, and held-path strict MDD.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| 2023 H1 | -13.09% | -24.66% | 13.19% | -1.87 | 218 | 1.00000 |
| 2023 H2 | -20.36% | -36.36% | 20.56% | -1.77 | 373 | 0.99999 |
| Q1 | -4.72% | -17.81% | 5.14% | -3.46 | 96 | 0.99901 |
| Q2 | -8.79% | -30.87% | 8.87% | -3.48 | 122 | 1.00000 |
| Q3 | -9.75% | -33.47% | 9.95% | -3.36 | 145 | 1.00000 |
| Q4 | -11.75% | -39.12% | 12.06% | -3.24 | 228 | 0.99586 |

Every quarter lost money. H2 also exceeded the 15% strict-MDD cap. The result
fails absolute return, CAGR/MDD, MDD, weekly-cluster significance, quarter
positivity, and control-dominance gates.

## Same-clock controls

| policy | H1 return | H1 mean/trade | H2 return | H2 mean/trade |
|---|---:|---:|---:|---:|
| PDF-10 | -13.09% | -6.43 bp | -20.36% | -6.09 bp |
| exact reverse | -11.46% | -5.57 bp | -19.84% | -5.91 bp |
| always long | -10.57% | -5.12 bp | -17.91% | -5.28 bp |
| always short | -13.95% | -6.88 bp | -22.23% | -6.72 bp |
| sign permutation | -12.57% | -6.15 bp | -18.17% | -5.36 bp |
| 5m price momentum | -12.01% | -5.86 bp | -16.27% | -4.74 bp |

The account-level round-trip drag is approximately 6 bp. Solving the frozen
cost multiplier for the pre-cost mean shows PDF versus exact reverse had only
about `-0.43/+0.43` bp per trade in H1 and `-0.09/+0.09` bp in H2. Thus the
failure is **not** a hidden profitable exact reversal. Directional information
is near zero at the 10-minute horizon and cannot pay the execution contract.

## Root cause and locked consequence

The new credibility statistics successfully produced a well-supported event
clock independent of CCLH, but display/firmness disagreement was mostly a
market-quality state, not a sufficiently large directional markout. Reversing
the side, relaxing thresholds, or extending this exact clock after seeing the
result would be post-outcome repair and is prohibited.

PDF-10 v1 is closed. Calendar 2024, 2025, and 2026 credibility outcomes remain
sealed. The next experiment must change the predictive object rather than gate
or flip PDF-10: use firmness to forecast **move magnitude / adverse-selection
risk**, and obtain direction from a separately causal mechanism whose expected
markout is large enough to clear the fixed 6 bp round-trip cost.
