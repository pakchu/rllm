# CCLH v1 frozen 2023 result — 2026-07-14

## Decision

**CCLH v1 is rejected.** The evaluator source was committed in `8ad4ebf` and
its exact SHA was recorded by the pre-outcome freeze manifest committed in
`66a7484`. Only then were the calendar-2023 research windows opened.

- result:
  `results/cross_collateral_liquidity_hysteresis_selection_2026-07-14.json`
- result SHA256:
  `475c927b5440ff08fe75b2b1c095c06271e9e11a13fef965e710a0d5eda37582`
- frozen evaluator SHA256:
  `f44fb011fa229c84424143d6da5fed0c06f6d4adfedd71fdca51353c257a80f3`
- still sealed: full 2024, full 2025, and 2026 YTD

No threshold, state direction, confirmation length, holding period, overlap
rule, stop, or regime was changed after returns opened. No Gemma/RL policy was
attached.

## Frozen CCLH results

All figures use 0.5x leverage, 5 bp fee plus 1 bp slippage per notional side,
next-five-minute-open entry, scheduled-open exit after 144 bars, full split
clock CAGR, and favorable-first/adverse-second held-path strict MDD.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| train 2023 H1 | +7.5064% | +15.7265% | 7.2915% | 2.1568 | 71 | 0.21404 |
| select 2023 H2 | -0.1841% | -0.3652% | 7.4233% | -0.0492 | 96 | 0.50446 |
| 2023 Q1 | +5.3236% | +23.4290% | 7.2915% | 3.2132 | 33 | 0.27004 |
| 2023 Q2 | +2.0724% | +8.5814% | 4.1894% | 2.0484 | 38 | 0.30683 |
| 2023 Q3 | -4.3049% | -16.0288% | 6.4391% | -2.4893 | 48 | 0.95501 |
| 2023 Q4 | +4.3061% | +18.2203% | 3.0580% | 5.9583 | 48 | 0.08117 |

H1 is positive but fails the required ratio and weekly significance. H2 is
slightly negative, with Q3 loss offset almost completely by Q4 recovery. The
state is more stable than the failed one-hour CLV impulse, but it is not a
qualified alpha.

## Same-clock controls

| policy | H1 abs | H1 CAGR/MDD | H2 abs | H2 CAGR/MDD |
|---|---:|---:|---:|---:|
| CCLH | +7.5064% | 2.1568 | -0.1841% | -0.0492 |
| exact reverse | -15.1270% | -1.7225 | -10.9434% | -1.6027 |
| always long | -2.2250% | -0.4788 | -10.3455% | -1.6027 |
| always short | -6.6803% | -1.3527 | -0.8486% | -0.2806 |
| permuted sign | -8.1093% | -0.9588 | -4.3541% | -1.0062 |
| 1h price momentum | +8.4967% | 2.4532 | +0.3508% | 0.1068 |
| CLV-overlap removed | +9.0331% | 3.0253 | +1.7912% | 0.5951 |

The exact reverse and directional controls clearly lose, so the full-depth
joint state contains some conditional information. However, the preregistered
price-momentum control has a better worst-half ratio than CCLH, and neither it
nor CLV-overlap removal reaches the required H2 ratio or significance. They
are controls observed on the same 2023 outcomes and cannot be promoted as OOS
alphas.

## Structural diagnostics

Every different-clock structural variant loses in both halves:

| diagnostic | H1 abs | H1 ratio | H2 abs | H2 ratio |
|---|---:|---:|---:|---:|
| USD-M only | -10.0638% | -1.5547 | -24.3839% | -1.6872 |
| COIN-M only | -0.9941% | -0.2591 | -13.6007% | -1.5563 |
| cross pressure only | -1.1956% | -0.2183 | -4.1372% | -0.5787 |
| cross elasticity only | -6.2546% | -0.9985 | -15.7054% | -1.6512 |

This supports the *interaction* claim—both cross-contract geometry legs are
needed—but interaction alone is insufficient for monetization.

## Frozen qualification failures

1. H1 CAGR/strict-MDD is below 3;
2. H1 weekly-cluster p-value is not below 0.10;
3. H2 absolute return is non-positive;
4. H2 CAGR/strict-MDD is below 3;
5. H2 weekly-cluster p-value is not below 0.10;
6. Q3 absolute return is non-positive;
7. CCLH does not beat the price-momentum control on worst-half ratio.

All support-count and strict-MDD ceilings pass. The failure is economic and
statistical, not insufficient incidence.

## Consequence

CCLH v1 is closed without a 2024 data build. The 2024+ outcomes remain sealed.
No Q3-specific gate, alternative hold, or overlap filter may rescue this name.

The useful negative evidence is narrower: transient vacuum impulses fail;
persistent joint full-depth geometry is directionally better, but its edge is
time-varying and below the significance/risk threshold. The next mechanism
must model *when displayed depth is credible* using an independent observable,
not optimize a calendar regime or the CCLH gate on these opened outcomes.
