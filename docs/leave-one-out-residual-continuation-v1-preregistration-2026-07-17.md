# LORC v1 preregistration — 2026-07-17

## Evidence boundary

LORC is an explicitly **derived** family. The 2023–2024 LORE direction-flip
diagnostic is research/selection data and is not claimed as confirmation.
Calendar 2025 post-entry returns remain unopened and are the first single-policy
confirmatory test. Calendar 2026 remains sealed unless every 2025 strategy gate
passes.

## Orthogonal mechanism

LORC holds no BTC position. It removes a leave-one-out alt factor, then buys the
12-hour residual winner and shorts the residual loser only when taker flow fails
to confirm both tails. Cross-leg weights neutralize the causal rolling factor
beta at gross 1.0. The mechanism and data source are distinct from BTC REX/OI,
funding/premium, Kimchi/FX, Markov, tree, and LLM alphas.

## Frozen single policy

- six Binance USD-M alts: ETH, SOL, BNB, XRP, ADA, DOGE;
- residual/flow horizon: 12 hours;
- residual z tails: winner >= 1.5 and loser <= -1.5;
- price/flow disagreement on both tails >= 1.0 z;
- long winner, short loser, factor-beta-neutral gross 1.0;
- entry: completed-hour signal + 5m open; fixed 12h exit;
- 6 bp/notional/side base cost, 10 bp stress, exact funding;
- full-calendar CAGR and global favorable-before-adverse strict MDD.

No second 2025 policy is available, so no hold/threshold/sign ranking can occur.

## Research evidence, not confirmation

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | +38.292% | +38.323% | 22.934% | 1.671 | 107 |
| 2024 | +110.659% | +110.338% | 11.773% | 9.372 | 117 |
| 2023–2024 | +191.325% | +70.620% | 22.934% | 3.079 | 224 |
| 2023–2024, 10 bp | +143.589% | +56.026% | 24.663% | 2.272 | 224 |

The 20,000-draw weekly-cluster sign-flip p-value was 0.00210. These numbers
only justify spending untouched 2025; they do not count as OOS evidence.

## 2025 pass contract

Calendar 2025 must have positive absolute return, CAGR/strict-MDD >= 3.0,
strict MDD <= 15%, positive H1 and H2, >= 60 trades, positive 10-bp stress,
weekly-cluster sign-flip p <= 0.10, and positive +5m entry-delay performance.
Daily mark-to-market correlation to BTC must be <= 0.30 in absolute value and
absolute BTC beta <= 0.15 for portfolio promotion.

Protocol hash: `151f7905b64a2eca471f56edf377a7b141f9ad8cb58fb7646c1f0b96a4a344ee`
