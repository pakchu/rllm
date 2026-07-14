# CFCF v1 pre-2024 selection result — 2026-07-14

## Decision

**CFCF v1 is rejected.** It failed the frozen train/2023 return gate and may
not be repaired by changing its quantile, direction, hold, branch, venue, or
execution rule under the same experiment name.

- preregistration commit: `17c6631`
- evaluator freeze commit: `ee08560`
- evaluator SHA256:
  `bf36ef4ad9e7416dea2146669a0b4aa4cbd155934ecc5789194b2d3c17c61bac`
- result artifact:
  `results/cross_venue_funding_consensus_fracture_selection_2026-07-14.json`
- result SHA256:
  `eac618ab2a15d539f558ee35e203c38b0c172bdcf024d8f9371153a3dcd82b02`
- opened outcomes: 2021-2023 only
- still sealed: full 2024, full 2025, and 2026 YTD

Before the one-shot run, the working evaluator hash was verified byte-for-byte
against Git commit `ee08560`. The loader then verified the frozen
preregistration, document, support artifact, scheduler, execution engine, three
source manifests, underlying source-file hashes, exact time grids, and the
223-candidate support replay.

## Frozen CFCF statistics

Every CAGR uses the complete split clock including idle cash. Strict MDD uses
the held path with the favorable extreme placed before the adverse extreme.
Execution is at 0.5x leverage with 5 bp fee plus 1 bp slippage per notional
side.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long/short | weekly p |
|---|---:|---:|---:|---:|---:|---:|---:|
| train 2021-2022 | +5.40% | +2.67% | 11.88% | 0.22 | 134 | 85 / 49 | 0.3146 |
| select 2023 | -4.82% | -4.82% | 13.62% | -0.35 | 89 | 35 / 54 | 0.7382 |
| 2023 H1 | -1.47% | -2.95% | 5.85% | -0.50 | 38 | 18 / 20 | 0.5963 |
| 2023 H2 | -3.40% | -6.64% | 9.75% | -0.68 | 51 | 17 / 34 | 0.7181 |

The frozen gate failed because:

1. train CAGR/MDD was below 3 and its weekly-cluster p-value was not below
   0.10;
2. full 2023 return was negative, CAGR/MDD was below 3, and p was not below
   0.10;
3. both 2023 halves were negative;
4. CFCF's minimum train/select ratio did not beat always-long.

The trade-count and strict-MDD ceilings passed, but they do not compensate for
the failed profitability, stability, and significance gates.

## Frozen controls

All controls used the CFCF non-overlap opportunity clock reserved before an
action or abstention. An omitted branch never released a later candidate.

| policy | train abs | train CAGR/MDD | train trades | 2023 abs | 2023 CAGR/MDD | 2023 trades |
|---|---:|---:|---:|---:|---:|---:|
| **CFCF convergence** | **+5.40%** | **0.22** | **134** | **-4.82%** | **-0.35** | **89** |
| exact reverse | -20.48% | -0.44 | 134 | -5.99% | -0.54 | 89 |
| always long | +13.68% | 0.84 | 134 | -2.82% | -0.29 | 89 |
| always short | -26.27% | -0.44 | 134 | -7.93% | -0.58 | 89 |
| Bybit-rich only | -6.94% | -0.23 | 49 | -4.31% | -0.38 | 54 |
| Bybit-cheap only | +13.25% | 1.15 | 85 | -0.53% | -0.10 | 35 |
| permuted branch | -5.48% | -0.18 | 134 | -15.93% | -0.91 | 89 |

## Interpretation

The support clock was real, balanced, causal, and executable, but support did
not imply alpha.

- The train gain came entirely from the Bybit-cheap/long branch. The
  Bybit-rich/short branch lost 6.94%.
- Always-long beat CFCF in train, so much of the apparent gain was ordinary BTC
  long exposure rather than information in the cross-venue fracture mapping.
- In 2023 both branches lost money, and both the frozen direction and its exact
  reverse were negative. This is stronger evidence against a stable directional
  mapping than a simple sign mistake.
- The permutation control was worse, so the labels were not completely
  structureless, but the observed structure was too weak and nonstationary to
  satisfy the economic gate.

The defensible conclusion is that contemporaneous Binance/Bybit premium and
realized-funding disagreement does not provide a sufficiently stable
next-funding-boundary directional BTC edge in this form. A successor must use a
different causal mechanism rather than mining this failed clock.

## RLLM decision

No Gemma fine-tuning or RL layer will be attached to CFCF. An LLM could learn
the train-period long bias or memorize venue-state regimes, but it cannot turn
this failed deterministic base into clean evidence. RLLM work remains gated on
an independently validated causal alpha.
