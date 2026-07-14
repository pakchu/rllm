# CARTA v1 pre-Gemma selection result — 2026-07-14

## Verdict

**CARTA v1 fails the frozen pre-Gemma learnability gate.** Neither the
relational ridge policy nor categorical Naive Bayes produced positive 2023
returns. Gemma training is not allowed under this frozen experiment, and 2024,
2025, and 2026 remain unopened.

- support freeze: `1f8439b`
- evaluator freeze: `413e84a`
- result: `results/causal_adaptive_relational_baseline_selection_2026-07-14.json`
- result SHA256: `b17ef30fd97bc8054a49e42c84d406439c547b97fbd8fb94f0baf59625c55a75`
- bandit source SHA256: `7cb4428b39c923dc909fbd380cef6bb8647c47a5acef099d75c8d5c22d518b68`
- evaluator source SHA256: `130bc08767d6f4d71541215a66b4a88fdc160081e14849ab0000066bb7f3dc21`
- execution: fixed candidate clock, next 5-minute open, 72-bar
  scheduled-open exit, `0.5x`, `6 bp` notional cost per side
- CAGR: complete split clock including idle cash
- strict MDD: complete held path, favorable extreme before adverse extreme

## 2023 policy comparison

| policy | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long / short | weekly p |
|---|---:|---:|---:|---:|---:|---:|---:|
| always abstain | 0.00% | 0.00% | 0.00% | 0.00 | 0 | 0 / 0 | 1.0000 |
| always follow | -15.85% | -15.86% | 16.90% | -0.94 | 236 | 99 / 137 | 0.9999 |
| always fade | -10.65% | -10.65% | 12.54% | -0.85 | 236 | 137 / 99 | 0.9979 |
| exact-signature memory | 0.00% | 0.00% | 0.00% | 0.00 | 0 | 0 / 0 | 1.0000 |
| shuffled ridge | -0.03% | -0.03% | 1.37% | -0.02 | 15 | 4 / 11 | 0.5213 |
| **relational ridge** | **-0.74%** | **-0.75%** | **2.04%** | **-0.36** | **31** | **28 / 3** | **0.7094** |
| **Naive Bayes** | **-5.65%** | **-5.65%** | **6.48%** | **-0.87** | **89** | **56 / 33** | **0.9895** |

The exact-signature memory policy abstains on every 2023 candidate because all
236 selection signatures were unseen in training. Its 170.87% in-sample
absolute return is oracle memorization, not deployable performance.

## Relational ridge detail

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long / short |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022, in-sample | +17.59% | +5.55% | 2.85% | 1.95 | 24 | 19 / 5 |
| select 2023 | -0.74% | -0.75% | 2.04% | -0.36 | 31 | 28 / 3 |
| 2023 H1 | +0.75% | +1.52% | 0.88% | 1.74 | 14 | 13 / 1 |
| 2023 H2 | -1.49% | -2.93% | 2.04% | -1.43 | 17 | 15 / 2 |

The policy selected only three `FOLLOW` and 28 `FADE` actions in full 2023.
Because the reference side itself varies, that action mix became 28 long and
only three short positions. The frozen direction-collapse guard correctly
rejects this apparently low-MDD result.

## Naive Bayes detail

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long / short |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022, in-sample | +14.05% | +4.48% | 3.54% | 1.26 | 115 | 72 / 43 |
| select 2023 | -5.65% | -5.65% | 6.48% | -0.87 | 89 | 56 / 33 |
| 2023 H1 | -2.61% | -5.20% | 3.03% | -1.71 | 36 | 21 / 15 |
| 2023 H2 | -3.12% | -6.10% | 4.83% | -1.26 | 53 | 35 / 18 |

Naive Bayes trades both action and position directions, but loses in both 2023
halves. This is broad generalization failure rather than one missed threshold.

## Reward and update audit

The frozen full-information reward prefers abstention in most events:

| window | candidates | abstain | follow | fade | mean follow utility | mean fade utility |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | 323 | 175 | 72 | 76 | -0.00369 | -0.00365 |
| select 2023 | 236 | 180 | 25 | 31 | -0.00228 | -0.00203 |
| 2023 H1 | 91 | 64 | 12 | 15 | -0.00251 | -0.00230 |
| 2023 H2 | 144 | 116 | 12 | 16 | -0.00216 | -0.00184 |

Label availability is heavily front-loaded: 265 labels exit by 2022-01-01,
while the fixed 2022 quarterly releases contain only 8, 2, 8, and 40 labels.
Thus the final 323-row fit is dominated by 2020–2021 semantics even though the
experiment was motivated by a post-2021 structural change.

## Interpretation

CARTA fixed the exact-state memorization problem but not the economic
nonstationarity problem:

1. both unconditional directions lose after cost, so the event clock has no
   stable default action;
2. compositional main effects and ten preregistered relation interactions look
   profitable in-sample but reverse in H2 2023;
3. the 97.5th-percentile support stopping rule leaves too few 2022 updates to
   overwrite 2020-dominated relationships;
4. the ridge policy converts that stale mapping into a severe actual long-side
   bias, while NB remains active but loses across both halves;
5. target complexity is not the immediate bottleneck—there is no causal cheap
   baseline for Gemma to improve upon under this frozen state/reward contract.

These findings cannot be used to lower the 0.975 event threshold, remove the
drawdown penalty, rebalance years, change the 9/72-bar clock, or tune the ridge
floor and still call the result CARTA v1. Any adaptive successor must be newly
named and preregistered.

## Next research constraint

The next candidate should make **state change itself** observable at training
time rather than relying on a large old replay buffer. A valid successor may
use causally decayed or finite-memory learning, but it must freeze the decay,
minimum recent support, fallback behavior, and no-trade rule before outcomes.
It must also retain a bidirectional execution guard so `FOLLOW/FADE` diversity
cannot hide an all-long or all-short portfolio.

No CARTA v1 model, Gemma adapter, or 2024+ evaluation should be produced.
