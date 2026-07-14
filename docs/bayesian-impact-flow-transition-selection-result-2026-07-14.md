# BIFT v1 pre-2024 selection result — 2026-07-14

## Verdict

**Rejected.** BIFT v1 failed the frozen train and 2023 selection gate. The
signal, threshold, branches, confirmation, and hold remain immutable; 2024,
2025, and 2026 outcomes were not opened.

- preregistration commit: `a1b9b0b`
- evaluator commit: `d5a1919`
- result artifact:
  `results/bayesian_impact_flow_transition_selection_2026-07-14.json`
- result SHA256:
  `1f3085d31bb612ba10755cd6b7c09e066748f8206c5e6a36dce219adffb38256`

## Frozen BIFT performance

All values use 0.5x leverage, 5 bp fee plus 1 bp slippage per notional side,
next-open entry, scheduled-open exit, full-clock CAGR, and favorable-first
held-path strict MDD.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | **-24.67%** | -9.01% | **32.71%** | -0.28 | 186 | 0.86745 |
| select 2023 | +2.77% | +2.77% | 9.99% | 0.28 | 86 | 0.35197 |
| 2023 H1 | +7.98% | +16.76% | 4.79% | 3.50 | 51 | 0.13563 |
| 2023 H2 | **-4.77%** | -9.24% | 9.99% | -0.93 | 34 | 0.91244 |

The attractive 2023 H1 ratio is not a qualification result: it misses the
weekly significance gate, reverses in H2, and is preceded by a deeply negative
three-year train result.

## Frozen controls

### Train 2020–2022

| policy | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| BIFT branch mapping | -24.67% | -9.01% | 32.71% | -0.28 | 186 | 0.86745 |
| always follow flow | +10.20% | +3.29% | 17.80% | 0.18 | 186 | 0.29196 |
| always fade flow | -30.78% | -11.54% | 33.33% | -0.35 | 186 | 0.94208 |
| propagation only | -5.34% | -1.81% | 19.72% | -0.09 | 133 | 0.57426 |
| absorption only | -20.42% | -7.33% | 25.89% | -0.28 | 53 | 0.97181 |
| permuted branch | -27.81% | -10.29% | 35.56% | -0.29 | 186 | 0.91705 |

### Select 2023

| policy | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| BIFT branch mapping | +2.77% | +2.77% | 9.99% | 0.28 | 86 | 0.35197 |
| always follow flow | +3.28% | +3.28% | 7.15% | 0.46 | 86 | 0.32895 |
| always fade flow | -13.20% | -13.20% | 18.05% | -0.73 | 86 | 0.95943 |
| propagation only | +4.90% | +4.90% | 7.77% | 0.63 | 57 | 0.24961 |
| absorption only | -2.03% | -2.03% | 5.29% | -0.38 | 29 | 0.70389 |
| permuted branch | +0.06% | +0.06% | 9.58% | 0.01 | 86 | 0.48165 |

Controls were applied only after the 272-event non-overlap candidate clock was
reserved, so abstaining controls did not release later opportunities.

## Frozen-gate failures

The evaluator recorded all failures rather than selecting a favorable subset:

1. train absolute return was non-positive;
2. train CAGR/strict-MDD was below 3;
3. train strict MDD exceeded 15%;
4. train weekly-cluster p-value was not below 0.10;
5. 2023 CAGR/strict-MDD was below 3;
6. 2023 weekly-cluster p-value was not below 0.10;
7. 2023 H2 return was non-positive;
8. BIFT did not beat always-follow flow on the frozen minimum
   train/selection ratio.

## Interpretation

The failure is structural, not a near-threshold miss.

- Persistent public flow has a weak positive gross tendency in train, as the
  always-follow control indicates, but it does not clear the strict risk or
  significance target after costs.
- The proposed absorption inference is harmful in both train and 2023.
- Propagation is also not invariant: negative in train, positive only in 2023
  H1, and negative again in 2023 H2.
- BOCPD found unusual states, but unusualness did not make the branch semantics
  temporally stable.

Therefore Gemma or RL training over the same BIFT state is not justified. A
larger model would learn a non-invariant mapping and risk reproducing the same
era dependence. The next experiment must introduce a different causal
observable or decision mechanism, not tune BIFT's percentile, branch, or hold.

## Seal state

- opened: train 2020–2022, select 2023, 2023 H1/H2
- still sealed: test2024, eval2025, ytd2026
- post-result repair allowed: **no**
- BIFT promotion or portfolio inclusion: **no**
