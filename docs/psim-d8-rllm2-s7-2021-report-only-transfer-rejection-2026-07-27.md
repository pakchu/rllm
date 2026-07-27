# PSIM-D8-RLLM2-S7 2021 report-only transfer rejection

Date: 2026-07-27 KST

Status: **TERMINAL REPORT-ONLY REJECTION — no 2021 repair or selection**

## Fixed primary result

The preregistered primary
`semantic_ridge_action_mean_residual_fqi` failed the sealed 2021 transfer:

| evaluation | absolute return | CAGR | strict MDD | CAGR / strict MDD | directional entries | all target changes |
|---|---:|---:|---:|---:|---:|---:|
| base, 6 bp | -0.92% | -0.92% | 36.21% | -0.03 | 153 | 163 |
| stress, 10 bp | -6.96% | -6.96% | 37.88% | -0.18 | 153 | 163 |
| delayed +5m, 6 bp | -3.58% | -3.58% | 36.71% | -0.10 | 153 | 163 |
| first half, 6 bp | +23.05% | +51.97% | 23.84% | 2.18 | 80 | 84 |
| second half, 6 bp | -17.88% | -32.37% | 36.21% | -0.89 | 74 | 81 |

The half-year checks are standalone robustness simulations. Each half resets
to flat with equity 1; they are not continuous full-path subperiod
attribution.

The primary weekly one-sided statistic had:

- local randomization p-value: `0.5102148978510215`;
- 41-policy familywise max-stat p-value: `1.0`;
- weekly clusters: `53`;
- Monte Carlo draws: `100,000`, shared seed `20260725`.

The strongest fixed nonsemantic comparator was `ethereum_only`, with +2.53%
absolute return, +2.54% CAGR, 2.11% strict MDD, and a 1.20 ratio. The primary
beat neither its return nor its ratio.

## Gate outcome

Passed:

- first-half absolute return was positive;
- 356 non-flat intervals exceeded the minimum 80;
- long and short shares were 39.61% and 60.39%;
- the action-code permutation schedule remained exactly identical.

Failed:

- base absolute return;
- 10 bp stress absolute return;
- delayed absolute return;
- second-half absolute return;
- minimum base CAGR / strict-MDD ratio;
- strongest nonsemantic comparator absolute return;
- strongest nonsemantic comparator ratio;
- familywise max-stat threshold.

The fixed all-checks-required decision is therefore `reject`.

## Scientific interpretation

S6R1 solved the outcome-blind action-prior and direction-balance defects, but
the resulting semantic Ridge policy did not generalize economically through
the complete 2021 calendar. Its positive standalone first half was more than
offset by the second half, and the result was not robust to either extra cost
or a five-minute delay.

Several other frozen family members had positive report-only values. Those
are not promoted or selected: choosing one after viewing this 2021 family
would be post-outcome model selection. In particular, this result does not
authorize an ExtraTrees substitution, a half-year gate, a new action
residual, or any threshold fitted to 2021.

## Execution and access boundary

- execution commit:
  `538d7f590c16859823c44451b34e1f71e2253bd8`;
- runner SHA-256:
  `51aa648b15394dc1d70c0cbfa82d885672ecdddf6eac126c0b49f24dc6bfa114`;
- evaluator-core SHA-256:
  `bda2354dbd846fb72990f8813630c1fbfdf6ce2d32ed26d66dbf8dbe2d08792b`;
- wall time: 4 minutes 12.63 seconds;
- maximum RSS: 363,716 KiB;
- market rows parsed: 105,120;
- funding rows parsed: 1,095;
- fixed policy schedules: 41;
- economic metric sets: 45;
- 2022-or-later outcomes opened: no;
- model loads / forwards: 0 / 0;
- selection or repair from 2021: no.

The attempt was written before any market or funding payload was opened,
read, hashed, or parsed. The runner exited with status 0. A surrounding zsh
logging wrapper later used the reserved variable name `status`; that wrapper
error occurred only after the complete result had been printed and does not
affect the runner artifact.

## Immutable evidence

- attempt file SHA-256:
  `922de8d8da63cc45c9536409177af80809d9d8b17bf60e9024e68691bedd8c0f`;
- attempt self-hash:
  `5ad95e0af8cb202e6dbe9f1ba2c3c5d8efbf509327aa0d88d90ae2e89cd0ada8`;
- result file SHA-256:
  `c061b82438a5b207801b321b864a252564fcd754f3ce09a4ff4d427c3327480a`;
- result self-hash:
  `545b58bd5346d6fa5c87195e06692432a0b0447cbc31a56a917830b62da59e71`.

Terminal action:

```text
RETIRE_UNCHANGED_S6R1_HYPOTHESIS_NO_2021_REPAIR
```

## Next permitted research

The S6R1 hypothesis is retired unchanged. A new alpha family must be specified
independently, selected without using these 2021 outcomes, and evaluated on a
separately isolated holdout or forward stream. This 2021 result may be used
only as a terminal failure record, not as a tuning surface.
