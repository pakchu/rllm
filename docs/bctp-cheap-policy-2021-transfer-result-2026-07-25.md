# BCTP-12H cheap-policy 2021 transfer result

## Decision

`BCTP-12H` cheap-policy continuation is **retired unchanged**. No primary passed
the frozen 2021 transfer gate, so no 2021 reward refit, 2022 schedule, 2022
market/funding access, or conditional Gemma training is authorized.

## Chronology

- Runner revision: `48770a692776a6ee3c47f8c061b20df9edc72d50`
- 2020 fit report:
  `fcd9629795fed7913b5d7dff683227ebf759d983b61200b2ed94a13912381574`
- Pre-outcome 2021 schedule seal:
  `602037067a87b9b2afde028dc601d1d996e406ec43327c7ad043af34c527590e`
- 2021 transfer report:
  `e2c62b724c405761125dd6c10702890651e0c4bd559f158bd65af15ea23d06ef`
- Familywise test: 31 frozen policies, 53 Monday-UTC clusters,
  100,000 shared Rademacher draws
- 2022 outcome rows opened: `0`

## Primary results

All values include the full 2021 calendar, 6 bp changed-notional costs, realized
funding, strict held-path MDD, and terminal flattening.

| Primary | Absolute return | CAGR | Strict MDD | CAGR/MDD | 10 bp stress return | +5m delay return | Non-flat intervals | Long / short share | pmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Categorical linear FQI | -32.81% | -32.83% | 45.23% | -0.73 | -38.72% | -32.04% | 469 | 60.55% / 39.45% | 1.000 |
| Categorical ridge FQI | -40.20% | -40.22% | 49.63% | -0.81 | -44.09% | -40.88% | 328 | 75.30% / 24.70% | 1.000 |
| ExtraTrees FQI | -4.02% | -4.02% | 18.56% | -0.22 | -6.87% | -2.46% | 91 | 70.33% / 29.67% | 1.000 |

Every primary failed positive base/stress/delay return, minimum CAGR/MDD,
required-control defeat, and familywise significance. Activity and direction
coverage alone were insufficient.

## Diagnostics

- `always_long` returned `+17.51%`, but strict MDD was `33.37%`, CAGR/MDD was
  only `0.53`, it had no short exposure, and familywise `pmax` was approximately
  `0.999`.
- The categorical ridge direction-flip control returned `+12.58%`, but strict
  MDD was `26.40%`, CAGR/MDD was `0.48`, and `pmax` was approximately `0.999`.
- ExtraTrees was the least damaging primary, but it still lost money and failed
  to beat exact memory and its shuffled-reward control.

The outcome is therefore not a near miss. Block-clearing topology, as encoded by
this frozen three-snapshot 12-hour MDP and reward, did not transfer into a
tradable 2021 alpha.

## Stop rule

Do not tune BCTP thresholds, rewards, features, model parameters, or controls
using this 2021 result. Do not open 2022 for repair. Future alpha work must start
as a separately named, source-justified, preregistered candidate.
