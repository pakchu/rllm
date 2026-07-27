# PSIM-D8-RLLM2-S6R1 schedule-readiness pass

Date: 2026-07-27 KST

Status: **OUTCOME-BLIND SCHEDULE PASS — 2021 economics still unopened**

## Result

The code-only S6R1 repair completed the unchanged S6 pipeline and passed every
preregistered schedule-readiness requirement:

| schedule diagnostic | value | gate | result |
|---|---:|---:|---|
| non-flat target rows | 356 | at least 80 | pass |
| long targets | 141 | — | — |
| short targets | 215 | — | — |
| flat targets | 9 | — | — |
| long share among non-flat | 39.61% | at least 20% | pass |
| short share among non-flat | 60.39% | at least 20% | pass |
| action-code permutation mismatches | 0 | exactly 0 | pass |
| S5-primary target Hamming distance | 282 | at least 1 | pass |
| degenerate-control Hamming checks | 7 / 7 positive | all positive | pass |
| delayed schedule identity | exact sequence, target, and +5m | required | pass |
| valid action labels | 3 / 3 only | required | pass |

This is not a profitability result. No S6R1 2021 absolute return, CAGR,
strict MDD, trade count, stress return, delayed return, half-year return, or
p-value has been computed yet.

## What changed from S5

S5 removed only the unconditional long-vs-short reward difference and
produced 74 non-flat rows. S6R1 uses the preregistered all-action residual:

\[
\mu_{p,a} = \operatorname{mean}_t R_{t,p,a}, \qquad
\bar{\mu}_p = \frac{1}{3}\sum_a \mu_{p,a}
\]

\[
R^*_{t,p,a} = R_{t,p,a} - \mu_{p,a} + \bar{\mu}_p
\]

This removes the unconditional flat/short/long prior within each current
position while retaining state-conditional reward deviations. The resulting
policy is active and directionally balanced without a quota or Q-margin.

The primary was fixed before execution:

```text
semantic_ridge_action_mean_residual_fqi
```

It was not selected from 2020 economic metrics or known 2021 outcomes.

## Execution and access boundary

- execution commit:
  `aee87ef7a543715b08a581f7c70a86547a39d135`;
- runner SHA-256:
  `c791054d32b5f4e1891916bf4bce9beb070d9fdddeb759318393374068595016`;
- wall time: 4.51 seconds;
- maximum RSS: 256,560 KiB;
- Ridge fits: 7;
- new schedules: 8 × 365 = 2,920 rows;
- delayed primary schedule: 365 rows.

Access:

- raw market/funding paths: 0;
- frozen 2020 transition-ledger rows parsed: 3,288;
- original 2020 reward values read: 3,288;
- residual reward values created: 3,288;
- 2020 economic metric sets computed: 0;
- 2021 market/funding paths: 0;
- 2021 market/funding rows: 0 / 0;
- 2021 rewards/economic metrics: 0 / 0;
- 2022-or-later outcomes: unopened;
- model loads/forwards: 0 / 0.

## Immutable evidence

- attempt SHA-256:
  `f4669c0a37878351bd89ad0d0554f02b6509a15b3ab80e4dbf7f2def9616cbaa`;
- attempt self-hash:
  `b50b2c4314603931e03c3ae37b92a3cc4d1ed426784440761dad9c17c731cdd6`;
- result SHA-256:
  `020c86002df8348c497407b70e24d6e85d583c3734242c6bddb0b5764b260d2f`;
- result self-hash:
  `63bec557d8e95a21becdad69c648f339d6f12510a28a679668a7bef3f4edd862`;
- schedule-manifest SHA-256:
  `816afbc7bca16df0313636194e4b0780bbe760cb7cd10f7944736c6968352644`;
- schedule-manifest self-hash:
  `314298356bbf3bc94e394ae362ee4f1894fd8f07e6e1a0b47137dd785f78970e`;
- residual-ledger SHA-256:
  `073565b87c67deb9754a4e6e99f1ba59e855ef8fe3937ed9a25cb3a9367a0273`;
- base-schedule SHA-256:
  `f850b3f9e18e9d942b8279065512fa33e77a9493abbd4d927672dc025ab41971`;
- delayed-schedule SHA-256:
  `b324347e72e08de8266dc39f7f800fae2a73a98129ab7092cb02b92893a02e28`.

Terminal action:

```text
SEAL_S6_2021_SCHEDULE_AUTHORIZE_SEPARATE_REPORT_ONLY_2021_TRANSFER_PREREGISTRATION
```

## Next authorized step

A separate report-only 2021 evaluator may now open the frozen market/funding
outcome exactly once. It must report the full preregistered family, fixed costs,
strict MDD, absolute return, full-calendar CAGR, trade count, delay/stress and
both-half checks, strongest nonsemantic control comparison, and weekly
familywise randomization result.

Because unrelated historical 2021 results exist in the repository, this can
only be described as a protocol-isolated policy-specific transfer, not a
globally pristine 2021 test. Success is still not live promotion; forward/live
confirmation remains mandatory.
