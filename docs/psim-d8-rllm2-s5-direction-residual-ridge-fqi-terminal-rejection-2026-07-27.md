# PSIM-D8-RLLM2-S5 direction-residual Ridge FQI terminal rejection

Date: 2026-07-27 KST

Status: **TERMINAL PRE-OUTCOME REJECTION — no S5 2021 economics opened**

## Result

S5 successfully removed the S4 long-direction collapse but failed the
preregistered minimum-activity gate:

| schedule diagnostic | value | gate | result |
|---|---:|---:|---|
| non-flat target rows | 74 | at least 80 | fail |
| long targets | 44 | — | — |
| short targets | 30 | — | — |
| flat targets | 291 | — | — |
| long share among non-flat | 59.46% | at least 20% | pass |
| short share among non-flat | 40.54% | at least 20% | pass |
| action-code permutation mismatches | 0 | exactly 0 | pass |
| degenerate-control Hamming checks | 7 / 7 positive | all positive | pass |
| delayed target counts equal base | yes | required | pass |

Only `minimum_nonflat_target_rows` failed. The threshold is not reduced from
80 to 74 and the result is not promoted as a near-pass.

There is no S5 2021 absolute return, CAGR, strict MDD, trade count, stress
return, delayed return, half-year return, or p-value. Those metrics were never
computed.

## What worked

The fixed per-position direction residual:

\[
\delta_p =
\frac{1}{2}
\operatorname{mean}_t
(R_{t,p,\text{long}} - R_{t,p,\text{short}})
\]

produced:

| current position | \(\delta_p\) |
|---|---:|
| short | 0.0013514627359444101 |
| flat | 0.0022275669631357190 |
| long | 0.0031416503931195293 |

After applying `long -= delta` and `short += delta`, the mean residual long
and short reward is equal within each current position to numeric tolerance.

Compared with S4:

- S4 Ridge: 99 long / 14 short, short share 12.39%;
- S5 Ridge: 44 long / 30 short, short share 40.54%.

The neutral action-code permutation is also exact for all 365 target rows.
Thus the direction correction and deterministic Ridge construction solved the
two S4 structural defects they were intended to solve.

## Why S5 still fails

S5 neutralized only the unconditional **long-vs-short** reward difference.
It deliberately left the unconditional flat-action baseline unchanged.
The semantic Ridge policy therefore became directionally balanced but still
abstained on 291 of 365 days.

This is an outcome-blind schedule diagnosis, not a profitability diagnosis.
It supports one separate next hypothesis: remove the unconditional mean of
all three actions per current position so that the model must learn
state-conditional deviations rather than a flat, long, or short prior.

It does not authorize:

- lowering the activity gate;
- adding a target quota;
- tuning a Q-margin on the 74 observed non-flat rows;
- opening S5 2021 economics;
- reusing S5 under another name; or
- selecting from known unrelated 2021 metrics.

## Execution and access boundary

Official execution:

- commit:
  `c814807c7a25b0a9ec28c2bd7db4bbd5ac6ac520`;
- runner SHA-256:
  `320382ccea359704a5fbe3cc2599c981dee456f26fd66c4f5a98c67274e179f2`;
- wall time: 4.50 seconds;
- maximum RSS: 253,924 KiB;
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
- 2021 market rows: 0;
- 2021 funding rows: 0;
- 2021 rewards: 0;
- 2021 economic metric sets: 0;
- model loads/forwards: 0 / 0.

The repository still contains unrelated historical 2021 transfer results, so
2021 is not globally pristine. S5 created no policy-specific 2021 metric.

## Immutable evidence

- attempt SHA-256:
  `ab59c4fe359fe4a8a3b05c7a128999489ba2679bf453e79c1bbb9428baaa0d42`;
- attempt hash:
  `f0c4dd9ac216ba9458a4541cc01af2290ff51d1bc4caa54d2b873e67d84a7efc`;
- terminal result SHA-256:
  `aabc479e030a3a08ddd07047ccb25ce872425655ca10d877b8e78c4687bf73a7`;
- result hash:
  `2331b46e3847d80200c88b3f3522ef35f3f710dead5ea6dca85575f249b0541e`;
- schedule manifest SHA-256:
  `76f66a4353d0d8e3bb8f2aa1671f6a5416dd7f344c22bdeed63a485f984a1d60`;
- schedule manifest hash:
  `ca0ef82f5e9886288942c2ff8b115fa9e64eeee143b358ebd90ceb32b7fb5022`;
- residual ledger SHA-256:
  `b395aa4cb0b0288eb7287eea297bc3b1b416e1326b379a48952a6fb8489fc19b`;
- base schedules SHA-256:
  `ed5ed208104242f571f4a07a53a815a1db2742957f43e9fcbecd9c18221a3833`;
- delayed schedule SHA-256:
  `e0580cf4bb1a0a3a05710945abfe20edb9c05bb47a30c34aafdb43556d85e8b5`.

Terminal action:

```text
TERMINAL_REJECT_S5_WITHOUT_2021_MARKET_FUNDING_REWARD_OR_METRIC_ACCESS
```

## Next permitted hypothesis

A separately preregistered S6 may use the same frozen Gemma representation,
PCA, transition ledger, Ridge parameters, controls, and readiness gate while
changing only the reward baseline:

\[
\mu_{p,a} = \operatorname{mean}_t R_{t,p,a}, \qquad
\bar{\mu}_p = \frac{1}{3}\sum_a \mu_{p,a}
\]

\[
R^*_{t,p,a} = R_{t,p,a} - \mu_{p,a} + \bar{\mu}_p
\]

This equalizes the unconditional means of flat, short, and long within each
current position. It is not a target quota and must again pass the exact same
80-row, 20%/20%, invariance, and degenerate-control gates before any
policy-specific 2021 economic evaluation.
