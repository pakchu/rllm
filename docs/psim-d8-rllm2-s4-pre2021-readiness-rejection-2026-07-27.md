# PSIM-D8-RLLM2-S4 pre-2021 readiness rejection

Date: 2026-07-27 KST

Status: **PRE-OOS NO-GO — S4 did not open 2021; global 2021 history is not pristine**

## Decision

S4 completed its declared job: it fit the complete policy family on 2020,
sealed 25 base schedules and two delayed primary schedules for 2021, and
opened no 2021 market or funding payload. A stricter outcome-blind readiness
audit then rejected both primaries **before** exercising the S4 authorization
to evaluate 2021.

There is therefore no 2021 return, CAGR, strict MDD, trade count, p-value, or
alpha claim for either S4 primary.

The rejection is structural:

| primary | 2021 flat | 2021 long | 2021 short | non-flat | long share | short share | neutral action-code mismatches | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `semantic_ridge_fqi` | 252 | 99 | 14 | 113 | 87.61% | 12.39% | 0 | reject: direction balance |
| `semantic_extra_trees_fqi` | 325 | 40 | 0 | 40 | 100.00% | 0.00% | 2 | reject: activity, direction, invariance |

The readiness contract was already fixed in the S4 preregistration before S4
opened any market outcome. It requires:

- at least 80 non-flat target rows;
- at least 20% long and 20% short among non-flat targets;
- exact target-sequence identity under a semantics-preserving action-code
  permutation; and
- at least one primary passing all three checks before any OOS payload opens.

No primary is eligible.

## Why the attractive 2020 numbers are not alpha

The following values are **in-sample 2020 training diagnostics only**:

| primary | absolute return | CAGR | strict MDD | CAGR / strict MDD | trades | non-flat intervals | long share | short share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `semantic_ridge_fqi` | +78.67% | 78.46% | 5.26% | 14.93 | 108 | 87 | 82.76% | 17.24% |
| `semantic_extra_trees_fqi` | +125.72% | 125.34% | 4.69% | 26.72 | 82 | 55 | 98.18% | 1.82% |

These diagnostics show exactly why an OOS readiness check is necessary.
Training on the strongly bullish 2020 period allowed both estimators to turn
semantic state into a mostly-long policy. High in-sample CAGR/MDD under that
directional exposure is not evidence that the LLM representation learned a
portable long/short alpha.

The 10 bp cost-stress and one-bar-delay 2020 diagnostics were also positive,
but they do not repair this regime-identification problem and remain
in-sample:

| primary | diagnostic | absolute return | CAGR | strict MDD | CAGR / strict MDD | trades |
|---|---|---:|---:|---:|---:|---:|
| Ridge | 10 bp stress | +74.48% | 74.28% | 5.34% | 13.91 | 108 |
| Ridge | 5m delay | +76.36% | 76.15% | 5.39% | 14.12 | 108 |
| ExtraTrees | 10 bp stress | +122.02% | 121.65% | 4.71% | 25.83 | 82 |
| ExtraTrees | 5m delay | +122.72% | 122.36% | 4.68% | 26.12 | 82 |

## Neutral-control defect

`semantic_ridge_fqi` is exactly invariant to its action-code permutation.
`semantic_extra_trees_fqi` is not. Its independently fitted permuted-code
control changes `TARGET_FLAT` to `TARGET_LONG` on:

- 2021-11-30 12:05 UTC; and
- 2021-12-01 12:05 UTC.

Only 2 of 365 targets differ, but a neutral encoding change is not an economic
hypothesis. Accepting those differences would make the result depend on
arbitrary label encoding. The successor must make this control byte-exact for
every estimator before any OOS evaluation.

## Outcome boundary and global-contamination disclosure

The audit read only the already sealed source-derived schedules, tracked
manifests, and prior-result documentation. It did not read a 2021 market or
funding payload:

- 2021 market paths: **0**;
- 2021 funding paths: **0**;
- 2021 market rows: **0**;
- 2021 funding rows: **0**;
- 2021 rewards: **0**; and
- 2021 economic metric sets: **0**.

The exact schedule inventory is:

- base schedules: 9,125 rows = 25 policies × 365 days;
- delayed schedules: 730 rows = 2 primaries × 365 days;
- base decision clock: 12:05 UTC;
- delayed decision clock: 12:10 UTC;
- duplicate `(policy_id, sequence_id)` rows: 0; and
- target domain: `FLAT / LONG / SHORT`.

This boundary is **protocol-local, not global**. The same repository already
contains a completed, unrelated BCTP 2021 transfer report and a tracked 2021
stage-source manifest:

- prior report:
  `docs/bctp-cheap-policy-2021-transfer-result-2026-07-25.md`;
- report SHA-256:
  `bd72c0839035860f289aec3ffcfe6e55721dd8a15bd65c93c3bb62475fa95d95`;
- prior 2021 stage-source manifest:
  `data/bctp_stage_sources/2021/source_manifest.json`;
- manifest SHA-256:
  `1d12d8dad47eda810933ddce7ac2d911a7a4f85262ecc02e91e22df2488c6e2d`;
- prior stage inventory: 105,120 5-minute market rows and 1,095 funding
  rows.

Accordingly, no later document may call 2021 a globally pristine historical
OOS period. A successor may use it only as a protocol-isolated,
policy-specific report-only transfer with this contamination disclosed and
must not use known 2021 metrics for design or selection. Truly pristine
evidence now requires forward or live data.

## Immutable evidence

S4 execution:

- execution commit:
  `e078f7c81693e30354b05ab6190c174235a3f548`;
- executed runner SHA-256:
  `f1c25b37c0c30b6bc9e81d2cbd32a7e70f5c39ba0404eb5ff4d9db977dcbe1e4`;
- attempt SHA-256:
  `717a140cec4a19ef057b2e44f7a508d25e977fd98b4447c3ea6f5dadd7e1e1fd`;
- attempt hash:
  `3f314bc9633351934b7b0c7ccc198f6ac48709e1ae7cf99ff4969900aaa553bc`;
- terminal-result SHA-256:
  `ca7a9a42eda7719a7e59b9927bf9c8e754689007468ec235ac4c5e2a1c619c75`;
- terminal result hash:
  `4253626d04e70f51dbd73df6c898c4a553b8df76b1ceb9c903c82e83ab2e3c09`;
- schedule-manifest SHA-256:
  `cd5d06f43d98b36e4d69b5630afe5e98f8d807aa05707e571aa49ed7c6ff4f6c`;
- schedule manifest hash:
  `e90ce8de7c82aa5ca22365f6c50a42bd975e994202b67df751a12ef804de6f85`.

Numeric and schedule artifacts:

- 2020 transition ledger:
  `07d465538d84648793ebbf302c54dace38ef71e88b22d7bd3a19fec500b99a7a`;
- 2020-only PCA32:
  `6a01d505ad2531683c9e8e9e0672456daf19a17a59b32d772fc857370210f0a2`;
- 2021 base schedules:
  `2f5b04e2514ca8b328fa6a92a45cea2828225090afc2427a58b6b778655394ba`;
- 2021 delayed schedules:
  `326d4e8c5e866b83b331d97ee54ea2abedce35abf85887539ee97220ed354484`.

Readiness rejection:

- audit artifact SHA-256:
  `e670c0965ba9fccccc06f472cff696e1392a2853d4d8016350191849d4548560`;
- audit result hash:
  `36e81a7a42aa3a06b95e01c5203ec8f363a11b6771bafd28614ce4f055611ba9`;
- terminal action:
  `REJECT_PSIM_D8_RLLM2_S4_BEFORE_2021_OUTCOME_OPEN_HOLD_S4_2021_EVALUATION_START_NEW_PREREGISTERED_ATTEMPT`.

## Next research step

S4 will not be rerun, repaired in place, or evaluated on 2021. A separately
preregistered successor may reuse the exact frozen Gemma-4-E4B-it source
representation and the already authorized 2020 transition ledger, but it must:

1. remove the 2020 bull-regime directional prior from the learned advantage
   without using any 2021 outcome;
2. use an estimator construction whose canonical and permuted action labels
   produce exactly identical schedules;
3. seal its 2021 schedules and pass this same activity/direction/invariance
   readiness gate; and only then
4. score 2021 once as a disclosed policy-specific transfer under the
   previously planned strict economics, stress/delay, half-year,
   strongest-control, and familywise max-stat gates.

Even a 2021 pass would authorize only further research. It would not restore
global OOS cleanliness or justify live capital without forward evidence.
