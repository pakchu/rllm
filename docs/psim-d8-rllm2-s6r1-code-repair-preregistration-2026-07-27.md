# PSIM-D8-RLLM2-S6R1 code-repair preregistration

Date: 2026-07-27 KST

Status: **PREREGISTERED — no S6R1 2021 outcome opened**

## Purpose

The original S6 attempt was sealed after a single symbol-resolution error:

```text
residual.reconstruct_reward_tensor
```

The reconstruction helper exists under the already imported, exact S5 core:

```text
residual.s5_core.reconstruct_reward_tensor
```

S6R1 authorizes only that code correction. It is not a new alpha hypothesis
and cannot alter the model, features, reward transform, fit period, FQI
parameters, policy family, controls, schedule-readiness thresholds, or future
transfer gates.

## Immutable scientific contract

S6R1 copies these complete S6 sections without modification:

1. frozen source and outcome artifacts;
2. all-action-mean residual reward contract;
3. Ridge FQI contract;
4. policy and control family;
5. pre-2021 schedule-readiness gate; and
6. future report-only 2021 transfer gate.

Their combined canonical hash is:

```text
2471fa5e5272ed33e970ba03c0b8268df4eaeb6a856eaa28e7fa8125b1e0b70d
```

The primary remains:

```text
semantic_ridge_action_mean_residual_fqi
```

The unchanged readiness requirements include:

- 365 base and 365 exactly delayed primary rows;
- at least 80 non-flat targets;
- at least 20% long and 20% short among non-flat targets;
- exact action-code permutation identity;
- valid action labels;
- exact `sequence_id + ":delay_5m"`, timestamp `+5m`, and target identity;
- positive Hamming distance from every fixed degenerate control; and
- positive Hamming distance from the sealed S5 primary.

No quota, Q-margin, threshold search, clipping, scaling, hyperparameter search,
ExtraTrees, QLoRA, model load, or new Gemma forward is authorized.

## Failure evidence

- failed S6 commit:
  `31cd9ba330a7f3c53b7a5a642d365e729d1e7cca`;
- failed runner SHA-256:
  `0a50b7abbdd1dea454f080f05afb36320235ce53afcbfda07e721cc02b35dadd`;
- failed attempt SHA-256:
  `23328b8c3ec233356700dea4618f66c1765d81b1ee5a3136aa9dfc5f9a54157e`;
- failed attempt self-hash:
  `0b686a89dc796800422b218888fd904a24ddbfa6c7ca2e350662621085e7c45d`;
- failure-record SHA-256:
  `b0c45438d29126cfac65b7d7d8ed9318d595b218ffd2062bcf64d13c521237ad`;
- failure self-hash:
  `9c95f0d80c1b44e4116edc43d9973b4cb240b9b3a1670226eca71043b47a3249`.

The old attempt remains immutable. S6R1 uses a new runner and six distinct
write-once output paths.

## Execution order

1. validate the exact failed S6 attempt and failure record;
2. implement the one-line namespace repair;
3. test the complete execution path with synthetic fixtures before commit;
4. independently review the runner;
5. commit and push the runner;
6. write the S6R1 attempt before parsing the frozen 2020 ledger;
7. reconstruct and residualize the original S4 2020 rewards;
8. fit the same seven Ridge estimators and eight-policy family;
9. seal source-only 2021 schedules without market or reward access; and
10. reject or authorize a separately preregistered report-only transfer solely
    from the unchanged outcome-blind readiness gate.

## Access boundary

At preregistration:

- raw market/funding paths read: 0;
- S6R1 2020 ledger rows parsed: 0;
- 2021 market/funding paths read: 0;
- 2021 rewards/economic metrics computed: 0;
- S6R1 policy-specific 2021 outcomes opened: no;
- 2022-or-later outcomes opened: no;
- model loads/forwards: 0 / 0.

Repository-wide 2021 is not globally pristine because unrelated historical
results exist. A successful S6R1 schedule gate can authorize only a
protocol-isolated, policy-specific, report-only 2021 transfer.

## Immutable preregistration

- file SHA-256:
  `a12db8dd9899a9b0e81f2e0ff80f1acaf7695019717cdfb4d14c7755d06576dc`;
- manifest hash:
  `f527487ec53eefe7ee0ddcc931512e8f845c510d3566f4a17104082454c88aff`.

Next authorized action:

```text
IMPLEMENT_FULL_EXECUTE_SMOKE_REVIEW_COMMIT_AND_PUSH_S6R1_RUNNER_THEN_EXECUTE_OUTCOME_BLIND_PRE2021_SCHEDULE_GATE_ONLY
```
