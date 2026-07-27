# PSIM-D8-RLLM2-S6 runner implementation failure

Date: 2026-07-27 KST

Status: **SEALED CODE FAILURE — no S6 schedule or 2021 economic result**

## Failure

The committed S6 runner wrote its preregistered attempt and then stopped with:

```text
AttributeError: module 'training.psim_action_mean_residual_fqi'
has no attribute 'reconstruct_reward_tensor'
```

The S6 core deliberately kept the exact S4 reward reconstruction helper under
`s5_core`, but the runner called it as a direct S6 module attribute. This is a
single symbol-resolution defect, not a model, reward, feature, gate, or outcome
failure.

The failed execution is immutable:

- execution commit:
  `31cd9ba330a7f3c53b7a5a642d365e729d1e7cca`;
- runner SHA-256:
  `0a50b7abbdd1dea454f080f05afb36320235ce53afcbfda07e721cc02b35dadd`;
- attempt SHA-256:
  `23328b8c3ec233356700dea4618f66c1765d81b1ee5a3136aa9dfc5f9a54157e`;
- attempt self-hash:
  `0b686a89dc796800422b218888fd904a24ddbfa6c7ca2e350662621085e7c45d`;
- failure-record SHA-256:
  `b0c45438d29126cfac65b7d7d8ed9318d595b218ffd2062bcf64d13c521237ad`;
- failure self-hash:
  `9c95f0d80c1b44e4116edc43d9973b4cb240b9b3a1670226eca71043b47a3249`.

The original attempt is retained and may not be deleted or overwritten.

## Access boundary

The exception happened after the frozen 3,288-row 2020 transition ledger was
parsed, but before the original reward tensor was reconstructed.

- raw market/funding paths read: 0;
- frozen 2020 transition-ledger rows parsed: 3,288;
- reconstructed 2020 reward tensors: 0;
- residual reward values created: 0;
- 2020 economic metric sets computed: 0;
- 2021 market/funding paths read: 0;
- 2021 rewards or economic metrics computed: 0;
- S6 policy-specific 2021 outcomes opened: no;
- 2022-or-later outcomes opened: no;
- model loads/forwards: 0 / 0.

The frozen source-only bundle was loaded and 365 source rows were indexed for
future schedule construction. That bundle was already authorized by the S6
preregistration and contains no S6 policy reward or economic result.

No S6 residual ledger, base schedule, delayed schedule, schedule manifest, or
terminal gate result was written.

## Authorized repair

The failed runner cannot be silently patched and resumed because its
write-once attempt binds the old execution commit and runner hash.

A separate `S6R1` preregistration may:

1. preserve the exact S6 scientific hypothesis;
2. preserve every model, source feature, reward formula, hyperparameter,
   control, and readiness threshold;
3. change only the reconstruction call to
   `residual.s5_core.reconstruct_reward_tensor`;
4. use a new committed runner and distinct write-once output paths; and
5. run the outcome-blind schedule-readiness gate before any 2021 market,
   funding, reward, or economic access.

It may not tune the candidate from this implementation failure.

Terminal action:

```text
SEAL_S6_IMPLEMENTATION_FAILURE_AND_PREREGISTER_S6R1_CODE_ONLY_REPAIR
```
