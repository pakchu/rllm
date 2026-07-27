# PSIM-D8-RLLM2-S1 source-feature seal terminal rejection

Date: 2026-07-27

Status: **TERMINAL REJECT — S1 may not be resumed, repaired, or rerun**

## Result

The official source-only S1 attempt ran from execution commit
`ff74a29de88acb04c9807586bd59b17dc4b3fc44` and failed during the frozen
Gemma embedding forward:

- completed source rows: **341 / 1,461**;
- model forwards started: **680**;
- embedding forwards started: **342**;
- relation-teacher forwards started: **338**;
- failure: `torch.OutOfMemoryError`;
- requested allocation at failure: **6.59 GiB**;
- market/funding rows parsed: **0 / 0**;
- rewards/economic metrics created: **0 / 0**; and
- 2020, test, and eval outcomes opened: **false**.

The failure result is canonical and sets both `resume_authorized=false` and
`rerun_authorized=false`.

## Exact evidence

- Attempt SHA-256:
  `d9ac5603db929c489e9a77a70aa2c2115e47d25ce318b9a4c35c7d56d434a4f8`
- Attempt hash:
  `cf2f9af9ff4589d18b13a845351eda399fcf4ea34f87fe8912101e9174dcb8f6`
- Failure result SHA-256:
  `6a4a2c2bc783d9aff4f189b530f46678278b406fb7a042c6dd9e3f6cc6161146`
- Failure result hash:
  `d9311f04a0a44c133993a6eb8c0d023c9b2704d364608e26b8f63196d0d4ca65`
- Executed runner SHA-256:
  `521b9e194a63a09d27f16765747938308c6aa7c8c74e8d5871d36264f881aee4`
- Raw failure log SHA-256:
  `b76be448e78f883688091dd6414974168e047239b8d5e155b4db69ac4046342c`

The last committed row shard was index 340 with terminal shard hash
`3536b15b28ad46f911896dd2622e5f059d0489beaee700a004d2710133cfaecd`.
The in-flight sentinel bound row index 341 and source-row hash
`539ff2a7ac56f1559cac390c9c31a3635df79f880e66cf0e3cfb6adfe292a48b`.
No partial checkpoint output may be reused.

## Root cause

The failed row was the frozen 2020-12-07 source row:

- policy prompt tokens: **29,727**;
- relation prompt tokens: **29,728**;
- selected eligible relation units: **57**; and
- policy prompt bytes: **122,113**.

The `sdpa` implementation materialized an attention allocation consistent with
the full long-prompt query/key surface. The 4-bit model weights fit, but the
single-pass quadratic attention workspace did not fit the 31.84 GiB device.
This is an operational sequence-length failure, not an alpha result and not a
market-data result.

## Authorized research direction

S1 itself remains terminal. A new successor may be considered only under a
fresh preregistration that:

1. binds this exact failure and prohibits S1 checkpoint reuse;
2. keeps the model, revision, prompts, source roster, relation mapping, and
   no-market boundary unchanged;
3. replaces the single-pass long-context operation with a deterministic
   chunked causal-cache operator;
4. proves chunked-versus-one-pass equivalence on a fixed pre-market source
   challenge before full source extraction; and
5. terminally rejects on equivalence, placement, memory, or source-integrity
   failure.

2020 outcomes remain sealed until such a successor completes successfully.
