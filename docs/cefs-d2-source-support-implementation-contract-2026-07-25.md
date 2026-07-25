# CEFS-D2 source-support implementation contract

Date: 2026-07-25

Status: **frozen before any CEFS-D2 source value is decoded.**

## Authority

```text
policy_id
  CEFS-D2
boundary
  docs/cboe-edge-flip-sequence-policy-d2-boundary-2026-07-25.md
  commit 2ee9831a34d808ba3a3abcdf9f93281ee2dc8ef8
  SHA256 d679d23465963485756bde7f7cc7660607cfbcc1326baaaf367b3eaac34e387b
preregistration producer
  training/preregister_cboe_edge_flip_sequence_policy_d2.py
  commit a7ac3239122adf06ded0f374c582cbca9df253da
  SHA256 ab0021a327c351e030098c388ae6a8d35016edcfc50cba42177c3206b4c6a9e4
preregistration artifact
  results/cboe_edge_flip_sequence_policy_d2_preregistration_2026-07-25.json
  commit 203e6b05ea85e36b099cf1f2e1752344fbcbdb77
  SHA256 9f5afdb4647de01e7f5c5130fba4b68cf0c10824f90f33802b57da33d314de2a
  manifest_hash fa1da6ebd2ecc674d64aa95ca0860434db071a62ffbe3442218c73699d312e00
```

The preregistration embeds and hashes all fourteen non-identity,
non-authority scientific sections from the sealed CEFS-D1 preregistration.
The D2 implementation may not project, reinterpret, or replace any of them.

## Exact pure-engine dependency

The runner imports pure deterministic support functions only from:

```text
training/build_cboe_edge_flip_sequence_policy_support.py
commit d7213f647128fc6160672bc61f080b3dcf7d1f42
SHA256 2069084d65146540488672115ee09f292cd31e6611bf92a569d534ab8a74c688
```

Permitted imports:

- physical panel parsing and immutable prefix snapshots;
- exact common-date join and VIX identity;
- exact comparison, ratio cross multiplication, and edge construction;
- five-state sequence and three prompt serialization;
- fixed clock, role containment, and overlap reservation;
- source-only control generation;
- source-support metric/check functions;
- deterministic prefix/synthetic append replay;
- deterministic CSV/gzip and atomic write-once publication; and
- strict terminal output/detail semantic validators.

Forbidden:

- D1 `run_official`;
- D1 seal/pass/rejection paths;
- mutating D1 module constants or state;
- monkeypatching a D1 policy ID or terminal action; and
- copying a D1 terminal result into D2.

The D2 execution seal must bind the exact D1 engine dependency in addition to
the D2 runner and tests.

## D2 implementation files and outputs

```text
runner
  training/build_cboe_edge_flip_sequence_policy_d2_support.py
tests
  tests/test_build_cboe_edge_flip_sequence_policy_d2_support.py
execution seal
  results/cefs_d2_source_support_execution_seal_2026-07-25.json
pass source
  data/cboe_edge_flip_sequence_policy_d2_source_2020_2023.csv.gz
pass controls
  data/cboe_edge_flip_sequence_policy_d2_controls_2020_2023.csv.gz
pass report
  results/cboe_edge_flip_sequence_policy_d2_source_support_2026-07-25.json
rejection report
  results/cboe_edge_flip_sequence_policy_d2_source_rejection_2026-07-25.json
```

All paths are repository-relative, create-once, regular non-symlink files.
Runner and tests must share one commit. A separately committed execution seal
must bind their commit and bytes before any real source value is decoded.

## Runtime precondition before source access

The runner must never use executable-name lookup for Git.

```text
Git executable
  /usr/bin/git
SHA256
  2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

Before Gate 1 and before source access:

1. require nonempty `PATH`;
2. require `/usr/bin` as an exact `PATH` component;
3. validate `/usr/bin/git` as a regular non-symlink file;
4. validate its exact SHA256;
5. execute `/usr/bin/git --version`;
6. use `/usr/bin/git` for every Git operation;
7. validate all D2 and inherited D1 authority bytes/commits;
8. validate the execution seal;
9. require a clean worktree; and
10. require no partial or conflicting D2 terminal state.

The runner and its official command may not assign a shell variable named
`path`. Tests must prove absolute Git still works when `PATH` itself cannot
resolve `git`, while the frozen runtime precondition rejects that environment
before source access.

## Physical parsing

Physical parsing is exactly the sealed D1 engine contract:

- exact UTF-8 headers and whole-file hashes;
- ASCII ten-byte Gregorian dates;
- dates parsed before non-date fields;
- strictly increasing unique physical dates;
- pre-2020 term/tail rows date-only;
- no at-or-after-2024 non-date parse;
- ASCII positive plain decimals via `Decimal`;
- ASCII positive base-ten volume integers;
- exact option-flow relation fields only;
- integrity-only `response_sha256`;
- exact term/tail VIX equality; and
- exact sorted three-panel date intersection.

During the one full physical parse, immutable prefix record containers are
sealed before the first row at each cutoff enters the date parser:

```text
2021-01-01
2022-01-01
2023-01-01
2024-01-01
```

Prefix rebuilds receive only those containers and may not inspect later rows
or dates.

## Exact schedules and controls

Use the full inherited D1 scientific contract without change:

- twelve primitive exact edges;
- five ordered edge states;
- exact flat/long/short current-target prompts;
- D+1 09:30/09:35 New York availability/entry;
- exact 288 five-minute bar hold;
- action-independent global reservation, equality accepted;
- action-independent `ROLE_CROSSING` and suppressed audit rows;
- controls only for complete TRAIN/TEST/EVAL primary rows; and
- exactly three positions × eight controls for every eligible schedule.

The source and control CSV columns and ordering are byte-identical to D1,
except for the new D2 output paths.

## Frozen gate sequence

Run exactly:

1. `runtime_authority_forbidden_access`;
2. `schema_chronology`;
3. `schedule_support`;
4. `primitive_edge_support`;
5. `state_diversity_stability`;
6. `source_only_controls`;
7. `determinism_append_replay`.

Gate 1 checks:

- runtime authority valid;
- absolute Git valid;
- frozen D2 authority valid;
- exact inherited D1 authority valid;
- execution seal valid;
- worktree clean; and
- every forbidden counter zero.

Gates 2–7 use the exact D1 metric/check functions and thresholds. Gate 3 must
compare full source schedule records, not a reduced clock projection. Gate 7
must compare complete ordered prefix schedule/control lists bidirectionally,
build twice byte-identically, and prove both prior schedules and controls
unchanged after the one fixed synthetic append.

## First-stop failure

At the first failed gate:

- stop all later stages;
- write no source/control pass CSV;
- write no pass report;
- atomically write only the D2 rejection report; and
- return exact action:

```text
retire_cefs_d2_unchanged_before_outcomes
```

The report contains only reached stage details, exact ordered gate records,
zero forbidden counters, canonical row hashes only if those rows were
constructed, and a self-consistent result hash.

## Pass publication

Only after all seven gates pass:

1. build deterministic source/control records twice;
2. build deterministic gzip bytes;
3. stage source gzip, control gzip, and pass report;
4. atomically publish with no-overwrite links;
5. roll back every newly published final path on any normal exception; and
6. return exact action:

```text
authorize_cefs_d2_economic_rllm_evaluator_freeze_only
```

The pass report binds gzip SHA256, decoded row count, canonical row hash,
authority, seal, gates, details, and forbidden counters.

## Terminal-state validation

Before Gate 1:

- a valid existing pass or rejection returns idempotently with no writes and
  no new gate evaluation;
- a conflicting, partial, unreadable, hash-drifted, or semantically invalid
  terminal state hard-aborts with no writes;
- pass outputs without a pass report are partial; and
- a rejection with pass outputs is conflicting.

A terminal pass is valid only if the runner:

- validates exact report fields, hash, policy, and action;
- validates all seven ordered passing gates;
- reconstructs every Gate 2–7 check from exact nested details;
- rejects unknown/missing nested detail fields or prefix keys;
- validates deterministic gzip, exact output schemas, row counts and
  canonical row hashes;
- reconstructs every schedule's clock, reservation, role, eligibility,
  sequence, current signature, and three prompts;
- regenerates every control record exactly;
- replays schedule, edge, diversity, and control metrics; and
- validates exact current runner/test/seal/prereg/D1 engine authority.

A terminal rejection must have exactly the reached gate prefix, all earlier
gates passed, the last gate failed, no later details, no pass output, and
stage-consistent canonical row hashes.

## Forbidden evidence

Before source-support pass, all remain zero:

```text
post_2023_source_non_date_rows_opened
market_rows_opened
funding_rows_opened
future_return_rows_built
reward_rows_built
model_rows_built
selected_action_rows_built
trade_rows_built
pnl_cagr_mdd_values_computed
comparator_action_rows_opened
```

No model family, reward, checkpoint, selected action, BTC outcome, funding,
trade, CAGR, MDD, or live claim is authorized by this contract.

## Execution order

1. commit this contract;
2. implement D2 runner/tests together without source decoding;
3. pass synthetic tests and independent review;
4. commit runner/tests together;
5. run only `create-seal`;
6. add exact seal artifact/tests and commit;
7. validate seal and all source-support tests on a clean worktree;
8. invoke the D2 runner directly with repository Python, without a shell
   preflight loop; and
9. stop permanently at the first failed gate or commit the complete pass.
