# PSIM-D2 terminal source-support rejection

Date: 2026-07-26 KST

## Terminal decision

The sealed PSIM-D2 source run was executed exactly once from:

```text
seal commit 6ea1b283c6f35b8c8eeeac838869968aa7560ae0
seal hash b6a101b2d6f41b70ac789ed243b8315589c109c4247d81e14c08d42c5aae0f27
seal SHA256 60dfe828df03751754d056366f977d6626ec00f3b795b5570f854918f022d800
```

It terminally rejected at Gate 4:

```text
historical_blob_preamble_dependency_integrity
REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

The canonical rejection artifact is:

```text
results/protocol_specification_intent_maturity_d2_source_rejection_2026-07-25.json
SHA256 461ea699ada0d6873422c537e63f5fcff3bca56a436caae9aeff4bb74761ca24
result_hash b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c
```

PSIM-D2 is not repaired or rerun. No D2 model/economic stage is authorized.
This is an operational source-run rejection, not evidence that the proposed
specification-intent relation has or lacks economic alpha.

## One-shot timing and environment

The run began at:

```text
2026-07-26T01:36:58Z
2026-07-26T10:36:58+09:00
```

The rejection artifact was atomically published at:

```text
2026-07-26T10:51:19.521431468+09:00
```

The measured evaluator duration was:

```text
real 861.47 seconds
user 114.07 seconds
sys 27.88 seconds
```

WSL disk use was `292 GiB` before and after the run, below the frozen
`300 GiB` limit. The four source roots occupied approximately `27 MiB`.

The shell wrapper attempted to assign the Python exit code to zsh's reserved
read-only `status` variable after Python had already published the rejection.
That wrapper postamble emitted an error but did not alter the evaluator,
artifact, result hash, released lock, or terminal decision.

## Gates that completed

### Gate 1: passed

All four independent roots passed the new D2 bare-object-database contract:

```text
/tmp/psim-d2-source/ethereum-a.git
/tmp/psim-d2-source/ethereum-b.git
/tmp/psim-d2-source/bitcoin-a.git
/tmp/psim-d2-source/bitcoin-b.git
```

Every Gate 1 check was true:

- exact official remote and frozen tip;
- fresh independent roots;
- bare repository with no worktree;
- exact absolute Git directory and common directory;
- exact `HEAD`, branch, sealed ref, and two-ref roster;
- SHA-1 commit object;
- no `.git`, index, checkout, shallow marker, alternate, symlink, or
  multi-link object;
- `git fsck --no-dangling`;
- no `git status`; and
- disk below `300 GiB`.

This proves PSIM-D2 corrected PSIM-D1's invalid no-checkout porcelain
assumption.

### Gate 2: passed

Complete first-parent replay was byte-identical between replicas:

| protocol | commits | replica hash |
|---|---:|---|
| Ethereum | 6,958 | `c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9` |
| Bitcoin | 1,482 | `7e60f24b78aa863a2b317a7dc3a32b2af8e367c3d25f4a97012f4ddfd28d89d2` |

Effective days were monotone, first-parent continuity held from root to the
sealed tip, and traversal used `refs/psim-d2/sealed-tip`.

### Gate 3: passed

Path/object incidence was byte-identical and issue-free:

| protocol | retained 2020–2023 proposal groups | replica hash |
|---|---:|---|
| Ethereum | 4,985 | `a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c` |
| Bitcoin | 371 | `3f7a8e10bb5f9ba57bb0231b5cd54a613fb81e67830c1ec1d9781fe0d22b6a8b` |

Both replicas reported no duplicate-tree-path, ambiguous-old/new,
event-identity, or source-interval issue.

### Gate 4: failed before semantic parsing

The access ledger at failure was:

```text
git_commands 21211
network_commands 13
source_path_rows_opened 16312
proposal_blobs_opened 259
proposal_text_rows_opened 0
daily_cards_built 0
```

No complete event was materialized because the blob batch did not finish.
Therefore no historical preamble, dependency, section, event, card, control,
model, or economic conclusion was produced.

## Root cause

The failure was a deterministic transport/process deadlock in the sealed
inherited D1 `_cat_file_batch` implementation:

1. the bare repositories were partial clones with `blob:none`;
2. `git cat-file --batch` requested missing historical blobs;
3. Git performed one lazy fetch after another;
4. Ethereum replica `a` accumulated `212` separate `.pack` and `212`
   `.promisor` files;
5. Git's automatic maintenance threshold triggered
   `git maintenance run --auto --no-quiet`;
6. maintenance launched `git gc --auto --no-quiet`;
7. the GC process remained in kernel wait channel `pipe_write` for more than
   eight minutes; and
8. `_cat_file_batch` had configured `stderr=PIPE` but reads stderr only after
   all requested objects finish, so the producer and consumer could not make
   progress.

The residual Git log was:

```text
Auto packing the repository in background for optimum performance.
See "git help gc" for manual housekeeping.
```

Its SHA-256 was:

```text
44677f705137e2381bbdbf6739c20637e61ea493713c7e26063b02b0cbd02c69
```

Process evidence showed:

```text
python -> git cat-file --batch
       -> git fetch ... --filter=blob:none --stdin
       -> git maintenance run --auto --no-quiet
       -> git gc --auto --no-quiet  [pipe_write]
```

The source directories and pack sizes stopped changing while this process
tree remained fixed. Draining the pipe, changing Git configuration, replacing
the batch implementation, prefetching blobs, or restarting would have been a
post-incidence repair forbidden by the preregistration.

To terminalize the already deadlocked one-shot attempt without salvaging its
result, only the stuck Git child subtree was sent `SIGTERM`. Python then
received a Git batch `RuntimeError`, failed Gate 4, published the canonical
rejection, and released the run lock. No code, parser, source row, threshold,
provider, or result was changed, and the run was not repeated.

## Forbidden-access proof

All forbidden fields remained zero:

- BTC market rows;
- funding rows;
- future returns;
- rewards;
- model loads and outputs;
- trades;
- PnL;
- CAGR; and
- strict MDD.

Pre-2020 and post-2023 proposal-blob counters also remained zero. The terminal
result explicitly records:

```text
profitability_result false
outcomes_opened false
artifacts null
events 0
daily_cards 0
```

## Publication state

The rejection is the only D2 terminal artifact. The run lock is absent. All
pass targets are absent:

```text
results/protocol_specification_intent_maturity_d2_source_support_2026-07-25.json
data/protocol_specification_intent_maturity_d2_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_d2_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_d2_source_controls_2026-07-25.json
results/.psim_d2_source_support_run.lock
```

The source roots are retained only as one-shot forensic residue. They are not
authorized for reuse, cache, repair, continuation, or another candidate.

## Stop condition and next boundary

PSIM-D2 is terminally rejected unchanged. The following are forbidden:

- rerunning D2;
- draining or patching D2 and continuing;
- reusing any D2 Git object;
- relaxing Gate 4;
- dropping unparsed proposals;
- changing parser, source interval, support floors, controls, or splits; and
- starting a D2 model/economic stage.

A later newly named candidate may be considered only through a fresh
outcome-blind decision and preregistration. Such a candidate must use new
independent source roots and explicitly solve batch hydration before source
incidence—for example one deterministic no-maintenance object hydration
transaction—without changing the inherited parser or source-support
thresholds. That would be a new transport mechanism, not a PSIM-D2 rerun.
