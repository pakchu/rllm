# PSIM-D2 source-support implementation contract

Date: 2026-07-25

## Scope

This document binds the implementation of:

```text
training/build_protocol_specification_intent_maturity_d2_source_support.py
```

PSIM-D2 is source-support-only. It makes no model, alpha, profitability,
portfolio, leverage, or live-trading claim. Official EIP/BIP source may be
opened only after the implementation, tests, this contract, and their exact
direct-child execution seal are committed and validated.

PSIM-D1 remains terminally rejected and is never repaired or rerun. PSIM-D2
imports PSIM-D1's already sealed causal parser, event, card, support, control,
and gates 2–12 implementation as a byte-bound library. PSIM-D2 owns only:

- D2 authority and execution-seal validation;
- fresh bare-object-database acquisition;
- Gate 1's bare repository receipt and validation;
- traversal from `refs/psim-d2/sealed-tip`;
- D2-namespaced source artifacts and terminal reports; and
- D2 one-shot orchestration.

## Frozen authority

The evaluator binds:

| authority | commit | SHA-256 / manifest |
|---|---|---|
| D2 decision | `73de336a1d24399927d43e08c8394450b1cd1cb0` | `e68c0217a6aa3927c88c1f48d9c45ed0b2be3cee4bc3c86d3cb4c6a88e1f8598` |
| D2 preregistration unit | `e853f7688a484b323c024115e3ef4af07e6a5896` | JSON `3b405de2bcdc1979855e8505148f7de3fbee366cb126e78b1b23e10f84cf470a`; manifest `917d2f318b268b01621c9e969237d76fc82d7e6aff408269842e660cc155d915` |
| D2 preregistration producer | `e853f7688a484b323c024115e3ef4af07e6a5896` | `e69b9d0a44f4ffe39657fb46eb9b92b0fe40b1e3041f289fa15bf40e83c2e679` |
| D2 preregistration document | `e853f7688a484b323c024115e3ef4af07e6a5896` | `b231f4fee51eeb7958a656ca36e7f08360ad950fab6e33e1fd0c652ae4d20b2c` |
| D1 terminal rejection | `2a7e4d72d56ff29e90075b3fb872c58c8dd5e310` | `9b0b2354c6edbcfe627527bf4370a4eb0c1e6c1bcb76843f843d9028b16e6494` |
| sealed D1 core runner | `80b656994f17548a7a599a548e23e9f1cd01302d` | `414e83256b3ea489a9e1cd0995f6061e5fab550cd12c795ef7e88eff8998d9fb` |
| sealed D1 core tests | `80b656994f17548a7a599a548e23e9f1cd01302d` | `343aa1a72cfbca23d9756988ced042b5c61a6e8fc5a21a0b6d18e45870e906e9` |

Any worktree, Git-blob, commit, canonical-JSON, manifest, runtime Git-version,
or inheritance-proof mismatch fails before official source access.

The D1 core may not be modified for D2. Its synthetic identity is also frozen:

```text
manifest_hash 24ad04222852e97ffbd37067102cb52b2e38d5d992fd4641ab416b0670168a61
stdout SHA256 4acc071bee5de333c804da59273d5d0ad1fcfc4e735e6f0ac78b5c1539e65a88
```

## Execution seal topology

The D2 runner, its dedicated evaluator tests, and this contract must share one
clean implementation `HEAD`. Seal creation reruns:

- the D2 synthetic self-check in a subprocess;
- the sealed D1 core evaluator tests;
- the D2 preregistration tests; and
- the D2 evaluator tests.

The execution seal and its dedicated seal test must be the only paths changed
in the exact direct-child commit. The official source run is authorized only
while that seal commit is the exact current `HEAD`.

## Exact bare acquisition

The local runtime is frozen to:

```text
git version 2.43.0
```

All inherited `GIT_*` variables are removed. System and global Git
configuration are disabled. Four roots must be absent:

```text
/tmp/psim-d2-source/ethereum-a.git
/tmp/psim-d2-source/ethereum-b.git
/tmp/psim-d2-source/bitcoin-a.git
/tmp/psim-d2-source/bitcoin-b.git
```

Each root uses:

```text
git clone --bare --filter=blob:none --single-branch --branch master --no-tags <remote> <root>
```

Then:

```text
git -C <root> remote get-url origin
git ls-remote --symref <remote> HEAD
git -C <root> fetch --no-tags --filter=blob:none origin <sealed-tip>
git -C <root> update-ref refs/psim-d2/sealed-tip <sealed-tip> <zero-oid>
```

The complete Gate 1 receipt requires:

- exact official remote, `master`, and frozen sealed tip;
- bare repository `true` and inside-worktree `false`;
- absolute Git directory equal to the configured root;
- common directory `.`;
- symbolic `HEAD` and local branch `refs/heads/master`;
- exact ref roster `refs/heads/master` and
  `refs/psim-d2/sealed-tip`;
- sealed ref resolving to the exact frozen SHA-1 commit;
- object format `sha1` and object type `commit`;
- no `.git`, index, worktree, commondir, gitdir, alternate, or shallow path;
- no object-store symlink or multi-link file;
- `git fsck --no-dangling`;
- no checkout and no `git status` invocation; and
- disk use at or below `300 GiB`.

A fresh exact clone command plus absent roots prohibits copied or reused D1
state. Alternate, symlink, and hard-link checks prohibit shared object stores.

## Frozen traversal and inherited core

Commit traversal begins only from:

```text
git -C <root> rev-list --first-parent --reverse refs/psim-d2/sealed-tip
```

The resolved sealed ref must equal the preregistered tip before traversal.
The moving local `master` and current remote `HEAD` are identity audit only.

After Gate 1, the exact D1 core implements all inherited semantics:

- SHA-1 recomputation for commit and blob objects;
- complete first-parent continuity and causal running-maximum day;
- `[2020-01-01, 2024-01-01)` source incidence;
- no pre-window event warm-up;
- first old side only as `PRE_WINDOW_BASELINE`;
- exact EIP/BIP path and historical preamble grammar;
- strict UTF-8, dependency, section, diff, and bucket rules;
- D2/D7/D30/D90 archive clocks;
- deterministic daily cards and pairing;
- exact famous-proposal quarantine;
- seven exact relation controls;
- split/support/vocabulary/card/replay/future-append gates; and
- zero market, model, reward, trade, PnL, CAGR, and strict-MDD access.

The D2 preregistration's structural inheritance proof is reloaded and rebuilt
at runtime. No downstream parser, threshold, control, split, schedule,
quarantine, representation vocabulary, or later economic criterion may
change.

## Gates and stop condition

All thirteen preregistered gate names execute in their inherited order. The
first failed gate stops evaluation and publishes only:

```text
REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

There is no repair, source drop, parser relaxation, threshold relaxation,
control removal, schedule removal, quarantine change, provider swap, rerun,
or transition to a market/model stage after rejection.

A source pass publishes only the D2-namespaced event, card, control, and source
result artifacts. It authorizes a later separately preregistered model and
economic protocol, not profitability.

## Publication safety

The runner requires an exclusive no-follow run lock before source access.
Pass artifacts are staged and hard-linked as one rollback-capable group.
Existing, partial, conflicting, escaped, or symlinked output paths reject.
Publication failure becomes Gate 13 rejection and removes runner-owned staged
links. Rejection and pass terminal states are canonical, hashed, mutually
exclusive, and non-overwriting.

The frozen output paths are:

```text
results/protocol_specification_intent_maturity_d2_source_support_2026-07-25.json
results/protocol_specification_intent_maturity_d2_source_rejection_2026-07-25.json
data/protocol_specification_intent_maturity_d2_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_d2_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_d2_source_controls_2026-07-25.json
```

## Synthetic verification before commit

The D2 self-check produced:

```text
stdout SHA256 744c0ae216e64a67c6737c8d86a1301450837978c4da2b158e02e15820163f42
manifest_hash fc0edec7e68eb5caa1bcde0cfe06aea46d89b07fb690aa02d628767b5e12ce4a
failed []
git_commands 0
network_calls 0
source_event_rows_opened 0
official_source_opened false
outcomes_opened false
```

The D2 evaluator-specific battery passed:

```text
17 passed
```

The seal-target combined battery passed:

```text
71 passed
```

It includes real local synthetic Git tests proving:

- exact bare clone argument order;
- no `git status`;
- no worktree/index/shared object store;
- exact ref roster;
- separate moving `master` and frozen sealed-ref traversal;
- fresh-root rejection before transport;
- hostile Gate 1 receipt rejection;
- D2-only non-profitability result identities;
- safe output and source-root boundaries;
- first-failure stop behavior;
- all thirteen mocked gates and atomic pass publication; and
- Gate 13 publication-failure conversion.

No `/tmp/psim-d2-source` root or official EIP/BIP source was opened while
building or testing this implementation.

An independent `code-reviewer` pass returned `PASS` with no concrete blocking
defect. It independently reran the 17-test D2 evaluator battery, syntax
compilation, D2 zero-access self-check, and D1 core commit/hash verification.
Its declared residual boundaries were the not-yet-created seal-test artifact,
the intentionally unexecuted official source run, and unavailable LSP
diagnostics. The direct-child seal test and sealed one-shot source run remain
separate required work units.
