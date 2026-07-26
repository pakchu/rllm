# Post-PSIM-D1 audit and PSIM-D2 source-axis decision

Date: 2026-07-25

## Scope

This document selects a newly named successor candidate after the terminal
PSIM-D1 source-support rejection. It does not reopen, repair, or rerun
PSIM-D1. It makes no source-support, model, alpha, profitability, or
live-trading claim.

## PSIM-D1 terminal boundary

The terminal evidence is bound to commit:

```text
2a7e4d72d56ff29e90075b3fb872c58c8dd5e310
```

and to:

| artifact | SHA-256 |
|---|---|
| `results/protocol_specification_intent_maturity_source_rejection_2026-07-25.json` | `9b0b2354c6edbcfe627527bf4370a4eb0c1e6c1bcb76843f843d9028b16e6494` |
| `docs/psim-d1-source-support-terminal-rejection-2026-07-25.md` | `8a6fdc27d980fb43316799d395c7f2bc214e58f99bcb402aeb500244f83e8e96` |

PSIM-D1 failed Gate 1,
`sealed_git_identity_and_object_integrity`, after eight Git commands and three
network-capable commands. Only remote identity was opened. Commit metadata,
proposal path incidence, historical proposal blobs, cards, models, market
data, funding, outcomes, rewards, trades, PnL, CAGR, and strict MDD remained
sealed.

The failure was structural: `git clone --no-checkout` produced an empty index
and no worktree files, so `git status --porcelain=v1` correctly represented the
entire `HEAD` tree as staged deletions. PSIM-D1 incorrectly treated nonempty
porcelain output as contamination.

PSIM-D1 remains terminally rejected unchanged:

```text
REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

## Official Git semantics used for the successor

Git's official clone documentation states that `--bare` makes the destination
itself the Git directory and implies no checkout because there is no worktree.
It also states that `--filter=blob:none` omits blob contents until needed:

- <https://git-scm.com/docs/git-clone>

Git's official `rev-parse` documentation defines
`--is-bare-repository`, `--is-inside-work-tree`, `--absolute-git-dir`, and
`--git-common-dir`:

- <https://git-scm.com/docs/git-rev-parse>

Git's official `fsck` documentation defines object-database connectivity and
validity checking:

- <https://git-scm.com/docs/git-fsck>

The bound local execution version is:

```text
git version 2.43.0
```

A synthetic local-only probe under Git 2.43.0 established the intended shape
without opening official EIP/BIP source:

```text
--is-bare-repository   true
--is-inside-work-tree  false
--absolute-git-dir     <clone-root>
--git-common-dir       .
symbolic HEAD          refs/heads/master
index                  absent
.git directory         absent
alternates             absent
linked worktrees       absent
fsck --no-dangling     pass
```

This probe may define repository-shape assertions only. It may not alter any
source-support threshold, semantic direction, model label, economic split, or
trading target.

## Selected successor

The next candidate is:

```text
PSIM-D2 — Protocol Specification Intent-Maturity relation RLLM,
bare object-database replay
```

PSIM-D2 preserves PSIM-D1's economic hypothesis:

> Relative maturity, dependency direction, and changed technical intent across
> causally archived EIP and BIP revision streams may form weak, orthogonal
> evidence that a later constrained single-model policy can combine.

The only authorized source-contract change is Gate 1's repository
representation and its corresponding execution-seal identity. No support
floor, split, archive delay, parser rule, event grammar, relation pairing,
control transform, quarantine list, model vocabulary, or later economic
criterion may change.

## Frozen official sources

PSIM-D2 uses the same sealed source tips:

| protocol | remote | branch | sealed tip |
|---|---|---|---|
| Ethereum EIPs | `https://github.com/ethereum/EIPs.git` | `master` | `5e82ef62895121027a6c5f0c23276e1b2bed3071` |
| Bitcoin BIPs | `https://github.com/bitcoin/bips.git` | `master` | `b289d016b99c81527623c10e995e0318f744ebf3` |

It must use four fresh and independent roots:

```text
/tmp/psim-d2-source/ethereum-a.git
/tmp/psim-d2-source/ethereum-b.git
/tmp/psim-d2-source/bitcoin-a.git
/tmp/psim-d2-source/bitcoin-b.git
```

PSIM-D1's object store cannot be reused, referenced, alternated, hard-linked,
copied, or used as a cache.

## Frozen bare-repository acquisition

Each root must be absent before the one-shot run. With all inherited `GIT_*`
variables removed and system/global Git configuration disabled, execute:

```text
git clone --bare --filter=blob:none --single-branch --branch master --no-tags <remote> <fresh-root>
```

Then validate remote identity and fetch the sealed tip:

```text
git -C <root> remote get-url origin
git ls-remote --symref <remote> HEAD
git -C <root> fetch --no-tags --filter=blob:none origin <sealed-tip>
git -C <root> update-ref refs/psim-d2/sealed-tip <sealed-tip> <zero-oid>
```

The following assertions replace the invalid D1 porcelain assertion:

1. `git -C <root> rev-parse --is-bare-repository` is exactly `true`;
2. `git -C <root> rev-parse --is-inside-work-tree` is exactly `false`;
3. `git -C <root> rev-parse --absolute-git-dir` resolves byte-for-byte to the
   configured root;
4. `git -C <root> rev-parse --git-common-dir` is exactly `.`;
5. `git -C <root> symbolic-ref HEAD` is `refs/heads/master`;
6. `refs/psim-d2/sealed-tip` resolves exactly to the sealed tip;
7. the complete ref roster is exactly `refs/heads/master` plus
   `refs/psim-d2/sealed-tip`, with no tag or remote-tracking ref;
8. object format is exactly `sha1`;
9. the sealed tip's object type is exactly `commit`;
10. `index`, `.git`, `worktrees`, `commondir`, `gitdir`, and
   `objects/info/alternates` are absent;
11. the repository is not shallow;
12. `git fsck --no-dangling` exits zero; and
13. total WSL disk use remains at or below `300 GiB`.

`git status` is forbidden for Gate 1 because a bare repository has no
worktree/index cleanliness semantics to audit. No checkout may be created.
The current local `HEAD` and current remote branch tip are identity audit only;
all source traversal starts from the exact `refs/psim-d2/sealed-tip` object.

## Frozen inherited source contract

Everything after repository preparation remains identical to PSIM-D1:

- complete first-parent traversal from repository root to the sealed tip;
- SHA-1 recomputation for commit and blob objects;
- running-maximum UTC committer-day causal clock;
- source interval `[2020-01-01, 2024-01-01)`;
- no pre-window warm-up event;
- first in-window old side only as immutable `PRE_WINDOW_BASELINE`;
- exact EIP/BIP path and preamble grammar;
- no rename detection and NUL-safe byte-exact paths;
- strict parser failure rather than repair;
- D2/D7/D30/D90 archive schedules, with D90 primary;
- one daily `12:05Z` relation card;
- deterministic Cartesian/trailing-90-day pairing;
- exact famous-proposal quarantine;
- the same seven controls and exact per-cell sensitivity threshold;
- the same train/test/eval split-support floors; and
- zero model, market, funding, outcome, reward, trade, PnL, CAGR, and strict
  MDD access.

The inherited source-support manifest must be copied mechanically and hash
compared during preregistration. Only fields whose semantics necessarily
change from D1 to D2 may differ:

- policy/candidate ID and protocol version;
- decision/preregistration/implementation/seal artifact identities;
- source-root and output paths;
- clone arguments;
- bare-repository Gate 1 fields; and
- terminal action names.

## Frozen gates

PSIM-D2 keeps the same thirteen gate names and order:

1. `sealed_git_identity_and_object_integrity`
2. `first_parent_traversal_and_causal_clock`
3. `path_object_grammar_and_unique_proposal_tree`
4. `historical_blob_preamble_dependency_integrity`
5. `split_annual_quarterly_unique_day_support`
6. `event_section_dependency_revision_vocabulary_diversity`
7. `daily_card_coverage_and_explicit_staleness`
8. `independent_replay_and_canonical_manifest_identity`
9. `future_append_invariance`
10. `relation_control_sensitivity`
11. `pairing_reset_quarantine_and_four_schedule_identity`
12. `forbidden_access_zero`
13. `terminal_publication`

The first failure action is:

```text
REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No source repair, source drop, parser change, threshold change, schedule
change, control change, quarantine change, rerun, or model/economic work is
allowed after the first D2 source attempt.

## Execution and stop condition

Before official source access, PSIM-D2 must:

1. commit a machine-readable preregistration proving the D1-inherited contract
   is unchanged outside the authorized delta;
2. implement synthetic bare-repository and causal replay tests;
3. pass an independent adversarial review;
4. commit the evaluator, tests, and implementation contract together;
5. create and commit an exact direct-child execution seal; and
6. run the sealed source gate exactly once.

Passing source support authorizes only a separately preregistered model and
economic stage. It does not itself authorize an alpha, profitability, leverage,
portfolio, or live-trading claim.
