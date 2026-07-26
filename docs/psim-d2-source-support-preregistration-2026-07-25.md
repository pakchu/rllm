# PSIM-D2 source-support preregistration

Date: 2026-07-25

## Decision

PSIM-D2 is preregistered as a newly named source-only successor to the
terminally rejected PSIM-D1 candidate. It preserves PSIM-D1's parser,
historical replay, archive, card, control, support, split, quarantine, and
forbidden-access contracts. Its only substantive change is Gate 1's
repository representation: four fresh independent bare Git object databases
replace D1's invalid no-checkout worktree-cleanliness assertion.

This artifact does not reopen or rerun PSIM-D1. It does not authorize official
EIP/BIP source access, a model, market or funding data, a reward, a trade, PnL,
CAGR, strict MDD, leverage, a portfolio, or live trading. The next unit may
only implement and seal a synthetic-only PSIM-D2 evaluator.

The successor decision is:

```text
docs/post-psim-d1-alpha-mechanism-audit-2026-07-25.md
SHA256 e68c0217a6aa3927c88c1f48d9c45ed0b2be3cee4bc3c86d3cb4c6a88e1f8598
commit 73de336a1d24399927d43e08c8394450b1cd1cb0
```

The D1 terminal authority is:

```text
results/protocol_specification_intent_maturity_source_rejection_2026-07-25.json
SHA256 9b0b2354c6edbcfe627527bf4370a4eb0c1e6c1bcb76843f843d9028b16e6494
result_hash 5815f7473410c7d75aabea8b6a97cfb7f963b1c6d29f8efa22f0a0a64d33655d
commit 2a7e4d72d56ff29e90075b3fb872c58c8dd5e310
```

PSIM-D1 remains terminally rejected:

```text
REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

## Machine-readable contract

The machine-readable D2 preregistration is:

```text
results/protocol_specification_intent_maturity_d2_preregistration_2026-07-25.json
SHA256 3b405de2bcdc1979855e8505148f7de3fbee366cb126e78b1b23e10f84cf470a
manifest_hash 917d2f318b268b01621c9e969237d76fc82d7e6aff408269842e660cc155d915
authorized_delta_hash e8a6714b81ab0d89a6ddd54157310cab89d7a02881c93bbd6241efd472dbaa48
```

The builder and tests are:

```text
training/preregister_protocol_specification_intent_maturity_d2.py
SHA256 e69b9d0a44f4ffe39657fb46eb9b92b0fe40b1e3041f289fa15bf40e83c2e679

tests/test_preregister_protocol_specification_intent_maturity_d2.py
SHA256 c5e705e752e7e6fe2dbe7cbe855bb1ead2727fb7092d466c766ded033c93c3f8
```

The D2 builder loads the exact canonical D1 preregistration and terminal
rejection, validates their SHA-256 and manifest/result identities, removes
only D1's outer manifest hash, applies the explicit D2 delta, and recursively
compares every leaf. Construction fails unless the observed changed-path set
is exactly the frozen authorized-path set. The resulting inheritance proof
stores the before/after values and a canonical delta hash. All unlisted
contract paths remain structurally and canonically identical.

## Frozen bare repository acquisition

The official Git semantics are documented by:

- <https://git-scm.com/docs/git-clone>
- <https://git-scm.com/docs/git-rev-parse>
- <https://git-scm.com/docs/git-fsck>

The bound local version is:

```text
git version 2.43.0
```

Every official source root must be absent before the one-shot run:

```text
/tmp/psim-d2-source/ethereum-a.git
/tmp/psim-d2-source/ethereum-b.git
/tmp/psim-d2-source/bitcoin-a.git
/tmp/psim-d2-source/bitcoin-b.git
```

Each independent root must use this exact argument sequence:

```text
git clone --bare --filter=blob:none --single-branch --branch master --no-tags <remote> <fresh-root>
```

All inherited `GIT_*` variables must be removed and system/global Git
configuration disabled. No D1 object, alternate, hard link, copied cache,
checkout, index, linked worktree, remote-tracking ref, or tag is allowed.
`git status` is forbidden because a bare repository has no worktree/index
cleanliness contract.

After exact remote identity validation and sealed-tip fetch, every root must
contain exactly:

```text
refs/heads/master
refs/psim-d2/sealed-tip
```

The evaluator must prove:

- bare repository `true`;
- inside worktree `false`;
- absolute Git directory equals the configured root;
- common directory is `.`;
- symbolic `HEAD` is `refs/heads/master`;
- sealed ref equals the frozen SHA-1 commit;
- object format is `sha1`;
- `.git`, `index`, `worktrees`, `commondir`, `gitdir`, alternates, and shallow
  markers are absent;
- `git fsck --no-dangling` passes; and
- WSL disk use is at most `300 GiB`.

All traversal starts from `refs/psim-d2/sealed-tip`, never from a moving
branch.

## Frozen source identities

| protocol | exact remote | branch | sealed tip | object format |
|---|---|---|---|---|
| Ethereum | `https://github.com/ethereum/EIPs.git` | `master` | `5e82ef62895121027a6c5f0c23276e1b2bed3071` | `sha1` |
| Bitcoin | `https://github.com/bitcoin/bips.git` | `master` | `b289d016b99c81527623c10e995e0318f744ebf3` | `sha1` |

The exact path grammars remain:

```text
^EIPS/eip-([1-9][0-9]*)\.md$
^bip-([0-9]{4})\.(mediawiki|md)$
```

## Mechanically inherited D1 contract

The following are byte-equivalent under canonical serialization outside the
frozen D2 delta:

- source interval `[2020-01-01, 2024-01-01)`;
- card interval through `2024-04-01`;
- complete first-parent causal traversal;
- SHA-1 commit/blob recomputation;
- running-maximum UTC committer-day clock;
- no pre-window event warm-up;
- first old blob only as `PRE_WINDOW_BASELINE`;
- exact path, event, preamble, dependency, section, and bucket grammars;
- strict parser rejection without repair;
- `ARCHIVE_D2`, `D7`, `D30`, and primary `D90`;
- one daily relation card at `12:05Z`;
- deterministic Cartesian/trailing-90-day pairing;
- famous-proposal quarantine;
- seven exact relation controls;
- per-cell control-sensitivity floor;
- train/test/eval source-support floors; and
- zero model, market, funding, outcome, reward, trade, PnL, CAGR, and strict
  MDD access.

No parser, support threshold, control, split, quarantine, schedule, model
vocabulary, or later economic criterion changed.

## Frozen gates and terminal action

The exact thirteen D1 gate names and order remain:

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

The first failed D2 gate retires D2 unchanged:

```text
REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No source repair, drop, parser change, threshold change, schedule change,
control change, quarantine change, rerun, or model/economic work is allowed
after the first official D2 source attempt.

## Verification

The combined D1 and D2 preregistration battery passed:

```text
70 passed
```

The D2-specific battery proves:

- exact D1 authority and terminal-result binding;
- exhaustive structural delta equality;
- parser/split/support/control inheritance;
- exact bare repository shape;
- inherited source identities;
- D2-namespaced artifact paths;
- canonical and idempotent output;
- fail-closed authority, delta, split, conflict, and symlink cases; and
- absence of network, Git execution, market, model, or trading imports/calls.

All forbidden counters remain zero. Official PSIM-D2 source incidence remains
closed until the evaluator, its synthetic adversarial tests, and an exact
direct-child execution seal are committed.
