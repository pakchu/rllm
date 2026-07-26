# PSIM-D1 source-support implementation contract

Date: 2026-07-25

## Scope

This file is the binding implementation contract for
`training/build_protocol_specification_intent_maturity_source_support.py`.
It implements only the source-support replay authorized by
`docs/psim-d1-source-support-preregistration-2026-07-25.md` and the sealed
machine preregistration. It makes no profitability claim.

The implementation must not open source incidence until an execution seal has
validated the runner, tests, preregistration artifact, preregistration script,
preregistration document, decision artifact, Git version, Python version, and
synthetic self-check manifest. Before that validation succeeds, the access
ledger must show zero opened source path rows, proposal blobs, proposal text,
dependencies, daily cards, market rows, model rows, trades, and outcomes.

## Sealed inputs

The runner must bind these preregistered artifacts before source access:

| artifact | required identity |
|---|---|
| decision document | `docs/post-gipr-d1-alpha-mechanism-audit-2026-07-25.md` at commit `6ebb43406f7197e2afb2e2fa5cb39b0a2cba2826`, SHA-256 `816fbb19c4ff9a841f75f75555e568f401e804b4aded258779ef4bce14ebaf04` |
| preregistration JSON | `results/protocol_specification_intent_maturity_preregistration_2026-07-25.json`, SHA-256 `bd4053574fe6285c34356baaa080e215f08bbf8142e9c0c968bffbdccb2dc736`, manifest hash `bdf49fb396779599eb329a407685435c05217f132ea856f9bb743914b5afbe81` |
| preregistration script | `training/preregister_protocol_specification_intent_maturity.py`, SHA-256 `982f14cff8c903c9ad528018ab996f053179cc054626bfaa8af7ab63405f858b` |
| preregistration document | `docs/psim-d1-source-support-preregistration-2026-07-25.md`, SHA-256 `09612ec67c093edf952bf54664d1f73a6796c96e2ac3be24b60c804ae700074d` |

Any mismatch rejects before source incidence opens.

The runner, evaluator tests, and this implementation contract must share one
clean implementation `HEAD` when the execution seal is created. The seal and
its dedicated seal test must then be the only two paths changed in the direct
child commit. An official run is valid only while that seal commit is the exact
current `HEAD`; an older ancestor seal cannot be reused after another commit.

## Official Git roots and commands

Use exactly four independent clone roots under the configured source root
(default `/tmp/psim-d1-source`):

| protocol | replica | root | remote | branch | sealed tip |
|---|---|---|---|---|---|
| Ethereum | `a` | `/tmp/psim-d1-source/ethereum-a` | `https://github.com/ethereum/EIPs.git` | `master` | `5e82ef62895121027a6c5f0c23276e1b2bed3071` |
| Ethereum | `b` | `/tmp/psim-d1-source/ethereum-b` | `https://github.com/ethereum/EIPs.git` | `master` | `5e82ef62895121027a6c5f0c23276e1b2bed3071` |
| Bitcoin | `a` | `/tmp/psim-d1-source/bitcoin-a` | `https://github.com/bitcoin/bips.git` | `master` | `b289d016b99c81527623c10e995e0318f744ebf3` |
| Bitcoin | `b` | `/tmp/psim-d1-source/bitcoin-b` | `https://github.com/bitcoin/bips.git` | `master` | `b289d016b99c81527623c10e995e0318f744ebf3` |

All four clone roots must be absent at the start of the one-shot source run.
An existing root rejects instead of being repaired or reused. For each fresh
root, run the exact shape:

```text
git clone --filter=blob:none --no-checkout --single-branch --branch master <remote> <root>
```

Then validate and fetch the sealed tip with the exact command shapes:

```text
git -C <root> remote get-url origin
git ls-remote --symref <remote> HEAD
git -C <root> fetch --no-tags --filter=blob:none origin <sealed-tip>
git -C <root> rev-parse --show-object-format
git -C <root> cat-file -t <sealed-tip>
git -C <root> fsck --no-dangling
git -C <root> status --porcelain=v1
```

The observed origin URL, remote HEAD symref (`refs/heads/master`), frozen local
tracking identity (`refs/remotes/origin/master`), sealed tip, object format
(`sha1`), no-dangling fsck result, empty worktree, absence of shared object
alternates, and disk guard (`<= 300 GiB`) are part of gate 1.
Shared object alternates, shared worktrees, shared caches, current checkouts,
GitHub PR/issue/label/review metadata, rendered pages, and commit subjects are
not allowed as source truth.

## Causal clock and source interval

Traverse each sealed tip with:

```text
git -C <root> rev-list --first-parent --reverse <sealed-tip>
```

For each commit object, verify the Git object SHA-1, parse only the tree, first
parent, and committer epoch, and require first-parent continuity from repository
root to sealed tip. `committer_day` is the UTC calendar day of the committer
epoch. `effective_day` is the running maximum of UTC committer days along the
first-parent chain. Committer time is not a publication receipt.

Source events are limited to:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

Daily cards continue through `2024-04-01T00:00:00Z` exclusive. Do not traverse
or fetch proposal blobs solely for pre-2020 warm-up or post-2023 incidence. The
old-side blob referenced by a retained first in-window `UPDATE` or `DELETE` may
be parsed only as that event's immutable `PRE_WINDOW_BASELINE`; it cannot create
a synthetic event, revision, age, or update gap. The first in-window event for
a proposal therefore has `window_revision_count = 0`, no prior update gap, and
age from its own effective day.

## Path and object grammar

Compute tree deltas against parent one, or the empty tree for the root commit,
with NUL-safe raw output and rename detection disabled:

```text
git -C <root> diff-tree --root --no-commit-id -r --raw -z --no-renames <commit>
git -C <root> diff-tree --no-commit-id -r --raw -z --no-renames <parent> <commit>
git -C <root> ls-tree -r -z --name-only <treeish>
git -C <root> cat-file --batch
```

Accept only these primary proposal paths:

```text
^EIPS/eip-([1-9][0-9]*)\.md$
^bip-([0-9]{4})\.(mediawiki|md)$
```

Raw diff headers must match `:<old_mode> <new_mode> <old_oid> <new_oid>
<status>` with 40-hex SHA-1 object IDs and status `A`, `D`, `M`, or `T`.
Zero OID means the side is absent. Group matching old/new sides by protocol,
commit, and proposal number into exactly one of `CREATE`, `UPDATE`, or
`DELETE`. Reject duplicate paths in a tree, number disagreement, duplicate event
IDs, multiple old or new blobs for one proposal number in one commit, malformed
UTF-8 paths, unsupported status, or any metadata parse failure.

`event_id` is:

```text
SHA256(protocol || NUL || commit_oid || NUL || canonical_decimal_proposal_number || NUL || old_blob_oid_or_NULL || NUL || new_blob_oid_or_NULL)
```

## Parser import and replay

The source-support runner must import the preregistration module and use its
strict parser functions directly:

- `normalize_blob_bytes`
- `parse_eip_preamble`
- `parse_bip_preamble`
- `parse_dependency_ids`
- `parse_positive_proposal_number`

Do not fork, loosen, or reinterpret `PSIM_PREAMBLE_STATE_MACHINE_V1`. Every
historical blob that is legally in the source interval must pass strict UTF-8,
NUL rejection, CRLF/CR-to-LF newline normalization, NFC normalization, trailing
horizontal whitespace stripping, blob/header/line/section/dependency bounds,
header key grammar, duplicate-key rejection after casefold, exact EIP fence or
BIP preamble termination rules, and dependency grammar. The path number and
preamble number must match. Today's EIP/BIP status vocabulary must not be
projected backward, and declared status must not be model visible.

Two independent replicas for each protocol must replay to byte-identical event
JSONL and daily-card JSONL fingerprints. Any disagreement rejects.

## Representation, diff, sections, and buckets

Use Python `difflib.SequenceMatcher(autojunk=False)` over normalized lines.
Changed sections come from old and new line section labels. Intent text may
include only ordered `SECTION|ADD|line` and `SECTION|REMOVE|line` rows for:

```text
ABSTRACT
MOTIVATION
SPECIFICATION
RATIONALE
BACKWARD_COMPATIBILITY
SECURITY
TESTS
IMPLEMENTATION
```

Model-visible source tokens are limited to deterministic buckets and states:
protocol, event type, revision/age/update-gap/stale-age buckets, old/new
section presence, changed-section bitset, dependency delta state, dependency
edge delta bucket, line-change bucket, changed-section-count bucket,
counterpart state, and prior card hash. Raw proposal numbers, hashes, paths,
timestamps, dates, authors, URLs, numeric values, prices, returns, ranks, PnL,
and outcomes are audit-only and must not become model inputs.

## Cards, pairing, and archive schedules

Build one daily card per schedule for every `12:05:00Z` decision timestamp from
`2020-01-01` through `2024-03-31`. For each event effective day, availability is
exactly `12:00:00Z` after the schedule delay:

| schedule | delay | role |
|---|---:|---|
| `ARCHIVE_D2` | 2 calendar days | diagnostic replay/control only |
| `ARCHIVE_D7` | 7 calendar days | diagnostic replay/control only |
| `ARCHIVE_D30` | 30 calendar days | diagnostic replay/control only |
| `ARCHIVE_D90` | 90 calendar days | primary later economic clock under a separate frozen protocol |

Shorter schedules cannot rescue a D90 support failure. Cards must chain with
`prior_card_hash`, carry explicit no-new-event and stale-age state for both
protocols, and reject if the relation-card event bound is exceeded. The exact
bound unit is one emitted relation unit (paired or sentinel);
`len(relation_units) > 64` rejects before any truncation.

Pairing is deterministic:

1. if both protocol event sets are nonempty on the decision day, emit the
   complete Cartesian product with state `SAME_DAY_CARTESIAN`;
2. if exactly one side is nonempty, pair every anchor with the most recent
   opposite-protocol event available in the preceding 90 days, tie-broken by
   maximum event ID, with state `TRAILING_90D`;
3. if no opposite event exists, use `NO_COUNTERPART`; and
4. if both sides are empty, emit one `NO_ANCHOR` unit.

No semantic score, market state, model output, or external metadata may select
or drop a pair.

## Control metrics

Evaluate all seven controls in every `control × archive schedule × split` cell:

1. `protocol_label_swap`
2. `within_day_event_order_reverse`
3. `proposal_version_pair_cyclic_permutation`
4. `old_new_direction_reverse`
5. `section_label_cyclic_rotation`
6. `dependency_edge_direction_reverse`
7. `availability_plus_seven_days`

The comparison unit is the canonical local daily-card payload for one decision
day. Exclude card hashes, prior-card hashes, raw object IDs, control name, and
audit counters from the comparison payload. A day is changed only when the
transformed local payload SHA-256 differs from the baseline local payload
SHA-256. Each exact cell requires at least four eligible decision days and
`changed / eligible >= 0.10`. Cells are never pooled or weighted. A zero-eligible
cell rejects.

## Split and support metrics

Assign source-event support by `effective_day`; assign daily-card and relation
support by the `ARCHIVE_D90` decision timestamp. The frozen splits are:

| split | decision interval |
|---|---|
| train | `[2020-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` |
| test | `[2022-01-01T00:00:00Z, 2023-01-01T00:00:00Z)` |
| eval | `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)` |

Measure and gate events total, events per protocol, events per protocol per
source year, unique proposals, unique event days, active months, active
quarters, nonexcluded relation units, counterpart fraction, top proposal event
share, top event-day share, event-type diversity, section diversity, dependency
delta diversity, revision-bucket diversity, top changed-section share, and
non-quarantined events per protocol exactly as frozen in the preregistration.
For relation support, `NO_ANCHOR` daily sentinels are not relation units.
Non-quarantined `NO_COUNTERPART` units remain in the denominator but not the
counterpart numerator; `SAME_DAY_CARTESIAN` and `TRAILING_90D` units count in
both.

## Frozen gates

Run all thirteen gates in this exact order and require every gate to pass:

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

The first failed gate terminally rejects PSIM-D1 unchanged with:

```text
REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No repair, source drop, parser relaxation, threshold relaxation, control
removal, schedule removal, quarantine change, model swap, resampling, rerun, or
post-failure rebuild is allowed after source incidence opens.

## Forbidden access

The source-support implementation must never open BTC market data, funding data,
future returns, rewards, model rows, model outputs, trades, PnL, CAGR, strict
MDD, portfolio state, or any market/outcome artifact. It must not load a model
or build a reward/action/trade. The forbidden-access gate must prove all
forbidden counters are zero, including `pre_2020_proposal_blobs_opened` and
`post_2023_proposal_blobs_opened`.

Famous-proposal quarantine remains exact and source-accounting-only:

```text
Ethereum: 20, 721, 1559, 3675, 4337, 4844, 4895
Bitcoin: 32, 39, 44, 141, 340, 341, 342
```

Quarantined events may support source coverage and controls but may not enter
later model training, inference, reward, or economics.

## Source artifact publication

Terminal publication writes only these source artifacts, with canonical JSON or
deterministic gzip bytes and manifest hashes:

```text
results/protocol_specification_intent_maturity_source_support_2026-07-25.json
results/protocol_specification_intent_maturity_source_rejection_2026-07-25.json
data/protocol_specification_intent_maturity_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_source_controls_2026-07-25.json
```

Publication must be terminal and non-overwriting: if any target path already
exists before publication, gate 13 rejects. The result must include gate results
in frozen order, artifact SHA-256 values, independent replay fingerprints,
source-support metrics, control metrics, the access ledger, and the exact
terminal action. A rejection artifact must include the first failed gate, failure
message, metrics available before failure, and the same access ledger. Neither
artifact may include model, market, funding, outcome, trade, or portfolio data.

Before source access, the runner acquires an exclusive no-follow run lock.
Cooperating concurrent runs cannot pass this boundary. Publication stages all
pass bytes, hard-links them as one rollback-capable group, writes the result
within that group, and releases the lock only after durable completion. A
publication-time conflict or linking failure replaces gate 13 with a failure,
rolls back runner-owned pass links, and publishes the terminal rejection when
no conflicting foreign target prevents it.
