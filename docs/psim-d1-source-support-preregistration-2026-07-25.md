# PSIM-D1 source-support preregistration

Date: 2026-07-25

## Decision

PSIM-D1 is preregistered for a source-only replay of immutable historical
specification versions in the official Ethereum EIPs and Bitcoin BIPs Git
repositories.

This artifact does not authorize cloning, reading a 2020–2023 proposal blob,
building a relation card, loading an LLM, opening BTC/funding data, or
evaluating a trade. The next unit may only implement and seal a synthetic-only
source-support evaluator.

The source-axis decision is:

```text
docs/post-gipr-d1-alpha-mechanism-audit-2026-07-25.md
SHA256 816fbb19c4ff9a841f75f75555e568f401e804b4aded258779ef4bce14ebaf04
commit 6ebb43406f7197e2afb2e2fa5cb39b0a2cba2826
```

The machine-readable preregistration is:

```text
results/protocol_specification_intent_maturity_preregistration_2026-07-25.json
SHA256 bd4053574fe6285c34356baaa080e215f08bbf8142e9c0c968bffbdccb2dc736
manifest_hash bdf49fb396779599eb329a407685435c05217f132ea856f9bb743914b5afbe81
```

The builder and tests are:

```text
training/preregister_protocol_specification_intent_maturity.py
SHA256 982f14cff8c903c9ad528018ab996f053179cc054626bfaa8af7ab63405f858b

tests/test_preregister_protocol_specification_intent_maturity.py
SHA256 d7d0a00ff40bae4d6ac7a0969d186b82d5e3d59b119fe8d076f37ccac7e0ac46
```

## Frozen Git identities

| protocol | exact remote | branch/symref | sealed tip | object format |
|---|---|---|---|---|
| Ethereum | `https://github.com/ethereum/EIPs.git` | `master` / `refs/remotes/origin/master` | `5e82ef62895121027a6c5f0c23276e1b2bed3071` | `sha1` |
| Bitcoin | `https://github.com/bitcoin/bips.git` | `master` / `refs/remotes/origin/master` | `b289d016b99c81527623c10e995e0318f744ebf3` | `sha1` |

The evaluator must use a blob-filtered, single-branch, no-checkout clone,
verify the exact remote/symref/tip/object format, pass
`git fsck --no-dangling`, and traverse the sealed tip's complete first-parent
chain oldest to newest. Inclusion never uses a commit subject, current
checkout, PR, issue, label, review, or current rendered page.

The exact primary-document paths are:

```text
^EIPS/eip-([1-9][0-9]*)\.md$
^bip-([0-9]{4})\.(mediawiki|md)$
```

Tree deltas are NUL-safe, byte-exact, case-sensitive, and computed against
parent one with rename detection disabled. Matching old/new blobs are grouped
by protocol and proposal number into exactly `CREATE`, `UPDATE`, or `DELETE`.
Ambiguous duplicate paths, number disagreement, or a metadata parse failure
rejects the source.

## Frozen source, reset, and availability

The source event interval is:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

No pre-2020 blob warm-up is allowed. Revision count resets to zero,
pre-first-event age is `PRE_WINDOW`, protocol stale age is `NO_EVENT_YET`, and
the old blob of the first in-window update is only a
`PRE_WINDOW_BASELINE`.

Committer time is explicitly not treated as a publication receipt. Four
running-maximum committer-day schedules are mandatory:

```text
ARCHIVE_D2
ARCHIVE_D7
ARCHIVE_D30
ARCHIVE_D90
```

Each becomes available at `12:00Z` after its exact calendar-day delay, and its
daily card is cut at `12:05Z`. `ARCHIVE_D90` is the only schedule that may
later support an economic claim. The shorter schedules are diagnostics and
cannot rescue a D90 failure.

## Frozen parser and representation

Every historical blob must pass strict UTF-8, NUL rejection, newline/NFC
normalization, byte/line/header/section/dependency bounds, exact number
matching, and strict preamble parsing.

`PSIM_PREAMBLE_STATE_MACHINE_V1` is executable in the preregistration builder
through `normalize_blob_bytes`, `parse_eip_preamble`, `parse_bip_preamble`,
and `parse_dependency_ids`. It fixes exact fences/tags, full-line comment
handling, first-colon field separation, opaque quoted/list values,
case-folded duplicate rejection, continuation behavior, empty-value policy,
BIP leading-blank limit, and dependency grammar. Synthetic acceptance and
rejection cases lock these rules before incidence.

EIP front matter and BIP RFC-822-style preambles are parsed as historical
tokens. Today's EIP/BIP status vocabularies are not projected backward, and
declared status is not model visible. Dependency lists reject malformed,
duplicate, and self references.

All bucket edges are fixed in the JSON manifest for:

- in-window revision count;
- in-window age;
- update gap;
- stale age;
- line-change count;
- changed-section count; and
- dependency-edge delta count.

Raw numeric values are audit-only and cannot reach the later model.

## Frozen cross-protocol relation cards

For each decision day:

1. if both protocol event sets are nonempty, use the complete Cartesian
   product;
2. if only one side is nonempty, pair each anchor with the most recent
   opposite-protocol event in the preceding 90 days;
3. otherwise use `NO_COUNTERPART`; and
4. when both sides are empty, emit one `NO_ANCHOR`.

No semantic score or market state may select a pair. Over-limit cards reject
instead of silently truncating.

A later single model, under a separately frozen semantic/economic protocol,
may emit one relation token and one target in one call. The relation vocabulary
is:

```text
CONVERGENT_INTENT
COMPLEMENTARY_INTENT
TECHNICAL_TENSION
INDEPENDENT_INTENT
INSUFFICIENT_EVIDENCE
ABSTAIN
```

The target vocabulary remains:

```text
TARGET_LONG
TARGET_FLAT
TARGET_SHORT
```

There is no analyzer/trader pair or free-form rationale.

## Frozen pretrained-future leakage controls

The fixed famous-proposal quarantine is:

```text
EIP: 20, 721, 1559, 3675, 4337, 4844, 4895
BIP: 32, 39, 44, 141, 340, 341, 342
```

These events remain in source-support accounting but can never enter model
training, inference, reward, or economics.

Before any market outcome, both the base and final model must pass the exact
eight-way identity-recovery challenge frozen in the manifest. It uses at most
16 lowest-hash events per protocol/source-year, seven same-stratum decoys,
forced choice, and one-sided exact binomial tests against chance `1/8`.
Bonferroni-adjusted `p < 0.01 / 3` terminally rejects PSIM-D1. Resampling,
redaction repair, proposal removal, or model swapping after the result is
forbidden.

## Frozen gates and terminal action

All thirteen source gates execute in manifest order:

1. sealed Git identity and object integrity;
2. first-parent traversal and causal clock;
3. path/object grammar and unique proposal tree;
4. historical blob/preamble/dependency integrity;
5. split, annual, quarterly, and unique-day support;
6. event/section/dependency/revision vocabulary diversity;
7. daily-card coverage and explicit staleness;
8. independent replay and canonical manifest identity;
9. future-append invariance;
10. seven relation-control sensitivities;
11. pairing/reset/quarantine/four-schedule identity;
12. zero forbidden access; and
13. terminal publication.

Control sensitivity is evaluated separately for every
`control × archive schedule × split` cell. The comparison unit is the
canonical local daily-card payload with chain hashes, raw object IDs, control
name, and audit counters excluded. A day changes only when its canonical
payload SHA-256 changes. Each cell requires at least four eligible days and a
changed fraction of at least `0.10`; cells are never pooled or weighted, and
zero eligibility rejects the candidate. The manifest fixes the exact
transformation and eligibility rule for all seven controls.

The first failure retires PSIM-D1 unchanged:

```text
REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No protocol, split, proposal type, path extension, parser rule, bucket,
support minimum, control, quarantine row, or schedule may be removed or
relaxed after incidence.

## Verification

The dedicated preregistration test battery passed:

```text
51 passed
```

All forbidden counters are zero:

- network calls: `0`;
- Git commands: `0`;
- source incidence rows: `0`;
- proposal blobs/text/dependencies/cards: `0`;
- BTC/funding/outcome/model/trade/PnL/CAGR/MDD: `0`.

Historical PSIM-D1 incidence remains closed until a synthetic-only evaluator,
its adversarial tests, and its execution seal are committed.
