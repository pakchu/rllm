# PSIM-D3 source-support preregistration

Date: 2026-07-26 KST

## Decision

PSIM-D3 is preregistered as a newly named source-only successor to the
terminally rejected PSIM-D2 candidate. It preserves D2's repository shape,
sealed source identities, complete replay, parser, archive, card, control,
support, split, quarantine, and forbidden-access contracts.

The only substantive change is Gate 4 transport. D2 allowed interactive
`git cat-file` to fault missing blobs in one at a time. D3 first derives the
complete retained in-window blob-OID manifest without opening blob content,
then submits that manifest to one explicit `git fetch --stdin` invocation per
replica with automatic maintenance disabled. All later blob decoding is
local-only and lazy fetching is disabled.

This preregistration does not reopen or rerun D2. It does not authorize
official EIP/BIP source execution yet, a model, market or funding data, an
outcome, reward, trade, PnL, CAGR, strict MDD, leverage, portfolio, or live
trading. The next unit may only implement and seal a synthetic-only D3
source-support evaluator.

The successor decision is:

```text
docs/post-psim-d2-alpha-mechanism-audit-2026-07-25.md
SHA-256 7fecb77f93bdf0f78cbdb45afbf866d3c726944627ed49bdf56ef69f0535ba4a
commit 126f7f1354eff90f30d5a6b3d60bd6641268b03b
```

The D2 terminal authority is:

```text
results/protocol_specification_intent_maturity_d2_source_rejection_2026-07-25.json
SHA-256 461ea699ada0d6873422c537e63f5fcff3bca56a436caae9aeff4bb74761ca24
result_hash b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c
first failure Gate 4 historical_blob_preamble_dependency_integrity
commit 0e98ba563fb38012f7cd5c65cc1f4ca3800f0483
```

PSIM-D2 remains terminally rejected:

```text
REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

The D2 source roots are forensic residue. D3 may not read, reuse, copy,
alternate, hard-link, repair, continue, or cache any D2 object.

## Machine-readable contract

The canonical D3 preregistration is:

```text
results/protocol_specification_intent_maturity_d3_preregistration_2026-07-25.json
SHA-256 332743f25d5be45ce4d022c67758051c01297f4cc18ccdf2138be75b5ef159ab
manifest_hash d87358780df573bde11a317bf2e56f0ce044b3fc2fad3a28ef6e154d64023d86
authorized_delta_hash a092091bc5f9316a90c828b2701526697a5ff29a3ca1ac82580acc30eada3b9e
batch_hydration_contract_hash 6701b544f055c5eaa5e1c22dc4963f975514b9e5833845ee92c8384bdec9cf39
git_binary_binding_hash 70aa4a393c76b2d310f4cc91367533a47a93537fa06ccaa2dcb5dc6100397ebf
```

The builder and tests are:

```text
training/preregister_protocol_specification_intent_maturity_d3.py
SHA-256 8eedf77cecacc77327ff6f1c0da399f8e53e89b5f807b28fcbcd52975e42cd76

tests/test_preregister_protocol_specification_intent_maturity_d3.py
SHA-256 883d5a06fdaf33d1738798f1285e4c22e07af6cdfa5b4255683eff7486bc147e
```

The builder loads and canonical-hash validates:

1. the exact D2 preregistration;
2. the exact terminal D2 Gate 4 rejection;
3. the D3 selection decision; and
4. the synthetic D3 transport probe.

It removes only the outer D2 manifest and inheritance-proof envelopes,
applies the explicit D3 delta, recursively compares every contract leaf, and
fails unless both the changed-path roster and complete before/after value
delta have their frozen identities. The newly added batch and Git-binary
contracts also have independent canonical hashes.

## Synthetic transport authority

No official source was used to select the transport. The canonical local
probe is:

```text
results/protocol_specification_intent_maturity_d3_transport_probe_2026-07-25.json
SHA-256 4a815145a1f2ab9c6c61d599cf0aaf2218172e9f71251e95ce7178c1f3be13b7
result_hash 0df158cddd9b663b2daca14e01bcaa5c2e64b7f5d976720282120585bc41c63a
commit 126f7f1354eff90f30d5a6b3d60bd6641268b03b
```

On the bound binary, six missing synthetic blob OIDs sent to one explicit
fetch produced one promisor pack containing exactly those six blobs, no
extra object, no ref or `FETCH_HEAD` mutation, no maintenance child, and no
post-hydration fetch. The same six OIDs sent through buffered `cat-file`
produced six promisor packs.

The observed one-pack result is not a D3 source criterion. A server may
produce multiple physical packs in one fetch. D3 accepts one or more new
promisor packs only when their complete union and the complete local object
store delta equal the requested blob set exactly.

## Official Git evidence and binary binding

The transport decision is grounded in:

- <https://git-scm.com/docs/partial-clone/2.43.0.html>
- <https://git-scm.com/docs/git-fetch/2.43.0.html>
- <https://git-scm.com/docs/git-cat-file/2.43.0.html>
- <https://github.com/git/git/blob/v2.43.0/promisor-remote.c#L17-L45>

Git 2.43 documents that a fully spelled hexadecimal object ID can be a
positive refspec source, `--stdin` reads refspecs one per line,
`--no-write-fetch-head` suppresses `FETCH_HEAD`, and
`--no-auto-maintenance` suppresses the otherwise default post-fetch
maintenance.

The exact local binary is frozen:

```text
/usr/bin/git
git version 2.43.0
SHA-256 2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

PATH lookup is forbidden. The evaluator must execute `/usr/bin/git`
directly and repeat the synthetic `GIT_NO_LAZY_FETCH=1` semantic check before
official source access. This is necessary because the local binary includes
the environment-variable behavior while the stock v2.43 CLI does not expose
the later `git --no-lazy-fetch` option.

## Frozen Gate 4 batch hydration

After Gate 3 completes independently in each replica, D3 must derive:

```text
sorted unique union of every non-null old_blob_oid and new_blob_oid
from retained 2020-2023 ProposalGroup rows
```

The manifest is lowercase full 40-hex SHA-1, one per LF-terminated line,
with a final LF. Every requested OID must be absent from the initial
`blob:none` object roster.

Exactly once per replica, invoke:

```text
/usr/bin/git -C <fresh-bare-root> \
  -c fetch.negotiationAlgorithm=noop \
  fetch origin \
  --no-tags \
  --no-write-fetch-head \
  --recurse-submodules=no \
  --filter=blob:none \
  --no-auto-maintenance \
  --stdin
```

The complete manifest is the command's only stdin. Python must use
`subprocess.run`/`communicate()` semantics so stdout and stderr are drained
while the finite command runs. The timeout is 1,800 seconds.

Before and after hydration, inventory:

- the complete local object roster and object types;
- every pack;
- every `.promisor` marker;
- every loose object;
- every ref; and
- `FETCH_HEAD` presence.

The source gate fails unless:

- at least one new pack exists;
- every new pack has a matching promisor marker;
- no new loose object exists;
- the union of all new pack objects equals the requested OID set;
- the complete local object-store delta equals the same set;
- every new object is type `blob`;
- the ref roster is unchanged;
- `FETCH_HEAD` remains absent;
- no maintenance child starts; and
- every Trace2 `child_start` has an unambiguous nonempty argv.

After hydration, blob decoding must run with:

```text
GIT_NO_LAZY_FETCH=1
```

`cat-file` is local decoding only. Missing objects, fetch children, or any
change to packs, promisor markers, loose objects, refs, object types, or
`FETCH_HEAD` immediately fails Gate 4.

The following transports are forbidden:

- interactive or buffered `cat-file` lazy hydration;
- per-object fetch;
- retry;
- fallback lazy fetch;
- full clone;
- `git fetch --refetch`;
- checkout; and
- D1/D2 source-object reuse.

## Fresh roots and sealed refs

Every official source root must be absent before a one-shot run:

```text
/tmp/psim-d3-source/ethereum-a.git
/tmp/psim-d3-source/ethereum-b.git
/tmp/psim-d3-source/bitcoin-a.git
/tmp/psim-d3-source/bitcoin-b.git
```

Clone arguments and bare-repository assertions remain D2-identical. The only
namespace change is:

```text
refs/psim-d3/sealed-tip
```

All traversal starts from that sealed ref, never a moving branch. Shared
objects, alternates, copied caches, hard links, checkout, index, linked
worktrees, tags, and remote-tracking refs remain forbidden. The existing
300 GiB WSL disk guard remains unchanged.

## Mechanically inherited D2 contract

Outside the frozen D3 delta, canonical contract paths remain D2-identical:

- official Ethereum/Bitcoin remotes, branches, and sealed commit tips;
- source interval `[2020-01-01, 2024-01-01)`;
- card interval through `2024-04-01`;
- complete first-parent causal traversal;
- SHA-1 commit/blob recomputation;
- running-maximum UTC committer-day clock;
- no pre-window event warm-up;
- first old blob only as `PRE_WINDOW_BASELINE`;
- exact path, event, preamble, dependency, section, and bucket grammars;
- strict parser rejection without repair or row dropping;
- `ARCHIVE_D2`, `D7`, `D30`, and primary `D90`;
- one daily relation card at `12:05Z`;
- deterministic Cartesian/trailing-90-day pairing;
- famous-proposal quarantine;
- seven exact relation controls;
- per-cell control-sensitivity floor;
- train/test/eval source-support floors; and
- zero model, market, funding, outcome, reward, trade, PnL, CAGR, and strict
  MDD access.

No parser, source row, source interval, support threshold, control, split,
quarantine, schedule, model vocabulary, or later economic criterion changed.

## Frozen gates and terminal action

The exact thirteen D2 gate names and order remain:

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

The first failed D3 gate retires D3 unchanged:

```text
REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No source repair, retry, source drop, parser change, threshold change,
schedule change, control change, quarantine change, rerun, or model/economic
work is allowed after the first official D3 source attempt.

## Verification and next boundary

The combined D1, D2, D3 transport-probe, and D3 preregistration battery
passed:

```text
187 passed
```

The D3 preregistration battery proves:

- exact decision, D2 preregistration, D2 terminal, and probe bindings;
- exhaustive path- and value-level D2 delta equality;
- parser/split/support/control inheritance;
- exact one-fetch command and four-replica limit;
- multiple-pack union handling without a fixed pack count;
- exact object-store, promisor, loose-object, ref, and `FETCH_HEAD` boundary;
- exact Git binary binding;
- fail-closed authority, delta, transport, split, conflict, and symlink
  cases;
- canonical and idempotent output; and
- absence of network, Git execution, official source, market, model, or
  trading imports/calls during preregistration.

All forbidden preregistration counters remain zero. Official PSIM-D3 source
incidence remains closed until the evaluator, its synthetic adversarial
tests, independent review, and exact direct-child execution seal are
committed.
