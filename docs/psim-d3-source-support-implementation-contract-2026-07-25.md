# PSIM-D3 source-support implementation contract

Date: 2026-07-25

## Scope

This document binds:

```text
training/build_protocol_specification_intent_maturity_d3_source_support.py
```

PSIM-D3 is source-support-only. It makes no model, alpha, profitability,
portfolio, leverage, or live-trading claim. Market data, models, rewards,
trades, PnL, CAGR, strict MDD, and outcomes remain forbidden.

PSIM-D2 is terminally rejected and is not repaired or rerun. PSIM-D3 keeps
the sealed PSIM-D1 parser, event, card, split, support, control, and Gates
2–13 semantics. Its only source-semantic change from D2 is the preregistered
Gate 4 transport:

1. derive the complete retained proposal-blob OID manifest after Gate 3;
2. hydrate it with one explicit batch fetch per fresh replica; and
3. decode the now-local blobs with lazy fetching disabled.

No official EIP/BIP source may be opened until this implementation, its
tests, this contract, and an exact direct-child execution seal are committed
and validated.

## Frozen authority

| authority | commit | SHA-256 / manifest |
|---|---|---|
| D3 decision and transport selection | `126f7f1354eff90f30d5a6b3d60bd6641268b03b` | `7fecb77f93bdf0f78cbdb45afbf866d3c726944627ed49bdf56ef69f0535ba4a` |
| D3 synthetic transport probe | `126f7f1354eff90f30d5a6b3d60bd6641268b03b` | JSON `4a815145a1f2ab9c6c61d599cf0aaf2218172e9f71251e95ce7178c1f3be13b7`; result `0df158cddd9b663b2daca14e01bcaa5c2e64b7f5d976720282120585bc41c63a` |
| D3 preregistration unit | `1760d5945f0c8adc90ea667a21cbf6201eb5567e` | JSON `332743f25d5be45ce4d022c67758051c01297f4cc18ccdf2138be75b5ef159ab`; manifest `d87358780df573bde11a317bf2e56f0ce044b3fc2fad3a28ef6e154d64023d86` |
| D3 preregistration producer | `1760d5945f0c8adc90ea667a21cbf6201eb5567e` | `8eedf77cecacc77327ff6f1c0da399f8e53e89b5f807b28fcbcd52975e42cd76` |
| D3 preregistration document | `1760d5945f0c8adc90ea667a21cbf6201eb5567e` | `66f5f7083428dcb7836afc52ced72ff3225da837d39dae080281bb775ed5008f` |
| D2 terminal rejection | `0e98ba563fb38012f7cd5c65cc1f4ca3800f0483` | JSON `461ea699ada0d6873422c537e63f5fcff3bca56a436caae9aeff4bb74761ca24`; result `b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c` |
| sealed D1 core runner | `80b656994f17548a7a599a548e23e9f1cd01302d` | `414e83256b3ea489a9e1cd0995f6061e5fab550cd12c795ef7e88eff8998d9fb` |
| sealed D1 core tests | `80b656994f17548a7a599a548e23e9f1cd01302d` | `343aa1a72cfbca23d9756988ced042b5c61a6e8fc5a21a0b6d18e45870e906e9` |

The D3 authorized-delta hash is:

```text
a092091bc5f9316a90c828b2701526697a5ff29a3ca1ac82580acc30eada3b9e
```

The batch-hydration and Git-binary contract hashes are:

```text
6701b544f055c5eaa5e1c22dc4963f975514b9e5833845ee92c8384bdec9cf39
70aa4a393c76b2d310f4cc91367533a47a93537fa06ccaa2dcb5dc6100397ebf
```

## Exact Git binary and local-command boundary

Every Git subprocess in the D3 evaluator executes:

```text
/usr/bin/git
git version 2.43.0
SHA-256 2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

PATH lookup is forbidden. Runtime authority validation repeats the path,
version, and binary SHA-256 checks before source access. All inherited
`GIT_*` variables are removed; system and global configuration are disabled.
Every non-network Git command additionally receives:

```text
GIT_NO_LAZY_FETCH=1
```

The committed synthetic probe and the seal-target probe tests repeat the
local binary's no-lazy semantic check. This binds the local behavior without
claiming that stock upstream Git 2.43 exposes the later
`git --no-lazy-fetch` command-line option.

## Fresh roots and Gate 1

The four roots must not exist before the one official attempt:

```text
/tmp/psim-d3-source/ethereum-a.git
/tmp/psim-d3-source/ethereum-b.git
/tmp/psim-d3-source/bitcoin-a.git
/tmp/psim-d3-source/bitcoin-b.git
```

Clone arguments remain D2-identical:

```text
/usr/bin/git clone \
  --bare \
  --filter=blob:none \
  --single-branch \
  --branch master \
  --no-tags \
  <remote> <fresh-root>
```

The sealed-tip fetch uses `--no-write-fetch-head` so Gate 4 begins with no
`FETCH_HEAD`. This changes no selected commit, tree, ref, parser, event, or
gate semantic. Gate 1 still requires the exact official remote and branch,
frozen sealed tip, exact two-ref roster, bare/no-worktree shape, no index or
alternate, no shared hard link or symlink, no checkout or `git status`,
`git fsck --no-dangling`, and disk use at or below 300 GiB.

Traversal begins only from:

```text
refs/psim-d3/sealed-tip
```

Gate 2 commit reads and Gate 3 `diff-tree`/`ls-tree` reads are local-only
through the exact binary. D3 does not call the D1 `git` PATH wrappers.

## Gate 4 OID derivation

For each replica independently, after Gate 3 has passed, derive:

```text
sorted unique union of every non-null old_blob_oid and new_blob_oid
from retained [2020-01-01, 2024-01-01) ProposalGroup rows
```

The manifest is nonempty lowercase full 40-hex SHA-1, one per
LF-terminated line, including a final LF. The complete local object roster
is inventoried with lazy fetching disabled. Every requested OID must be
absent before hydration.

## One explicit hydration command

Exactly once per replica, execute:

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

The complete manifest is the command's only stdin. The implementation uses
finite `subprocess.run`/`communicate()` semantics, drains stdout and stderr,
and applies the frozen 1,800-second timeout. There is no retry, per-object
fetch, lazy-hydration fallback, full clone, `--refetch`, checkout, or D1/D2
object reuse.

Before and after the fetch, the evaluator inventories:

- the complete local object roster and type map;
- `.pack` roster;
- `.promisor` roster;
- loose-object roster;
- complete ref name/OID roster; and
- `FETCH_HEAD` absence.

Gate 4 fails unless:

- at least one new pack exists;
- every new pack has the matching `.promisor` marker;
- no new loose object exists;
- every preexisting object remains unchanged;
- the union of all new pack objects equals the requested OID set;
- the complete local object-store delta equals the same OID set;
- every new object is type `blob`;
- refs are unchanged;
- `FETCH_HEAD` remains absent; and
- Trace2 starts no maintenance child and contains no malformed or ambiguous
  `child_start.argv`.

The physical number of new packs is deliberately not fixed.

## Local-only decode and D1 semantic identity

After hydration, one finite:

```text
/usr/bin/git -C <fresh-bare-root> cat-file --batch
```

decodes the sorted manifest with `GIT_NO_LAZY_FETCH=1`. The byte parser is
length-delimited and rejects missing, reordered, mistyped, truncated, or
trailing output. A second Trace2 stream must contain no fetch or maintenance
child. The complete object, pack, promisor, loose-object, ref, object-type,
and `FETCH_HEAD` snapshot must be byte-equivalent before and after decoding.

Only the raw-object acquisition loop differs from D1. Feature parsing and
event construction call the exact sealed D1 pure helpers and preserve the
exact D1 order, revision state, baseline role, dependency delta, section
delta, intent text, quarantine, and availability-clock semantics. A real
local synthetic partial-clone test compares D3 event rows against D1 event
rows from the same Git objects and requires equality.

Each replica emits a canonical hashed hydration receipt containing the OID
manifest hash, before/after/post-read snapshot hashes, trace hashes, fetch
count, pack/promisor/loose-object deltas, child-process counts, and final
invariance flags. A failed hydration emits a hashed forensic failure receipt
with its last completed stage. Official pass requires four successful
receipts and exactly four total batch fetches.

## Inherited source-support semantics

Gates 5–13 and their prerequisites remain the sealed D1 implementation:

- exact EIP/BIP path and historical preamble grammar;
- SHA-1 commit and blob recomputation;
- complete first-parent continuity and causal running-maximum day;
- no pre-window warm-up and explicit `PRE_WINDOW_BASELINE`;
- strict UTF-8, dependency, section, diff, and bucket rules;
- D2/D7/D30/D90 archive clocks;
- deterministic daily relation cards and pairing;
- famous-proposal quarantine;
- seven exact relation controls;
- split, support, vocabulary, replay, future-append, control, and publication
  gates; and
- zero market/model/reward/trade/PnL/CAGR/strict-MDD access.

Any parser error, source difference, transport-boundary violation, gate
failure, or publication failure terminates D3 unchanged:

```text
REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

There is no repair, source drop, parser relaxation, threshold relaxation,
provider swap, retry, or D3 rerun.

## Execution seal topology

The runner, evaluator tests, and this implementation contract must share one
clean implementation commit. Seal creation reruns:

- the D3 synthetic self-check;
- D1 preregistration and sealed core evaluator tests;
- D2 preregistration and evaluator tests;
- D3 transport-probe tests;
- D3 preregistration tests; and
- D3 evaluator tests.

The seal JSON and its dedicated seal test must be the only paths in the exact
direct-child commit. The official one-shot may execute only while that seal
commit is the exact current `HEAD`.

## Synthetic evidence before implementation commit

The D3 self-check produced:

```text
stdout SHA-256 b8d4d6fa1e39c4ec8d2fb42cc4a1f333bbe05467bb0f1d6b4643914ca1909088
manifest_hash d641559c614f3d3b32fdb4f41c75c47816271dbd45b832eff855566382e258b5
failed []
network_calls 0
git_commands 0
source_event_rows_opened 0
official_source_opened false
outcomes_opened false
```

The combined D1/D2/D3 synthetic battery passed:

```text
192 passed
```

The battery includes exact-binary invocation, fresh bare acquisition, sealed
ref traversal, one-fetch partial-clone hydration, D1/D3 event equivalence,
multiple-promisor-pack acceptance, extra/missing/wrong-type object rejection,
promisor/loose/ref/`FETCH_HEAD` boundary rejection, Trace2 ambiguity and
child classification, post-read mutation rejection, length-delimited
`cat-file` parsing, publication rollback, and zero-outcome assertions.

No official EIP/BIP source, market data, model, or outcome was opened while
creating this implementation unit.

## Publication paths

```text
results/protocol_specification_intent_maturity_d3_source_support_2026-07-25.json
results/protocol_specification_intent_maturity_d3_source_rejection_2026-07-25.json
data/protocol_specification_intent_maturity_d3_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_d3_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_d3_source_controls_2026-07-25.json
```

## Upstream references

- [Git partial clone 2.43.0](https://git-scm.com/docs/partial-clone/2.43.0.html)
- [git-fetch 2.43.0](https://git-scm.com/docs/git-fetch/2.43.0.html)
- [git-cat-file 2.43.0](https://git-scm.com/docs/git-cat-file/2.43.0.html)
- [Git v2.43 promisor remote implementation](https://github.com/git/git/blob/v2.43.0/promisor-remote.c#L17-L45)
- [Current Git environment documentation](https://git-scm.com/docs/git)
