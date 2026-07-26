# PSIM-D3 terminal source-support rejection

Date: 2026-07-26 KST

## Terminal decision

The sealed PSIM-D3 source run executed exactly once from:

```text
seal commit 4e931cbda848e6914e912bdd10eeb35c250dd821
implementation commit cf85aedaad0a0e2b15a440362d03702aad10175f
seal hash 1663fcc711bcde6d8e48a24434225957dfef5728f01ccb385051f96ccba3841b
seal SHA-256 e2d4a503ac90fa971c7229aed0885862e48529fe17866065c2257de00f3fda50
```

It terminally rejected at Gate 4:

```text
historical_blob_preamble_dependency_integrity
REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

The canonical terminal artifact is:

```text
results/protocol_specification_intent_maturity_d3_source_rejection_2026-07-25.json
SHA-256 a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802
result_hash b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c
```

PSIM-D3 is not repaired or rerun. No D3 model or economic stage is
authorized. This is a source-support rejection, not evidence that the
proposed specification-intent relation has or lacks economic alpha.

## One-shot timing and environment

The one-shot shell log was created at:

```text
2026-07-26T12:32:53.392335429+09:00
```

The rejection was atomically published at:

```text
2026-07-26T12:35:07.580350972+09:00
```

The observed wrapper duration, including the sealed 192-test revalidation,
was approximately 134.2 seconds. WSL disk use remained 292 GiB, below the
frozen 300 GiB limit. The four fresh source roots and Trace2 evidence occupied
approximately 28 MiB.

The Python process exited with code 2 only because the sealed terminal
decision was `reject`. The run lock was released.

## Gates that completed

### Gate 1: passed

All four fresh independent roots passed:

```text
/tmp/psim-d3-source/ethereum-a.git
/tmp/psim-d3-source/ethereum-b.git
/tmp/psim-d3-source/bitcoin-a.git
/tmp/psim-d3-source/bitcoin-b.git
```

Every Gate 1 check was true:

- exact official remote, branch, and frozen sealed tip;
- bare repository with no worktree or index;
- exact `HEAD`, branch, D3 sealed ref, and two-ref roster;
- no alternate, shared object, symlink, hard link, checkout, shallow state,
  or `git status`;
- `FETCH_HEAD` absent;
- `git fsck --no-dangling`; and
- disk below 300 GiB.

### Gate 2: passed

Complete first-parent replay was identical between replicas:

| protocol | commits | replica hash |
|---|---:|---|
| Ethereum | 6,958 | `c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9` |
| Bitcoin | 1,482 | `7e60f24b78aa863a2b317a7dc3a32b2af8e367c3d25f4a97012f4ddfd28d89d2` |

Effective days were monotone, first-parent continuity held, and traversal
used `refs/psim-d3/sealed-tip`.

### Gate 3: passed

Path/object incidence was identical and issue-free:

| protocol | retained 2020–2023 proposal groups | replica hash |
|---|---:|---|
| Ethereum | 4,985 | `a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c` |
| Bitcoin | 371 | `3f7a8e10bb5f9ba57bb0231b5cd54a613fb81e67830c1ec1d9781fe0d22b6a8b` |

Both replicas had no duplicate-tree-path, ambiguous-old/new, event-identity,
or source-interval issue.

## D3 transport result: succeeded

The D3 mechanism corrected the D2 transport failure for Ethereum replica
`a`. Its canonical hydration receipt proves:

```text
requested blobs             5,206
explicit fetch invocations      1
new packs                       1
new promisor markers            1
new loose objects               0
new object-store objects    5,206
maintenance children            0
post-read fetch children         0
refs unchanged               true
FETCH_HEAD absent             true
post-read snapshot unchanged true
```

The exact evidence includes:

```text
OID manifest SHA-256
8aa47dbe594df92a42ce87f6872f2bb3548f5370371f7668b26c80a47c53c944

before snapshot hash
7e986ff6557019d2d42925681cb6a45de737ba6f2da00e065c3496e0b6660efb

hydrated/post-read snapshot hash
34ff16dc38c729e5178a0fe7cfd377cd1d88b030c42f5d2f8c325bdfc987f96a

fetch Trace2 SHA-256
bcf1744d9a451eca796ef974c9e69c48dafbd531f91b6a139dac274a4e0fd74f

local-read Trace2 SHA-256
7fd83d7f643d3a1f21615922d0c58aa97d65490805a48d6017eefab35737c474

receipt hash
4c273231fb5a4f4675c35107b68d65d13ee16e97443deae6f6a95c451d7b2e3e
```

There was no per-object lazy fetch, automatic maintenance, stderr deadlock,
extra object, ref mutation, loose object, or post-read mutation. PSIM-D3
therefore solved the specific D2 batch-hydration transport defect.

## Gate 4 semantic parser failure

After successful local-only hydration, the frozen D1 parser opened 17 unique
proposal texts and rejected the 18th. The canonical terminal artifact records
`ValueError`; a post-terminal local no-lazy forensic replay isolated:

```text
error          PSIM blank line inside header
protocol       ethereum
proposal       2378
side           new
path           EIPS/eip-2378.md
commit         b788f38a216ca4cfea9d9de8ccfcf4cf658c8950
blob OID       ac34c07b91d6dffa14922951473f50dd587eb900
effective day  2020-01-29
raw SHA-256    a2fd3d87db7861f2b50739bf6c9015b968abc6fb6ffee7629492626034f41bb1
```

The historical EIP front matter contains a blank line immediately before its
closing `---` delimiter:

```text
created: 2019-11-13

---
```

The preregistered D1 EIP grammar rejects every blank line inside front matter.
The source-support contract requires parser success fraction `1.0`, forbids
row dropping and parser relaxation, and stops at the first failed gate.
Therefore the evaluator correctly rejected rather than deleting EIP-2378,
normalizing its header, or continuing to later replicas.

This was not a transport failure, missing object, source corruption, future
reference, or outcome-dependent decision. It exposed a preregistered parser
coverage assumption that the synthetic corpus had not represented.

## Forensic boundary

The root cause was isolated only after terminal publication, without invoking
`run` again:

- existing `ethereum-a.git` forensic residue only;
- exact `/usr/bin/git`;
- `GIT_NO_LAZY_FETCH=1`;
- zero network commands;
- no object, pack, promisor, ref, symbolic-HEAD, or `FETCH_HEAD` mutation;
- no market, model, reward, trade, PnL, CAGR, strict MDD, or outcomes.

The before and after object-store snapshots were equal. The forensic replay
did not alter the terminal artifact or salvage D3.

## Access boundary

At rejection:

```text
git_commands                 21,225
network_commands                 13
source_path_rows_opened      16,312
proposal_blobs_opened         5,206
proposal_text_rows_opened        17
daily_cards_built                 0
```

All forbidden fields remained zero:

- BTC market rows;
- funding rows;
- future returns;
- rewards;
- model loads and model outputs;
- trades;
- PnL;
- CAGR; and
- strict MDD.

Pre-2020 and post-2023 proposal-blob counters were also zero. No complete
event/card/control/pass artifact was published.

## Publication state

The rejection is the only D3 terminal artifact. The run lock and every pass
target are absent:

```text
results/protocol_specification_intent_maturity_d3_source_support_2026-07-25.json
data/protocol_specification_intent_maturity_d3_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_d3_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_d3_source_controls_2026-07-25.json
results/.psim_d3_source_support_run.lock
```

The source roots are retained only as one-shot forensic residue. They are not
authorized for reuse, repair, continuation, or a later candidate.

## Stop condition and next boundary

PSIM-D3 is terminally rejected unchanged. The following are forbidden:

- rerunning D3;
- changing D3's parser and continuing;
- dropping EIP-2378 or any other rejected proposal;
- normalizing historical source in place;
- relaxing parser success, source interval, support floors, controls, or
  splits;
- reusing any D1/D2/D3 source object or root; and
- starting a D3 model/economic stage.

A later candidate requires a new outcome-blind decision, identifier,
preregistration, fresh independent roots, synthetic malformed-header corpus,
and an explicit exact parser grammar. The minimal research question is
whether a deterministic historical front-matter grammar that permits blank
lines without permitting duplicate keys, ambiguous delimiters, or silent
normalization can support the complete frozen source interval. That would be
a new source mechanism, not a D3 repair.
