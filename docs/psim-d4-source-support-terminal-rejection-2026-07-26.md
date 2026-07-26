# PSIM-D4 terminal source-support rejection

Date: 2026-07-26 KST

## Terminal decision

The sealed PSIM-D4 source run executed exactly once from:

```text
seal commit 0482b34fa47d9e1decf7bf3707deecd71d3ce1c1
implementation commit 2d3216d5a144ba8eb694270301231850f0e015ca
seal hash 097ad1112607f1f4e5b47ada4abfc11700d4532dff56a50afc243f4c597867da
seal SHA-256 66a63c7c06fc1f19d85106ccaee04c1f2e384bf69f9a2cc5a9907d78c565b88a
```

It terminally rejected at Gate 4:

```text
historical_blob_preamble_dependency_integrity
REJECT_PSIM_D4_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

The canonical terminal artifact is:

```text
results/protocol_specification_intent_maturity_d4_source_rejection_2026-07-26.json
SHA-256 4d947075c0f54c5cd09c732710da0502c87d89fa52029fe81367dd3f27ab2aaf
result_hash 8563ef3ace444896295d7076cd0f839e8f62f89899e312d711f9768f5cbf84aa
```

PSIM-D4 is not repaired or rerun. No D4 model or economic stage is
authorized. This is a source-support rejection, not evidence that the
specification-intent relation has or lacks economic alpha.

## One-shot timing and environment

The first fresh-root files were created at:

```text
2026-07-26T14:02:20.076546960+09:00
```

The rejection was atomically published at:

```text
2026-07-26T14:03:29.758613069+09:00
```

The observed wrapper duration, including sealed 289-test revalidation, was
approximately 102.9 seconds. WSL disk use remained 292 GiB and Gate 1
recorded 291 GiB at its start, both below the frozen 300 GiB limit. The four
fresh roots and Trace2 evidence occupy approximately 28 MiB.

The Python process exited with code 2 only because the sealed terminal
decision was `reject`. The run lock was released.

## Gates that completed

### Gate 1: passed

All four fresh independent roots passed:

```text
/tmp/psim-d4-source/ethereum-a.git
/tmp/psim-d4-source/ethereum-b.git
/tmp/psim-d4-source/bitcoin-a.git
/tmp/psim-d4-source/bitcoin-b.git
```

Every Gate 1 check was true:

- exact official remote, branch, and frozen sealed tip;
- bare repository with no worktree or index;
- exact `HEAD`, branch, D4 sealed ref, and two-ref roster;
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
used `refs/psim-d4/sealed-tip`.

### Gate 3: passed

Path/object incidence was identical and issue-free:

| protocol | retained 2020–2023 proposal groups | replica hash |
|---|---:|---|
| Ethereum | 4,985 | `a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c` |
| Bitcoin | 371 | `3f7a8e10bb5f9ba57bb0231b5cd54a613fb81e67830c1ec1d9781fe0d22b6a8b` |

Both replicas had no duplicate-tree-path, ambiguous-old/new, event-identity,
or source-interval issue.

## D3 transport inheritance: succeeded

The inherited one-fetch D3 transport again succeeded for Ethereum replica
`a`. Its canonical D4 hydration receipt proves:

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
931b4bd69d8aefef0fee0c9665d2706100651792283ba8024a9c889b0e6a3790

hydrated/post-read snapshot hash
fd0ac6636ab7a954e46deb82188f9963f135b1b92152785c6f50205995766a2a

fetch Trace2 SHA-256
dfcb966d237f402694e68316bec9a3bc32cd3a93147a5d818eb8cea048a652a5

local-read Trace2 SHA-256
addfd252ca90e7768cc6e4f1ec8338157ec44216b9da1f8a3dfb0573977bda81

receipt hash
7b6d000b06029648e2ba253aab2885f2bd015d49ca6553f499cfa85ef7c957f2
```

There was no per-object lazy fetch, automatic maintenance, stderr deadlock,
extra object, ref mutation, loose object, or post-read mutation.

The Trace2 sibling directory inside the D4 root retained the inherited name
`.psim-d3-traces`. This was a byte-equal D3 transport implementation detail,
not predecessor object reuse, and did not cause the rejection. It should be
renamed explicitly by any future independently preregistered evaluator rather
than silently carried forward.

## Gate 4 semantic parser failure

D4 resolved D3's normalized-empty separator failure. It successfully parsed
44 unique proposal texts, compared with D3's 17, then rejected the next
historical EIP:

```text
error          PSIM duplicate normalized header key
protocol       ethereum
proposal       2544
side           new
path           EIPS/eip-2544.md
commit         bd912a490d97da82a73313facf4458bbaa0dab2b
blob OID       5ea7653f919002a0e83744b5ecdf624ccd9b4f31
effective day  2020-03-03
raw SHA-256    2198de4ecea78342143e366f3496d9d84d9b327c50bcce327061a1a2f84b94a0
```

The normalized front matter contains the same key twice:

```text
status: Draft
...
category: ERC
status: Draft
```

The D4 contract deliberately preserved D1's duplicate-key rejection. It
authorized only normalized-empty separators and prohibited duplicate-key
relaxation, row dropping, or post-run parser repair. The evaluator therefore
correctly rejected rather than choosing the first or last `status`, deleting
one line, or continuing to later blobs.

This was not D3's blank-line failure, a transport failure, missing object,
source corruption, future reference, or outcome-dependent decision. It
demonstrates that historical EIP front matter contains more than one
compatibility class absent from the original synthetic corpus.

## Post-terminal forensic boundary

The root cause was isolated only after terminal publication, without invoking
`run` again:

- existing `ethereum-a.git` forensic residue only;
- exact D4 parser and exact `/usr/bin/git`;
- `GIT_NO_LAZY_FETCH=1`;
- zero network commands;
- 6,958 commit rows, 4,985 groups, and 5,206 already hydrated blobs;
- no object, pack, promisor, ref, symbolic-HEAD, or `FETCH_HEAD` mutation; and
- no market, model, reward, trade, PnL, CAGR, strict MDD, or outcomes.

The complete object-store hashes were identical:

```text
before fd0ac6636ab7a954e46deb82188f9963f135b1b92152785c6f50205995766a2a
after  fd0ac6636ab7a954e46deb82188f9963f135b1b92152785c6f50205995766a2a
```

The local forensic replay used 9,635 non-network Git commands and left the
terminal artifact SHA-256 unchanged. It did not salvage or continue D4.

## Access boundary

At rejection:

```text
git_commands                 21,225
network_commands                 13
source_path_rows_opened      16,312
proposal_blobs_opened         5,206
proposal_text_rows_opened        44
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

The rejection is the only D4 terminal artifact. The run lock and every pass
target are absent:

```text
results/protocol_specification_intent_maturity_d4_source_support_2026-07-26.json
data/protocol_specification_intent_maturity_d4_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_d4_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_d4_source_controls_2026-07-26.json
results/.psim_d4_source_support_run.lock
```

The source roots are retained only as one-shot forensic residue. They are not
authorized for reuse, repair, continuation, or a later candidate.

## Stop condition and next boundary

PSIM-D4 is terminally rejected unchanged. The following are forbidden:

- rerunning D4;
- changing D4's parser and continuing;
- selecting the first or last duplicate field after observing the failure;
- dropping EIP-2544 or any other rejected proposal;
- normalizing historical source in place;
- relaxing parser success, source interval, support floors, controls, or
  splits;
- reusing any D1/D2/D3/D4 source object or root as a later candidate's
  source; and
- starting a D4 model or economic stage.

The sequential “fix the first historical syntax exception and rerun” process
is now rejected as a research design. Before another source candidate is
preregistered, an outcome-blind, read-only historical grammar census should
classify all front-matter compatibility classes in the already terminal
forensic corpus. That census may inform a new parser contract but may not
provide source objects to the new candidate.

Any later candidate requires a new identifier, decision, preregistration,
synthetic adversarial corpus, independently reviewed parser semantics, fresh
roots, and its own one-shot seal. A duplicate-key policy must be justified by
the historical publishing semantics or excluded explicitly; it cannot be
chosen because it advances toward a model or profitability result.
