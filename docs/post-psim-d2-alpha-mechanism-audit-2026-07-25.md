# Post-PSIM-D2 audit and PSIM-D3 transport decision

Date: 2026-07-26 KST

## Scope

This document selects a newly named source-transport successor after the
terminal PSIM-D2 source-support rejection. It does not reopen, repair, resume,
or rerun PSIM-D2. It makes no source-support, model, alpha, profitability,
portfolio, leverage, or live-trading claim.

The only question answered here is whether a fresh candidate can hydrate the
already identified in-window proposal blobs in one explicit Git fetch session,
instead of allowing `git cat-file` to fault them in one at a time.

## PSIM-D2 terminal boundary

PSIM-D2 is terminally rejected at commit:

```text
0e98ba563fb38012f7cd5c65cc1f4ca3800f0483
```

The canonical rejection remains:

| artifact | identity |
|---|---|
| `results/protocol_specification_intent_maturity_d2_source_rejection_2026-07-25.json` | SHA-256 `461ea699ada0d6873422c537e63f5fcff3bca56a436caae9aeff4bb74761ca24` |
| terminal result | `b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c` |
| execution seal | `b6a101b2d6f41b70ac789ed243b8315589c109c4247d81e14c08d42c5aae0f27` |

Its terminal action is unchanged:

```text
REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

D2 passed repository identity, complete first-parent replay, and in-window
proposal-path incidence. It then failed Gate 4 before semantic parsing. The
inherited interactive `git cat-file --batch` loop requested each missing blob
and waited for its response before submitting the next OID. That produced 212
separate promisor packs in the first Ethereum replica, crossed Git's automatic
maintenance threshold, and deadlocked when `git gc --auto` filled a stderr
pipe that the parent planned to drain only after the object loop.

No market row, model, outcome, reward, trade, PnL, CAGR, or strict MDD was
opened. The four D2 source roots remain forensic residue and may not be read,
reused, copied, alternated, repaired, or cached by a successor.

## Official Git 2.43 evidence

The bound local version is:

```text
git version 2.43.0
/usr/bin/git
SHA-256 2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

Git's 2.43 partial-clone design states that dynamic object fetching is slow
because it normally fetches objects one at a time. It also states that a
missing-object request may send the hashes of all requested objects in one
`git fetch` subprocess and receive a packfile:

- <https://git-scm.com/docs/partial-clone/2.43.0.html>

Git 2.43 documents both pieces needed for a supported porcelain invocation:

1. `git fetch --stdin` reads one refspec per line; and
2. a positive refspec source may be a fully spelled hexadecimal object name.

It also documents that `--no-write-fetch-head` suppresses `FETCH_HEAD` and
that `--no-auto-maintenance` suppresses the otherwise default post-fetch
`git maintenance run --auto`:

- <https://git-scm.com/docs/git-fetch/2.43.0.html>

The exact v2.43 implementation of `promisor_remote_get_direct()` already uses
one child process with this shape and writes every requested OID to its stdin:

- <https://github.com/git/git/blob/v2.43.0/promisor-remote.c#L17-L45>

```text
git -c fetch.negotiationAlgorithm=noop fetch origin
    --no-tags
    --no-write-fetch-head
    --recurse-submodules=no
    --filter=blob:none
    --stdin
```

The successor adds the documented `--no-auto-maintenance` switch. It does not
change the selected OIDs, parser, source interval, archive schedule, support
floors, controls, or economic hypothesis.

`git cat-file --batch-command --buffer` is not an alternative hydration
transport. Its documentation says `--buffer` changes output flushing and
stdio buffering; it does not promise to aggregate missing-object network
wants:

- <https://git-scm.com/docs/git-cat-file/2.43.0.html>

The v2.43 implementation still resolves each batch object separately:

- <https://github.com/git/git/blob/v2.43.0/builtin/cat-file.c#L452-L501>
- <https://github.com/git/git/blob/v2.43.0/object-file.c#L1622-L1628>

### `GIT_NO_LAZY_FETCH` version note

The local `/usr/bin/git` accepts `GIT_NO_LAZY_FETCH=1` and emits
`lazy fetching disabled` for a missing promisor object without starting a
fetch child. However, the v2.43 documentation does not expose the later
`git --no-lazy-fetch` CLI option. Therefore PSIM-D3 does not infer this
capability from the version string. It binds the exact local binary SHA-256
and requires a fresh synthetic semantic probe before any official source
access. Current Git documentation describes the environment variable:

- <https://git-scm.com/docs/git>

This environment variable is used only after explicit hydration, to fail
closed if any requested object is still missing. It is never used as the
hydration mechanism.

## Synthetic-only transport evidence

The deterministic probe is:

```text
training/probe_protocol_specification_intent_maturity_d3_transport.py
```

Its canonical output is:

```text
results/protocol_specification_intent_maturity_d3_transport_probe_2026-07-25.json
result_hash 0df158cddd9b663b2daca14e01bcaa5c2e64b7f5d976720282120585bc41c63a
SHA-256 4a815145a1f2ab9c6c61d599cf0aaf2218172e9f71251e95ce7178c1f3be13b7
```

The probe creates only local synthetic repositories and never opens official
EIP/BIP source, market data, a model, or outcomes.

Its local `file://` origin explicitly enables both
`uploadpack.allowFilter=true` and `uploadpack.allowAnySHA1InWant=true`.
This proves the client-side batching and fail-closed inventory contract; it
does not claim that every remote server accepts arbitrary object wants.
PSIM-D2's terminal evidence separately established that the two frozen GitHub
remotes accepted the same blob wants one at a time. D3 must still fail closed
if either remote declines the single batched request.

### Selected transport

Six known missing blob OIDs were sorted, deduplicated, and submitted to one
explicit `git fetch --stdin` process:

| assertion | observed |
|---|---:|
| fetch invocations | 1 |
| requested blobs | 6 |
| new packs | 1 |
| new promisor markers | 1 |
| objects in the new pack | exactly the 6 requested blobs |
| nonrequested blob transferred | no |
| ref roster changed | no |
| `FETCH_HEAD` created | no |
| maintenance child started | no |
| post-hydration fetch child | no |
| post-hydration pack change | no |

The one-pack observation is synthetic evidence, not a general Git contract.
PSIM-D3's source gate is bound to one explicit fetch invocation per replica,
not to the server returning exactly one physical pack. Every new pack, loose
object, and promisor marker must instead be inventoried, and the complete new
object set must equal the requested blob set exactly.

The subprocess uses `subprocess.run(..., stdout=PIPE, stderr=PIPE)`, which
drains both streams through `communicate()`. It does not repeat D2's
write-one/read-one protocol or defer stderr consumption until an unbounded
interactive loop finishes.

### Rejected transport control

The same six OIDs sent as buffered `cat-file` commands produced:

```text
requested blobs       6
new promisor packs    6
```

This confirms on the bound binary that `--buffer` changes stdout buffering
but still faults in one object per network fetch. It is forbidden as a D3
hydration transport.

## Selected successor

The selected candidate is:

```text
PSIM-D3 — Protocol Specification Intent-Maturity relation RLLM,
targeted batch-hydration bare replay
```

Its source/economic hypothesis remains exactly D1/D2's:

> Relative maturity, dependency direction, and changed technical intent across
> causally archived EIP and BIP revision streams may form weak, orthogonal
> evidence that a later constrained single-model policy can combine.

The selection authorizes only a new preregistration and synthetic evaluator
work. It does not authorize official source execution yet.

## Frozen transport delta for preregistration

PSIM-D3 must use four fresh roots:

```text
/tmp/psim-d3-source/ethereum-a.git
/tmp/psim-d3-source/ethereum-b.git
/tmp/psim-d3-source/bitcoin-a.git
/tmp/psim-d3-source/bitcoin-b.git
```

The clone, sealed tips, bare-repository shape, traversal, causal clock, and
path-incidence rules remain D2-identical. After Gate 3 has identified the
in-window `ProposalGroup` rows, Gate 4 must:

1. derive the sorted, unique union of each replica's non-null old/new blob
   OIDs without opening blob content;
2. prove those OIDs are absent from the initial `blob:none` object roster;
3. write the complete OID manifest to one stdin stream;
4. invoke the exact batch-fetch command once for that replica, adding
   `--no-auto-maintenance`;
5. consume stdout and stderr concurrently through `subprocess.run`;
6. prove the ref roster is unchanged and `FETCH_HEAD` is absent;
7. inventory every new pack, promisor marker, and loose object;
8. prove the complete new object set equals the requested OID set and every
   object type is `blob`;
9. prove no maintenance child ran;
10. read requested blobs only with `GIT_NO_LAZY_FETCH=1`;
11. prove the read created no fetch child and changed no pack, loose-object,
    ref, or `FETCH_HEAD` roster; and
12. continue through the mechanically inherited D2 parser and Gates 5–13.

One explicit batch fetch is allowed per replica. Any retry, fallback lazy
fetch, missing OID, extra transferred object, unexpected object type, ref
mutation, maintenance child, trace ambiguity, or post-hydration object-store
mutation fails Gate 4 and terminally rejects D3 before semantic parsing.

The execution environment must scrub inherited `GIT_*` variables, disable
system/global Git configuration, bind the exact Git binary, and preserve the
existing 300 GiB disk guard. Trace files may contain process evidence only;
they may not become a source, model, or outcome side channel.

## Inherited contract and forbidden changes

Outside that transport delta, PSIM-D3 must mechanically preserve D2:

- exact official remotes, branches, and sealed tips;
- complete root-to-tip first-parent replay;
- SHA-1 recomputation and running-maximum causal day;
- source interval `[2020-01-01, 2024-01-01)`;
- exact proposal path, preamble, dependency, section, and event grammar;
- no source repair, row drop, rename detection, or parser fallback;
- D2/D7/D30/D90 archive schedules and D90 primary decisions;
- daily relation cards, pairing, quarantine, and seven controls;
- exact support floors, train/test/eval boundaries, and append invariance;
- zero model, market, funding, outcome, reward, trade, PnL, CAGR, and strict
  MDD access during source support; and
- first-failure terminal rejection with no rerun.

PSIM-D2 roots and objects are forbidden inputs. Full clone or `--refetch` is
also forbidden because either could hydrate proposal blobs outside the frozen
2020–2023 incidence boundary.

## Next boundary

Before any official source access, PSIM-D3 must:

1. commit a canonical preregistration hash-bound to this decision, the D2
   terminal record, and the synthetic probe;
2. mechanically prove the D2 contract changed only at authorized transport,
   identity, path, and execution-binding fields;
3. implement synthetic tests for multiple packs, extra objects, missing
   objects, lazy-fetch attempts, maintenance attempts, ref mutations, and
   stderr-heavy subprocesses;
4. receive independent adversarial review;
5. commit the evaluator and tests together;
6. create an exact direct-child execution seal; and
7. run the sealed D3 official source gate exactly once.

Only a clean source-support pass could authorize a separately preregistered
model/economic stage. It would still not itself establish alpha or
profitability.
