# BCIMS source-axis decision — 2026-07-22

## Decision

Select **BCIMS (Bitcoin Core Immutable Merge Surface)** as the next
outcome-blind BTC source axis. BCIMS observes integrations into the official
Bitcoin Core `master` first-parent history. Its source payload is the immutable
merge commit object, its two parent identities, and the path surface changed
relative to the first parent.

BCIMS is a source, not an alpha. This decision does **not** authorize a long or
short action, holding period, threshold, semantic class, reward, checkpoint,
portfolio weight, or economic claim. Those remain sealed until a separately
committed source protocol passes.

## Why this axis

The repository has already tested extensive price, funding, open-interest,
liquidation, order-flow, depth, options, macro, stablecoin, regulatory-text,
social-attention, and Bitcoin chain-state families. It has not used Bitcoin
Core's integration history as a trading source.

The source is also aligned with the intended role of an LLM. Merge messages
contain natural-language change descriptions, component prefixes, pull-request
descriptions, reviewer acknowledgements, and immutable changed-path context.
Any later model can reason over those tokens instead of being asked to perform
fragile arithmetic on raw market numbers. Numeric market outcomes remain
strictly outside this stage.

The broad hypothesis is only that the composition and cadence of publicly
integrated protocol, networking, wallet, mempool, mining, security, test, and
release work may contain information that is not represented by exchange or
chain-state telemetry. Source support does not establish that hypothesis or a
trading direction.

## Evidence boundary

No BTC bar, return, funding row, PnL, CAGR, MDD, prior-alpha outcome, model
prediction, or market clock was opened while selecting BCIMS.

A source-only parser probe was allowed exclusively on commits dated on or after
`2024-01-01`, outside the future BCIMS source-support interval. The probe used
Git `2.43.0`, a blob-filtered no-checkout clone of the official repository, and
sealed remote `master` at:

```text
bc49bd154a31b285c0f89be51767a424ac380924
```

The clone occupied 74 MiB. Its 2024-plus first-parent history contained 3,016
commits, all with exactly two parents. Exact immutable subjects routed 2,950 to
`Merge bitcoin/bitcoin#...` and 66 to `Merge bitcoin-core/gui#...`; there were
no other two-parent subject forms, no non-two-parent first-parent commits, and
no descending committer timestamps in that out-of-window probe. These counts
prove current transport and parser feasibility only. They are not alpha
support and reveal nothing about the frozen 2020–2023 interval.

## Official source and authority

The sole historical content authority is the official upstream Git repository:

- <https://github.com/bitcoin/bitcoin>
- <https://github.com/bitcoin/bitcoin.git>
- [Bitcoin Core contribution process](https://github.com/bitcoin/bitcoin/blob/master/CONTRIBUTING.md)
- [Git first-parent semantics](https://git-scm.com/docs/git-log#Documentation/git-log.txt---first-parent)
- [Git commit-object model](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)

Bitcoin Core's contribution guide states that patch proposals use pull
requests, that non-GUI work uses `bitcoin/bitcoin`, that GUI work uses
`bitcoin-core/gui`, and that the PR description is included in the merge commit
message. BCIMS therefore reads the Git commit object itself. It does not use a
currently edited GitHub PR title, body, label, milestone, issue state, reaction,
review state, contributor profile, or API search result as historical truth.

## Frozen source boundary

The later source protocol must preserve these boundaries:

1. Clone only `https://github.com/bitcoin/bitcoin.git`, branch `master`, with
   `--single-branch --filter=blob:none --no-checkout`.
2. Bind the build to an explicitly sealed reachable tip and record the remote
   symref, Git version, object format, fetch receipt time, and repository object
   integrity result.
3. Traverse only the sealed tip's first-parent chain, oldest to newest.
4. The historical source interval is `[2020-01-01, 2024-01-01)` after applying
   the causal-availability rule below. The 2024-plus probe is excluded.
5. A primary event must have exactly two parents and an exact first subject
   line matching `Merge bitcoin/bitcoin#[1-9][0-9]*: <nonempty title>`.
6. An exact `Merge bitcoin-core/gui#[1-9][0-9]*: <nonempty title>` event is a
   mandatory GUI comparator, not a primary event.
7. Any other first-parent commit form is retained for audit but cannot enter
   the primary or comparator strata. Duplicate PR numbers within a stratum,
   malformed UTF-8, missing parents, or object-hash failure rejects the source.
8. Retain the raw commit object bytes, commit/tree/parent hashes, normalized UTC
   author and committer times, the full immutable commit message, exact regex
   captures, and a NUL-safe path delta against parent one.
9. Compute the path delta with rename detection disabled. Path identity is raw
   Git path bytes; no current filesystem checkout, language detector, mutable
   repository metadata, or LLM may add or remove a source event.
10. Blob contents and market data remain unopened during source support. A
    later semantic protocol may explicitly authorize immutable diff text only
    after BCIMS passes.

## Causal availability

Git committer time is object content, not a server receipt log. BCIMS therefore
does not trade at that timestamp.

For a historical integration commit, define `commit_day_utc` from the normalized
committer timestamp. Its earliest usable time is **12:00 UTC on the second
calendar day after `commit_day_utc`**. If first-parent committer days ever move
backward, the effective day is the running maximum of the current and all prior
first-parent committer days before applying the two-day delay. This conservative
daily clock prevents timestamp reversal and avoids pretending that an object
creation time is an exact publication time.

For live operation, availability is the later of that historical floor and the
durable local fetch, object verification, extraction, hash, and manifest-commit
time. An observed force-push, unreachable sealed ancestor, changed object hash,
or remote default-branch change halts BCIMS; history is never silently rebuilt.

## Source-only gates to freeze next

Before opening 2020–2023 incidence, the next commit must define executable
gates for:

- complete first-parent traversal and object/hash integrity;
- exact primary/comparator subject parsing;
- minimum annual, quarterly, and unique-day support;
- concentration across immutable top-level path surfaces;
- duplicate PR and timestamp-order handling;
- deterministic replay from a sealed tip; and
- live parity under the same two-day availability floor.

Failure is `REJECT_NO_REPAIR`. A failed threshold may not be relaxed after
incidence is read, and no subset, direction flip, path alias, LLM relabel, or
market-conditioned repair may be introduced on this branch.

## Deferred semantic and economic stages

Only a source pass may authorize a separately committed semantic protocol. That
protocol may compare a small local LLM against deterministic path/component
baselines while requiring quote-or-path evidence and abstention. It must freeze
all labels and model artifacts before any return is opened.

Only a semantic pass may authorize a separately committed economic evaluator.
The evaluator must use disjoint train/test/eval periods, realistic costs,
absolute return, full-calendar CAGR, strict MDD, trade count, and untouched OOS
reporting. None of those choices is made here.
