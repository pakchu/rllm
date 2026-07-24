# GitHub Advisory first-add source-axis decision — 2026-07-24

## Decision

Select:

```text
GHAD-GRFA-D1 — GitHub-reviewed Advisory first-add daily ledger
```

as the next independent, outcome-blind BTC RLLM source axis.

GHAD-GRFA-D1 asks only whether the first appearance of GitHub-reviewed
open-source vulnerability advisories can produce a causal, replayable,
text-rich daily security-stress ledger with enough support for a later
mechanism decision.

This decision authorizes one source-only builder and one source-support audit.
It does **not** authorize:

- a crypto-relevance term, package, ecosystem, vulnerability class, severity
  transform, semantic label, direction, action, hold, threshold, or position;
- a prompt, Gemma adapter, checkpoint, reward, RL policy, or portfolio weight;
- a BTC bar, price, return, funding, premium, open interest, liquidation, PnL,
  CAGR, MDD, existing-alpha outcome, or market clock; or
- a claim that software vulnerability disclosures predict BTC.

A separate committed mechanism decision must precede the first advisory
selection, semantic classification, market join, or model input.

## Why this axis is selected

The repository has already explored or retired broad families based on:

- BTC price action, funding, premium, open interest, liquidation, order flow,
  public trades, depth, options, and cross-venue timing;
- stablecoin, custody, bridge, chain, Lightning, and Bitcoin Core activity;
- CFTC, SEC, Federal Reserve, New York Fed, Treasury, BLS, EIA, OFR, and other
  macro or regulatory releases;
- GDELT, Wikimedia, BitMEX Trollbox, announcements, maintenance notices, BGP,
  Tor, and weather-derived attention or infrastructure events; and
- many deterministic, tree, Markov, HMM, BOCPD, CNN, LLM, and RL policy
  variants over those source families.

Repository-wide searches found no prior GitHub Advisory Database, GHSA, OSV
advisory-history, or open-source vulnerability-disclosure alpha source.

The prior GitHub-adjacent Bitcoin Core immutable-merge axis was retired at its
own source boundary:

- `docs/bitcoin-core-immutable-merge-surface-source-axis-decision-2026-07-22.md`
- `docs/bitcoin-core-immutable-merge-surface-source-rejection-2026-07-22.md`

GHAD-GRFA-D1 does not repair or relabel that source. Bitcoin Core merge
integration and the cross-ecosystem publication of reviewed vulnerability
advisories are different observables with different identities, clocks,
schemas, and live collectors.

An independent architecture review ranked this source above:

1. DOL weekly claims, retained only as a reserve source; and
2. a mixed Census/USDA macro-release ledger, rejected for this branch because
   of mixed-vintage complexity and effective sparsity.

The reason is not an assumed trading direction. GitHub-reviewed advisories are
dense, source-new, machine-readable, naturally textual, and better suited to a
later bounded LLM extractor than another well-known macro headline.

## Official source contract

### Repository

Official repository:

<https://github.com/github/advisory-database>

Official GitHub documentation states that advisory records are published as
individual JSON files in the Open Source Vulnerability format:

<https://docs.github.com/en/code-security/concepts/vulnerability-reporting-and-management/github-advisory-database>

The public repository describes the database and contribution process:

<https://github.com/github/advisory-database>

Only this Git remote is authorized:

```text
https://github.com/github/advisory-database.git
```

Only this source subtree is authorized:

```text
advisories/github-reviewed/**/GHSA-*.json
```

The following are excluded:

- `advisories/unreviewed/**`;
- repository issues, pull requests, discussions, reactions, stars, forks,
  contributors, and web-search ranking;
- mutable `github.com/advisories` HTML pages;
- NVD, npm, OSV.dev, mirrors, package registries, exploit feeds, download
  counts, repository popularity, and third-party enrichment; and
- any market or portfolio data.

### Frozen remote identity

A metadata-only `git ls-remote` and blobless shallow transport probe selected:

```text
default branch: refs/heads/main
commit:         40e5791b176b832cb09323d3962abe2fe3249e34
tree:           283dcf468588e3f9fd4a1d7a671df11527788dfc
parent:         0ab828c5a28c008f4c6f3344a8bb783484c41378
committer UTC:  2026-07-24T14:36:46Z
```

The source audit must fetch and resolve that exact commit and tree. A later
`main`, tag, branch, API result, or repository archive may not replace it.

GitHub's public repository metadata reported:

```text
private:        false
default_branch: main
created_at:     2022-02-11T22:59:38Z
license:        CC-BY-4.0
```

The metadata probe cloned with:

- `--filter=blob:none`;
- `--depth=1`;
- `--no-checkout`;
- `--single-branch`;
- no credentials; and
- no Git LFS checkout.

It resolved the exact frozen commit, reported a 41.18 MiB pack containing
351,358 non-working-tree objects, opened no advisory blob, performed no
checkout, and was deleted after the transport measurement.

## Frozen Git history semantics

### Default-branch entry, not arbitrary commit time

The causal source history is the exact **first-parent chain** ending at the
frozen commit.

For every first-parent transition:

1. compare the child tree only against its first parent;
2. disable rename and copy inference;
3. enumerate exact additions, modifications, deletions, and type changes under
   the authorized subtree;
4. derive the advisory identity from the canonical `GHSA-xxxx-xxxx-xxxx`
   filename and the OSV `id`; and
5. treat only the earliest first-parent addition of that identity as its
   source event.

An addition on a side branch does not establish availability before that
content enters the pinned default-branch first-parent history. A later path
move, re-addition, edit, withdrawal, or deletion is not a new first-add event.

The builder may not use:

- `git log --all`;
- author dates;
- side-branch commit dates;
- pull-request creation, review, merge, or issue times;
- current file modification time;
- current GitHub API ordering; or
- rename heuristics.

### Conservative causal clock

For first-parent commit `i`, define:

```text
ordered_commit_time_i =
    max(committer_time_i, ordered_commit_time_(i-1))
```

For an advisory first added at commit `i`, parse the initial blob's exact OSV
`published` timestamp and define:

```text
source_floor =
    max(ordered_commit_time_i, osv_published)
```

Historical availability is 12:00 UTC on the next UTC calendar day after
`source_floor`.

This deliberately avoids treating the advisory's `published` field or a
backdated Git commit as proof that the selected default-branch blob was already
public.

Live availability is the later of:

- the same historical floor;
- durable local receipt of the containing default-branch commit;
- exact commit/tree verification;
- initial-blob parsing and hashing; and
- append-only manifest commit.

No event is backdated after live receipt.

### Source window

The source-support window contains reviewed first-add identities satisfying
both:

```text
initial OSV published:
    [2022-02-11T22:59:38Z, 2026-01-01T00:00:00Z)

historical causal availability:
    [2022-02-12T12:00:00Z, 2026-01-01T00:00:00Z)
```

This excludes legacy advisories imported at repository creation and keeps 2026
market outcomes outside the later train/test/eval process. An advisory
published before 2026 but first added or causally available in 2026 is not a
source-support row.

Source-only schema and transport parity may inspect aggregate counts and hashes
for post-2025 additions reachable from the frozen commit, but may not publish
their text, packages, semantic labels, or candidate decisions.

## Frozen OSV source schema

Official OSV schema:

<https://ossf.github.io/osv-schema/>

The source builder may parse only JSON objects and these fields:

- `schema_version`;
- `id`;
- `modified`;
- `published`;
- `withdrawn`;
- `aliases`;
- `summary`;
- `details`;
- `severity`;
- `affected[].package.ecosystem`;
- `affected[].package.name`;
- `affected[].package.purl`;
- `affected[].ranges`;
- `affected[].versions`;
- `references`;
- `credits`;
- `database_specific`;
- `ecosystem_specific`.

At source-support stage, textual values are opened only to validate encoding,
type, length, and non-emptiness. They may not be searched, embedded,
classified, summarized, scored, prompted, printed, or used to select a
mechanism.

The source audit may publish:

- commit, tree, parent, blob, manifest, parser, and output hashes;
- first-parent commit and source-event counts;
- aggregate counts by year, month, UTC availability day, ecosystem, schema
  version, withdrawal state, and structural severity type;
- null, empty, duplicate, malformed, and mutation counts;
- text byte-length distributions without text;
- source transport size and disk use; and
- pass/reject gate booleans.

It may not publish:

- advisory summary or details;
- package names, purls, aliases, references, credits, or URLs;
- GHSA or CVE identities;
- semantic relevance or vulnerability categories;
- source-event dates at row level; or
- any market, candidate, action, model, or economic output.

## Exact identity and structure rules

Every selected initial blob must:

- be a regular Git blob reached from the exact pinned first-parent history;
- have an authorized ASCII path and `.json` suffix;
- have a basename exactly equal to `<GHSA-ID>.json`;
- decode as strict UTF-8 without BOM or replacement;
- parse as one JSON object with duplicate-key rejection;
- contain an `id` exactly equal to the basename GHSA identity;
- contain RFC3339 UTC `published` and `modified` timestamps;
- satisfy `modified >= published`;
- contain a nonempty `affected` list;
- contain at least one affected package with nonempty ecosystem and name;
- encode `summary` and `details`, when present, as UTF-8 strings without NUL
  code points;
- contain only finite JSON numbers; and
- reproduce its exact raw SHA-256 and canonical structural digest.

Duplicate GHSA identities, path/ID disagreement, conflicting initial blobs,
submodules, symlinks, Git LFS pointers, invalid object types, malformed JSON,
invalid UTF-8, or a missing required field reject the source.

JSON canonicalization is used only for structural replay. The exact raw blob
remains authoritative.

## Source-only support gates

All gates are frozen before an advisory body is opened.

### History integrity

- the exact remote, commit, tree, parent, and first-parent chain resolve;
- every first-parent transition is replayable;
- no graft, replace ref, shallow boundary, alternate object directory,
  submodule, worktree checkout, or local hook affects the result;
- the authorized subtree contains only regular JSON blobs;
- every selected identity has exactly one earliest first-parent addition; and
- a second replay from the sealed object database reproduces every aggregate
  and hash.

### Coverage

The 2022 partial year must contain at least 500 selected reviewed first-add
events.

Each of 2023, 2024, and 2025 must contain:

- at least 1,000 selected events;
- at least 200 unique availability days;
- at least 50 events in every calendar month; and
- no single availability day above 10% of that year's events.

Across the full source window:

- at least five affected-package ecosystems must appear;
- no single ecosystem may exceed 80% of selected events;
- at least 95% of non-withdrawn events must have both nonempty summary and
  details;
- every selected row must have exact first-add, publication, and availability
  clocks; and
- exact replay must produce zero unexplained additions, duplicates, or hash
  disagreements.

The source audit does not relax a failed threshold, quarantine a malformed
record, substitute a current version, or change the source window.

## Transport, disk, and execution boundary

The source audit must run in a fresh isolated process with:

- no credential helper, interactive prompt, cookie, API token, SSH remote,
  proxy, alternate remote, or browser session;
- `GIT_CONFIG_NOSYSTEM=1`;
- an empty explicit global Git config;
- disabled hooks;
- no replacement refs, alternates, grafts, submodules, smudge filters, or LFS
  fetch;
- protocol v2 and HTTPS to the exact remote;
- `--filter=blob:none` during the commit/tree fetch; and
- a two-phase materialization rule:
  1. derive exact path-first-add candidate blob object IDs only from
     first-parent additions under
     `advisories/github-reviewed/**/GHSA-*.json`; then
  2. retrieve only those candidate initial blobs and select source-window rows
     only after structural and causal-window validation.

Before network access and before every materialization phase:

- the boundary, builder, and tests must be committed and equal to `HEAD`;
- the worktree must be clean;
- filesystem use must be below 300 GiB;
- at least 8 GiB must remain free;
- the Git object store must remain below 8 GiB;
- candidate initial advisory blobs plus manifests must remain below 2 GiB; and
- an exclusive one-shot sentinel and append-only hash-chained manifest must be
  durable.

The source attempt is consumed when its sentinel is created. It is not retried,
resumed, repaired, widened, narrowed, or redirected after any network or
source-structure failure.

Ignored local storage may retain the Git object database and initial blobs.
Only the aggregate source report and immutable hashes may be committed.

## No-leak and no-outcome boundary

Before a passing source report is committed, the process may not open:

- BTC, ETH, equity, FX, commodity, rate, volatility, or any other market data;
- funding, premium, open interest, liquidation, order-flow, depth, return,
  target, PnL, CAGR, MDD, reward, checkpoint, or portfolio data;
- existing alpha incidence, direction, weight, or performance;
- package popularity, download counts, repository activity, exploit activity,
  or asset exposure;
- a crypto term list or advisory semantic label; or
- any LLM, embedding model, tokenizer, adapter, prompt, or RL policy.

The source audit process must fail if market/model modules, paths, environment
variables, or database connections are imported or opened.

## Later mechanism and RLLM boundary

A source pass authorizes exactly one later, separately committed mechanism
decision.

That later decision may define:

- an outcome-blind advisory relevance/support policy;
- deterministic structural facts derived from initial OSV blobs;
- a bounded single-model Gemma 4 extractor over initial source text;
- synthetic prompt-injection, entity-swap, timestamp-swap, negation,
  memorization, and label-balance controls;
- a minimum annual and unique-day relevance-support gate before market access;
- a causal daily aggregation policy; and
- train/test/eval and untouched-2026 market boundaries.

The LLM should be used for its comparative and deductive text reasoning, not
for arithmetic over raw price arrays. Deterministic code must compute clocks,
counts, rolling state, positions, rewards, costs, and risk.

No model may:

- create, delete, or retime source events;
- inspect a later advisory version when classifying its first-add event;
- see market outcomes during source relevance labeling or adapter selection;
- use eval or 2026 rewards for prompt, checkpoint, or hyperparameter choice;
- receive future bars or post-entry facts; or
- restore the discarded analyzer/trader two-model split.

## Pass, reject, and next action

`SOURCE_SUPPORT_PASS` requires every integrity, structure, coverage, transport,
disk, and no-leak gate.

Any source failure is `TERMINAL_REJECT` and permanently retires
GHAD-GRFA-D1. There is no parser repair, package-list fallback, API fallback,
current-snapshot substitution, or DOL/Census branch switch inside this
boundary.

On pass:

1. commit the aggregate source report;
2. commit one mechanism decision before semantic or market access;
3. run the outcome-blind relevance/support gate;
4. freeze the evaluator and market splits;
5. open train outcomes only; and
6. preserve 2026 as untouched final market evaluation.

On reject, return to source selection. DOL weekly claims remains a distinct
reserve axis and is not automatically authorized.
