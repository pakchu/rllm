# PSIM-D1 source-support terminal rejection

Date: 2026-07-25

## Terminal result

PSIM-D1 was executed exactly once from the committed execution seal and is
terminally rejected unchanged.

| field | value |
|---|---|
| implementation commit | `80b656994f17548a7a599a548e23e9f1cd01302d` |
| seal commit | `d537ef0` |
| execution-seal hash | `c26397920fa1137845f5dea56eab72cb1a8d4ead401e7ee3e249c5c1e39aa506` |
| first failed gate | `1 — sealed_git_identity_and_object_integrity` |
| decision | `reject` |
| terminal action | `REJECT_PSIM_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES` |
| result hash | `5815f7473410c7d75aabea8b6a97cfb7f963b1c6d29f8efa22f0a0a64d33655d` |
| rejection artifact SHA-256 | `9b0b2354c6edbcfe627527bf4370a4eb0c1e6c1bcb76843f843d9028b16e6494` |

No repair, parser change, threshold change, source drop, provider swap, or
rerun was performed.

## Opened boundary

The run stopped inside the first replica preparation:

- Git commands: `8`
- network-capable Git commands: `3`
- disk used at start: `291 GiB` under the frozen `300 GiB` guard
- source class opened: `git_remote_identity`
- commit metadata opened: no
- proposal path incidence opened: no
- proposal blobs opened: no
- daily cards built: `0`
- model, market, funding, outcome, trade, PnL, CAGR, and strict-MDD counters:
  all zero

No pass artifact was written. The exclusive run lock was removed and only the
terminal rejection artifact exists.

## Forensic root cause

Read-only inspection of the already-created
`/tmp/psim-d1-source/ethereum-a` root showed:

- origin URL matched `https://github.com/ethereum/EIPs.git`;
- object format was `sha1`;
- sealed tip `5e82ef62895121027a6c5f0c23276e1b2bed3071` resolved to a commit;
- `.git/objects/info/alternates` was absent;
- `git fsck --no-dangling` exited `0` with no output; and
- the worktree contained no checked-out files.

The failure came from the frozen cleanliness assertion. A non-bare
`git clone --no-checkout` has an empty index while `HEAD` points at the sealed
tree. Consequently:

```text
git status --porcelain=v1
```

reported `1,348` staged deletions such as:

```text
D  .gitattributes
D  .github/CODEOWNERS
...
```

This is the expected representation of that no-checkout clone shape, not
external worktree contamination. The evaluator incorrectly required empty
porcelain output and therefore raised `RuntimeError` after the eighth Git
command.

## Interpretation

This rejection is implementation-feasibility evidence, not evidence against
the PSIM economic hypothesis. It provides no source-support, model, alpha,
profitability, or live-trading result because proposal incidence and all
outcomes remained sealed.

Any successor must be a newly named and newly preregistered candidate. It may
replace the invalid cleanliness predicate with an object-database invariant
appropriate for `--no-checkout`, but PSIM-D1 itself remains terminal and cannot
be rerun.
