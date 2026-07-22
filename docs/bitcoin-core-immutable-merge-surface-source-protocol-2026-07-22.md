# BCIMS source protocol — 2026-07-22

## Status

**Frozen before 2020–2023 source incidence and before every market outcome.**

- source axis: `docs/bitcoin-core-immutable-merge-surface-source-axis-decision-2026-07-22.md`
- executable protocol: `training/preregister_bitcoin_core_immutable_merge_surface.py`
- manifest: `results/bitcoin_core_immutable_merge_surface_source_protocol_2026-07-22.json`
- manifest hash: `d2edac0aa3e8ccdf7b17b7bd0d8d7e60a053624ac3626904956cab714bac6cd3`
- manifest file SHA-256: `991d83099d3910e8d278ec79f36f8fbfdf55270f7a05f2cb00546b6f3f819f39`
- outcomes opened: `false`
- historical source incidence opened: `false`
- semantic model opened: `false`

## Membership

The sole source is the first-parent history reachable from a sealed tip of the
official `https://github.com/bitcoin/bitcoin.git` `master` branch.

```text
bc49bd154a31b285c0f89be51767a424ac380924
```

- `Merge bitcoin/bitcoin#N: title` with exactly two parents is primary.
- `Merge bitcoin-core/gui#N: title` with exactly two parents is the mandatory
  GUI comparator.
- every other first-parent form is audit-only;
- duplicate PR numbers within a stratum are fatal; and
- an LLM cannot add, delete, route, or retime an event.

The build retains raw commit objects and a NUL-safe, no-rename path delta against
parent one. Mutable GitHub PR metadata and blob contents are excluded from this
stage.

## Causal clock

The historical floor is 12:00 UTC on the second calendar day after the running
maximum UTC committer day. Live availability is the later of that floor and
durable local fetch, object verification, extraction, hashing, and manifest
commit. Git committer time is not claimed to be an exact server receipt time.

## Frozen source gates

Primary Core events must satisfy all of:

- at least 2,400 total events;
- at least 500 events and 180 unique availability days in every year;
- at least 100 events in every quarter;
- no calendar month above 12% of primary events;
- at least six top-level changed-path surfaces in every year; and
- no fractionally attributed top-level surface above 70%.

The GUI comparator must contain at least 30 events, at least five per year, and
at least 25 unique availability days. Audit-only first-parent forms may not
exceed 5% overall or 10% in any year. Object, traversal, hash, path, parent,
duplicate, disk, replay, or support failure is `REJECT_NO_REPAIR`.

## Deferred LLM/RLLM stage

A source pass may authorize one separately frozen local-LLM semantic stage over
the immutable merge message and exact path evidence. It must include
path-only, keyword-only, and cadence-only baselines; support abstention; freeze
the model and labels before returns open; and perform any adapter or RLLM
selection on train only. The prior analyzer/trader two-model split remains
forbidden.

Only a later semantic pass can authorize market evaluation with disjoint
train/test/eval windows, full-calendar CAGR, absolute return, strict MDD,
realistic costs, trade count, and untouched OOS reporting.
