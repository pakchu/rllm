# BCIMS source rejection — 2026-07-22

## Decision

**`REJECT_NO_REPAIR`.** The exact frozen Bitcoin Core Immutable Merge Surface
(`BCIMS`) policy is retired before any semantic model, market clock, or outcome
is opened.

The source-only run completed against official Bitcoin Core `master` history at
sealed tip:

```text
bc49bd154a31b285c0f89be51767a424ac380924
```

It extracted 5,550 causally delayed first-parent rows for 2020–2023:

| Stratum | Rows |
|---|---:|
| exact `Merge bitcoin/bitcoin#N: ...` primary | 3,436 |
| exact `Merge bitcoin-core/gui#N: ...` comparator | 255 |
| audit-only | 1,859 |

## Frozen failure

The exact parser was frozen from an excluded 2024-plus probe. Historical
Bitcoin Core used a different immutable merge-subject convention:

```text
Merge #17741: build: Included test_bitcoin-qt in msvc build
```

Audit-only rows consisted of 1,854 two-parent `Merge #N: ...` subjects and five
one-parent direct commits. This produced:

- overall audit-only fraction: **33.50%**, above the frozen 5% maximum;
- annual audit-only fraction: **98.07% / 27.74% / 0.14% / 0.00%** for
  2020–2023, versus a 10% maximum in every year;
- exact primary events by year: **0 / 1,008 / 1,314 / 1,114**, versus at least
  500 in every year; and
- consequent failures for 2020 quarterly, unique-day, and changed-surface
  support.

The full failed-gate battery is sealed in
`results/bitcoin_core_immutable_merge_surface_source_support_2026-07-22.json`.

## Extraction integrity

The rejection is not caused by transport or nondeterminism:

- two independent extraction passes produced identical 5,550 rows;
- canonical JSONL SHA-256:
  `7cd4b2c9a86889466be63a2ac3413dfa700b307bc26b763feab4c4f095db897a`;
- ordered event-ID SHA-256:
  `f11eb9d7c03e7631964c08bc403369f2c0d7bade8a70e198ee45e072aab6f998`;
- ordered raw commit/path evidence SHA-256:
  `1499688f201a26c35f628c42fbb86ebcdf82abcc719fdfe4a6d373d6bdaa13c6`;
- the blobless clone contained zero local blob objects before fetch, after
  fetch, and after both extraction passes; and
- no mutable GitHub PR metadata or post-seal remote HEAD identity entered the
  source or support artifacts.

The machine-readable rejection is
`results/bitcoin_core_immutable_merge_surface_source_rejection_2026-07-22.json`,
with result hash
`38650591438b3a7e2dd83de10f086facbe5375ef805160a98786b557d888523c`.

## Why this cannot be repaired

Adding `Merge #N: ...` now would be a source-definition change after seeing its
historical incidence. The same applies to year-specific aliases, dropping
2020, lowering the audit thresholds, or semantically routing the audit-only
rows. Those may be reasonable for a new clean-room candidate, but they are not
BCIMS as frozen.

The retained rows are rejection evidence only. They may not be used for an
LLM, RLLM reward, signal, direction, holding-period search, or performance
test.

## Closed boundaries

No BTC price, funding, return, PnL, CAGR, strict MDD, prior-alpha clock,
semantic label, LLM checkpoint, or portfolio result was opened. A successor
must use a genuinely different observable and mechanism under a new
outcome-blind preregistration.
