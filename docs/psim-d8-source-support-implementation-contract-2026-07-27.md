# PSIM-D8 Source-Support Implementation Contract

Date: 2026-07-27 KST

Status: reviewed implementation candidate; official source execution remains
forbidden until a canonical direct-child execution seal is committed.

Scope: source representation only. Market, model, reward, outcome, trade, PnL,
return, CAGR, and strict-MDD access remain forbidden.

## Frozen predecessor

- PSIM-D7 is terminally rejected at source Gate 5. Its canonical result hash
  is
  `45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b`.
- The D7 official source runner, source root, and terminal artifact must not
  be repaired, reused, deleted, or rerun.
- D8 binds the committed D7 evaluator, tests, terminal rejection, D7 grammar
  probe, D6 evaluator, D6 terminal rejection, and inherited D1-D7 authority.
- Every D7 source behavior remains frozen except the one preregistered
  relation-card representation delta below.

## Exact authorized D8 delta

The only semantic source delta is
`PSIM_D8_LOGICAL_DAY_CARD_WITH_ORDERED_RELATION_SUBCARDS_V1`.

For each archive schedule and logical decision day:

1. D8 emits exactly one top-level logical `DailyCard`, preserving the D7
   logical-card denominator and card order.
2. The card retains the exact complete, ordered D7 `relation_units` roster in
   its source/audit payload.
3. The roster is partitioned into deterministic contiguous slices of at most
   64 relation units: `[64*k, min(64*(k+1), N))`.
4. A canonical manifest binds every slice's schedule, decision time, ordinal,
   range, count, payload hash, prior-subcard hash, and complete-roster hash.
5. The complete manifest is bound into the local-payload hash and inherited
   logical-card hash chain.
6. Empty rosters, gaps, overlaps, duplicates, reordered slices, cap changes,
   stale hashes, incomplete coverage, and manifest/card identity mismatches
   fail closed.
7. Full cards and manifests are audit-only. No model-visible subcard,
   subcard aggregation policy, model, reward, or economic evaluation is
   authorized by this implementation.

The frozen relation-subcard contract hash is
`c86aaf1e9975d62c88c45f89dc6943fef7e2ed8902ecc840ea9f569e09e1e0fb`.

## D7 equivalence boundary

The evaluator retains the D7 implementation for:

- official EIP/BIP repositories, sealed tips, clone and hydration behavior;
- 2020-01-01 through 2023-12-31 source interval;
- schedules, train/test/eval split boundaries, and quarantine roster;
- D6 lossless UTF-8 transport and D7 Bitcoin grammar overlay;
- event identity, pairing, grouping, explicit staleness, and card hash chain;
- all 13 source gates in their original order;
- all seven relation controls and unique logical-day denominators;
- typed failure publication and zero market/model/outcome access before a
  complete source pass.

Regression tests compare D7 and D8 cards built from the same synthetic event
roster and require byte-equivalent semantic payloads after removing only D8
manifest and consequent hash fields. Tests exercise exact 64, 65, and 72 unit
boundaries and require `[64]`, `[64, 1]`, and `[64, 8]` subcard counts without
changing the logical card count.

## Control and gate enforcement

- Every relation control rebuilds cards from its transformed event roster.
  It therefore rebuilds and validates the relation-subcard manifest instead
  of copying the baseline manifest.
- Gate 7 validates every complete D8 envelope and binds manifest validity to
  schedule coverage and explicit staleness.
- Pass artifacts retain one row per logical card and include the audit
  manifest. They do not authorize or construct a model-visible representation.
- Tamper tests cover missing manifests, complete-roster hashes, slice payload
  hashes, prior hashes, ranges, ordering, cardinality, and the frozen cap.

## Verification contract

The implementation requires all of the following with no failures, skips,
errors, expected failures, or unexpected passes:

- 602 inherited pre-rebase D1-D7 authority tests in a fresh detached
  worktree;
- 113 current D7-evaluator, D8-mechanism, D8-preregistration, and
  D8-evaluator tests;
- a synthetic D8 self-check that opens no official source, market, model, or
  outcome data.

The verification environment removes ambient pytest selectors and third-party
plugin autoloading and freezes locale, timezone, hash seed, and import path.

## Fresh-root and execution discipline

- The only D8 source root is `/tmp/psim-d8-source`.
- The only D8 sealed ref is `refs/psim-d8/sealed-tip`.
- The source root must be absent before the one authorized official attempt.
- This implementation commit does not authorize that attempt.
- A direct-child commit may add only the canonical D8 execution-seal JSON and
  its execution-seal test.
- The official source runner must validate that seal before creating or
  opening the source root.
- If any D8 source gate fails, or if another source semantic delta would be
  required, PSIM is retired. No D9 successor is authorized.
- A D8 source pass would not establish alpha. Any model-visible slice,
  aggregation, memorization test, market join, OOS evaluation, or economic
  statistic requires a separate preregistration after the source result is
  frozen.

No official PSIM-D8 source execution occurs in this implementation unit.
