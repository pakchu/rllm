# FLNSR-2016 strict evaluator freeze — 2026-07-23

## Decision

The strict evaluator is **frozen**. No FLNSR Bitcoin return, funding cash flow,
CAGR, or drawdown was opened while producing this artifact. The next permitted
operation is the one-time 2020–2022 Stage1 evaluation.

## Frozen identities

- Source-support commit: `cd731fe3de60f3eea4461f41ad4cb8c6175c606f`
- Preregistration manifest: `4277c6f7fb5a9c075492dd901f55d7b2d4e8b39dbab1e0c40010f996d80dc00c`
- Source-support manifest: `377fd7afd68e99e9c3cd93340505b80fc1ab7263ae2cf8ef117daa304227d895`
- Evaluator source SHA256: `fad04d652d368922bdc5a8e847453d5e22959697e973eb61093b1c30eeff15b2`
- Evaluator-freeze manifest: `09dade9c6e5198465a8480d8559c31f703d5517d9f2b0a58a1c6a87e8c427f50`
- Evaluator-freeze file SHA256: `5109f4b91cc3b17ea3cea8ffc79864f393c36341efe30234c36ed77d1b8cbfdf`
- Strict-engine SHA256: `e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23`

## Frozen schedules

| Clock | All | 2020–2022 | 2023 | Schedule SHA256 |
|---|---:|---:|---:|---|
| primary | 89 | 67 | 22 | `244324ab8f2e772f8c29f34593de6c83ba29ada543266b53ca64238b0afd1246` |
| one-extra-bar delay | 89 | 67 | 22 | `f66d0590c63d2df27f6c3567e630a14018459f3c5dd6bd747919fdd6453a0693` |

Every source clock retains the frozen ten-minute entry delay and seven-day
hold. The delay diagnostic shifts both entry and exit by exactly one five-minute
bar, retains all events, and does not change side. All clocks are non-overlapping
inside each split and use the preregistered `exit_time <= split_end` containment
rule.

## Economic gate mapping

The freeze copies the preregistration's `economic_gates` object verbatim. The
evaluator applies, at every opened stage:

- absolute return greater than zero;
- CAGR / strict MDD at least 3;
- strict MDD at most 15%;
- positive 10 bp/notional/side stress result;
- positive one-extra-bar-delay result;
- mean gross underlying return at least 30 bp;
- UTC-month clustered one-sided sign-flip `p <= 0.05` with 20,000 draws and
  seed `20260723`;
- positive 2020, 2021, and 2022 train returns, or positive 2023 H1 and H2
  selection returns;
- primary mean gross return at least 5 bp above each of `liquidity_only`,
  `narrative_only`, and `disagreement`.

The side-flip, stale-narrative, and deterministic-random-side clocks remain
report-only falsification diagnostics and cannot rescue or reject the primary.

## Physical outcome isolation

- Freeze OHLC rows parsed: **0**
- Freeze funding rows parsed: **0**
- Simulations during freeze: **0**
- Opened outcome windows: **none**
- 2023 remains physically sealed until a hash-valid Stage1 report is replayed
  from frozen sources and passes every gate.
- Stage1 must match independently sealed prefix hashes:
  - market: `744ac1ad59e53c088e1b6697ecaa073b2cd12cec5823957ac6ffaf2feab896bd`
  - funding: `9a211053a26eb6b3dd0f00a32cb43f2706cea2ca876ed42a936a669039ddff0b`
- Those prefix hashes were copied from the already-disclosed FLCC train report
  `results/federal_liquidity_component_concordance_stage1_2020_2022_2026-07-17.json`
  (file SHA256 `10dc911ad06c7e523d612ff34675421388fefb94fa93e157bfac7e93bd1d82a6`).
- The funding parser checks the boundary timestamp before splitting or
  converting any value from the first sealed row.

## Audit boundary

The supported module API is exactly `freeze_evaluator`, `evaluate_stage1`,
`evaluate_stage2`, and `main`. `evaluate_stage2` reaches the 2023 parser only
through an internal Stage1 hash check and full Stage1 replay. Underscore-prefixed
parser helpers and direct filesystem access are trusted implementation internals,
not authorization endpoints; Python cannot provide a security boundary against
a caller deliberately bypassing its own API.

Artifacts are write-once: an identical rerun verifies the existing bytes, while
different content is rejected rather than overwriting the freeze or a stage
result.
