# FLCC-1 strict evaluator freeze — 2026-07-17

## Status

The evaluator is frozen **before opening any BTC outcome**.

| Freeze check | Value |
|---|---:|
| Parsed execution OHLC rows | 0 |
| Parsed funding rows | 0 |
| Simulations run | 0 |
| Source-clock invariant checks | 96 / 96 passed |
| 2023 outcome opened | No |
| 2024+ outcome opened | No |

## Integrity

| Artifact | SHA-256 |
|---|---|
| Evaluator source | `6b8c35638b1554abf88bf9f8c064f40aa94dadab6c11312e4ebe94e7d7a4f54a` |
| Evaluator freeze JSON | `3daa5701640918b25bd7cd0cd4cb900c491515c9afdb5e9f1c021eaf0161afbb` |
| Freeze manifest | `645f78d2349e699a26d44ef130b20c40d38fdc147a2658ff02f3a31a79d2782f` |
| Frozen clock ledger | `7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c` |
| Reused strict engine | `e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23` |

The evaluator rebuilds every source-only event from the archived H.4.1 panel
and requires exact equality with the frozen ledger. Every candidate/control
schedule is separately hash-bound.

## Physical outcome isolation

### Stage 1

- Physical parse window: `[2020-01-01, 2023-01-01)`.
- The five-minute market parser stops at the first timestamp at or beyond the
  end boundary before parsing its OHLC fields.
- The funding parser applies the same stop rule.
- The full 2020–2023 data-file hash is deliberately not recomputed during a
  stage, because doing so would read sealed rows beyond the boundary.
- All four preregistered FLCC candidates run only in this window.

### Stage 2

- Physical parse window: `[2023-01-01, 2024-01-01)`.
- It is unreachable unless a canonical-hash-valid Stage-1 result is bound to
  this evaluator freeze, passes every gate, and names exactly one member of the
  frozen four-candidate family.
- A self-sealed Stage-1 report is insufficient: before parsing 2023, the
  evaluator physically replays only the 2020–2022 window and requires exact
  reproduction of all candidate metrics, controls, gates, diagnostics,
  selection order, and the sealed report.
- Only that exact Stage-1 winner may run. There is no runner-up fallback.

2024, 2025, and 2026 YTD are not parsed by either stage.

## Statistics and execution

The evaluator reuses the repository's previously audited strict engine:

- 0.5x fixed gross;
- 6 bp/notional/side base cost and 10 bp stress cost;
- exact realized BTCUSDT funding on `[entry, exit)`;
- full wall-clock CAGR including idle cash;
- strict MDD from global/pre-entry high-water, entry/exit cost, funding,
  favorable-before-adverse held OHLC, and hypothetical liquidation;
- weekly clustered one-sided sign-flip test.

Stage 1 applies `p <= 0.025`, the Bonferroni correction for four family
members. Stage 2 applies `p <= 0.10` to the single frozen winner. Both require
CAGR/strict-MDD at least 3 and strict MDD at most 15%.

## Controls

Each candidate is compared against:

- net-liquidity tail without component breadth;
- component concordance without the net tail;
- direction flip;
- one-release delay;
- hash-fixed random side.

The primary must beat both mechanism components, while a strongly qualifying
flipped, delayed, or random clock falsifies the candidate interpretation.
"Qualifying" applies the complete base battery to each falsification clock,
including 10 bp stress profitability and every subperiod's sign/support gates;
it is not inferred from headline performance alone.

## Verification

- Targeted tests: 25 passed across source builder, source clock,
  preregistration, and evaluator.
- A real strict-engine smoke simulation verifies that every frozen schedule
  carries the reused engine's required causal `signal_day` field.
- A regression test proves that a self-sealed fake Stage-1 pass is rejected
  before any execution-data loader is called.
- Ruff: passed.
- Pyright on the three production modules: 0 errors.
- Offline evaluator-freeze replay: manifest verified.
