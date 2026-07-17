# SFRD-1 evaluator freeze

## Decision

The strict evaluator for the singleton SOFR Rate Dislocation candidate
`SFRD-1` is frozen. This work unit opened **no BTCUSDT OHLC, funding, or
performance outcome** and ran no simulation.

- Evaluator manifest: `a4f5eac7bfd433cbc311c23ee2de594116c5e989bfebf5d3d419f82f294e782e`
- Evaluator artifact SHA-256: `913651dff03c7c62b16d8beaf6056a4f2725827ec178d49d7af8f823cbaa78b9`
- Evaluator source SHA-256: `658c92629616410f76c3231db154f6998cbc37c936ff60235af89686a3ae1914`
- Reused strict engine SHA-256: `e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23`
- Frozen support commit: `216f3e95b8e7c9b10441560cf93deac54b374b03`

The artifact was built twice and was byte-identical. Its verifier confirms
zero opened windows, zero parsed OHLC rows, zero parsed funding rows, no
simulation, no mutable parameters, and unchanged source/control schedules.

## Frozen source-only clocks

All clocks use source information available at the decision timestamp, enter
five minutes later, hold five calendar days, and reserve events globally so
positions never overlap within a clock.

| Clock | Pre-2024 | 2021-2022 | 2023 | Train L/S | 2023 L/S |
|---|---:|---:|---:|---:|---:|
| primary | 158 | 48 | 40 | 31 / 17 | 20 / 20 |
| direction flip | 158 | 48 | 40 | 17 / 31 | 20 / 20 |
| SOFR level tail | 56 | 12 | 10 | 7 / 5 | 0 / 10 |
| five-observation change tail | 125 | 41 | 24 | 22 / 19 | 17 / 7 |
| month turn | 49 | 10 | 10 | 4 / 6 | 0 / 10 |
| one-observation delay | 146 | 43 | 34 | 27 / 16 | 17 / 17 |
| deterministic random side | 158 | 48 | 40 | 25 / 23 | 27 / 13 |

Primary maximum UTC-entry-month concentration remains 5/48 = 10.42% in
2021-2022 and 5/40 = 12.50% in 2023, below the frozen 15% ceiling.

## Sequential isolation

1. Stage 1 may physically parse only `[2021-01-01, 2023-01-01)`.
2. Stage 2 may physically parse only `[2023-01-01, 2024-01-01)`, and only
   after an unchanged, manifest-bound Stage-1 report passes every gate.
3. 2024, 2025, and 2026 YTD remain sealed in both stages.
4. Orthogonality analysis remains forbidden until the deterministic candidate
   passes both standalone stages.

The evaluator rejects a modified evaluator hash, static-input ledger,
selection protocol, strict engine, clock hash, clock count/distribution, Stage-1
manifest, opened-window ledger, or gate set before a downstream window opens.

## Frozen performance contract

- Position: fixed 0.5x account gross.
- Base cost: 6 bp/notional/side.
- Stress cost: 10 bp/notional/side.
- Funding: exact BTCUSDT settlements on `[entry, exit)` with the correct side.
- CAGR: complete wall-clock split, including idle cash.
- Strict MDD: global/pre-entry high-water, entry cost,
  favorable-before-adverse held OHLC, exact funding, hypothetical exit cost,
  and realized exit cost.
- Significance: UTC-ISO-week cluster sign flip, 20,000 draws, seed `20260717`,
  one-sided `p <= 0.10`.

Each standalone stage requires positive absolute return, CAGR/strict MDD at
least 3.0, strict MDD at most 15%, mean gross underlying move at least 35 bp,
and positive absolute return under stress costs. Frozen total, side,
subperiod, and month-concentration gates also apply. Mechanism controls must be
strictly worse under the preregistered comparison, while a fully qualifying
one-observation delay or random-side control rejects the claim.

## Verification

- Ruff: passed.
- Pyright: 0 errors, 0 warnings.
- Related source/preregistration/support/evaluator tests: 35 passed.
- Evaluator-specific tests: 8 passed, including the negative and positive
  Stage-2 authorization paths.
- Independent code review: **APPROVE**, with no remaining findings.

No return, CAGR, strict-MDD, or trade-outcome statistic is reported here
because this freeze deliberately opened no outcome data.
