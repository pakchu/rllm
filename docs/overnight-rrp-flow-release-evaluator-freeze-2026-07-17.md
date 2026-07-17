# ORFR-1 strict evaluator freeze — 2026-07-17

## Status

The evaluator is frozen **before opening any BTC outcome**.

| Freeze check | Value |
|---|---:|
| Parsed execution OHLC rows | 0 |
| Parsed funding rows | 0 |
| Simulations run | 0 |
| Stage1 outcome opened | No |
| 2023 outcome opened | No |
| Mutable parameters | 0 |

## Frozen execution schedules

Contained-trade counts require signal, entry, and exit to remain inside the
physical split. One 2022 source event exits at the next 2023 operation and is
therefore correctly excluded from Stage1.

| Clock | Stage1 trades | L/S | Sealed 2023 trades | L/S |
|---|---:|---:|---:|---:|
| Primary | 111 | 63 / 48 | 74 | 50 / 24 |
| One-day-delta control | 98 | 57 / 41 | 83 | 45 / 38 |
| Direction flip | 111 | 48 / 63 | 74 | 24 / 50 |
| One-release delay | 111 | 63 / 48 | 74 | 50 / 24 |
| Deterministic random side | 111 | 48 / 63 | 74 | 40 / 34 |

Every entry is exactly five minutes after the frozen source decision. Every
exit is the next normal ON RRP result clock plus five minutes. Holding periods
are variable by construction—normally one day and longer across weekends and
holidays. All five schedules are globally non-overlapping.

## Sequential isolation

1. Stage1 may physically parse only `[2021-01-01, 2023-01-01)`.
2. Stage2 may physically parse only `[2023-01-01, 2024-01-01)`.
3. Before any 2023 row is parsed, Stage2 requires a canonical-hash-valid
   Stage1 report that passed every gate and exactly replays under this frozen
   evaluator.
4. A failed or modified Stage1 report rejects before the execution loader.
5. 2024 and later are never parsed.

The parser stops at the physical end boundary before decoding future OHLC or
funding fields. It deliberately does not hash the full combined execution file
during a stage, because hashing it would read sealed rows.

## Frozen performance contract

- fixed 0.5x gross;
- 6 bp/notional/side base cost and 10 bp stress cost;
- exact BTCUSDT funding on `[entry, exit)`;
- full wall-clock CAGR including idle cash;
- strict MDD from global/pre-entry high-water, entry and hypothetical-exit
  costs, favorable-before-adverse held OHLC, funding, and realized exit cost;
- deterministic two-sided UTC-ISO-week sign-flip test, 20,000 draws, seed
  `20260717` when exact enumeration is infeasible.

Stage1 requires at least 100 contained trades, 45 per year, 35 per side,
CAGR/strict-MDD >= 3, strict MDD <= 15%, `p <= 0.10`, positive stress return,
positive return in both years, at least 35 bp mean gross underlying return,
and a 0.25 ratio margin over the one-day-delta mechanism control. No
falsification control may fully qualify.

Stage2 has its separately frozen 60-trade, 20-per-half, and 15-per-side floors.

## Integrity

| Artifact | SHA-256 / manifest |
|---|---|
| Evaluator source | `8bd60256d065da1750c9852b7c7b47375ad1cd65842b4b8e256cc43b470a8567` |
| Evaluator freeze JSON | `f8b47e3ae7d2d6ac62baef61c8e117b7640b9d902433e45670fab7e19fbb0c9c` |
| Evaluator freeze manifest | `1f5ea49c04d670afc6ef612e0c439076c37213cbe0b2f7f5b2bfa20b43009730` |
| Reused strict engine | `e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23` |

Two consecutive freeze builds were byte-identical.
