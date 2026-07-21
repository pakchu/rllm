# DEWH-144 source-support result — rejected before novelty and economics

## Decision

DEWH-144 is retired. The frozen strike-wall handoff conjunction produced only
41 train clocks and 10 source-only 2023 selection clocks, below the immutable
support floors. No threshold, rank window, distance band, side, timing rule,
or hold was changed after incidence was observed.

- immutable result:
  `results/deribit_expiry_wall_handoff_source_gate_2026-07-21.json`;
- artifact SHA-256:
  `791ac8e41793552f99aed28f2a93bfae2946c687f30444a1a6bb07baed01a4af`;
- result hash:
  `22501638475845783b5f87fd35aa27a22444b98a426015e4e174c0d633866c79`;
- pure-clock artifact: not created;
- novelty evaluation: skipped; and
- failure action: `retire_before_economic_evaluation`.

## Frozen source incidence

The source contained 1,484 wall-valid Deribit BTC option expiries from 2019
through 2023. Of these, 1,278 had the required 180 observations in the strict
prior 365-calendar-day rank window. The frozen singleton required all of:

```text
rank(total_position) >= 0.50
rank(wall_share)     >= 0.70
0.25 <= abs(signed normalized wall distance) <= 1.00
side = sign(signed normalized wall distance)
hold = 144 five-minute bars
```

It selected 51 non-overlapping, split-contained clocks. No candidate was
dropped for overlap or a split boundary.

| Window | Accepted | LONG | SHORT | Active months | Largest month | Largest weekday |
|---|---:|---:|---:|---:|---:|---:|
| train: 2020H2–2022 | **41** | 18 | 23 | 18 | 14.63% | 26.83% |
| selection: 2023 | **10** | 3 | 7 | 8 | 20.00% | 40.00% |

Calendar incidence was:

```text
2020H2 12
2021H1 10   2021H2 6
2022H1  6   2022H2 7
2023H1  7   2023H2 3

2023Q1 2   2023Q2 5   2023Q3 2   2023Q4 1
```

## Failed immutable checks

Eleven preregistered checks failed:

1. train total 60–240: observed **41**;
2. 2021 at least 18: observed **16**;
3. 2022 at least 18: observed **13**;
4. every contained train half at least 8: observed 6 in 2021H2, 6 in
   2022H1, and 7 in 2022H2;
5. train LONG at least 20: observed **18**;
6. train active months at least 24: observed **18**;
7. selection total 20–100: observed **10**;
8. each 2023 half at least 8: observed 7 and 3;
9. each 2023 quarter at least 3: observed 2, 5, 2, and 1;
10. selection LONG at least 6: observed **3**; and
11. maximum selection weekday share at most 35%: observed **40%**.

The train month and weekday concentration checks passed, as did 2020H2,
train SHORT, selection active-month, selection SHORT, and selection
month-concentration checks. Passing checks do not override the conjunctive
failure.

## Frozen diagnostic controls

Controls diagnose incidence only and cannot replace the singleton.

| Clock construction | Train | 2023 selection |
|---|---:|---:|
| expiry-time-only deterministic random side | 914 | 365 |
| wall-concentration gate ablation | 127 | 44 |
| total-position gate ablation | 140 | 27 |
| normalized-distance-band ablation | 95 | 28 |
| largest-instrument concentration substitution | 27 | 10 |
| exact direction flip | 41 | 10 |
| deterministic random side on primary clocks | 41 | 10 |
| fixed alternating side on primary clocks | 41 | 10 |
| one additional five-minute entry delay | 41 | 10 |
| frozen DEHR release side on exact train DEWH clocks | 41 | unavailable |

The daily expiry clock itself is abundant. Each individual gate ablation is
also materially denser than the primary. The failure is the exact frozen
intersection of high total position, high strike-wall concentration, and the
moderate normalized distance band. Promoting an ablation after seeing these
counts would be a new incidence-informed hypothesis, not DEWH-144.

The DEHR control used only the frozen 2019–2022 source and covered all 41 train
DEWH clocks. It intentionally had no 2023 source. It was diagnostic and did
not change primary incidence.

## Outcome boundary

```text
DEWH source rows read            = 1,484
DEHR diagnostic rows read        = 1,119 (expiry and release side only)
comparator novelty rows read     = 0
market rows loaded               = 0
funding rows loaded              = 0
performance artifacts parsed     = 0
return/PnL fields read           = 0
strict simulation calls          = 0
post-2023 source rows loaded      = 0
network calls                    = 0
economic outcomes computed       = false
```

This is only a source-support rejection. It makes no claim about return,
CAGR, MDD, or profitability. Because support failed, the eight-member novelty
cohort was not parsed and no pure clock was published.

## No-repair rule

DEWH-144 must not be reopened by:

- lowering the 0.50 total-position or 0.70 wall-share ranks;
- widening or removing the 0.25–1.00 normalized-distance band;
- reducing the 180-expiry minimum or changing the 365-day window;
- reversing the away-from-wall side;
- changing the conservative entry clock or 144-bar hold;
- substituting an ablation or control for the primary; or
- opening economic outcomes to select among such repairs.

A later Deribit option mechanism requires a new identifier, independently
frozen logic, and a new source-support contract before BTC outcomes are read.
