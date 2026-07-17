# SFRD-1 preregistration — Secured Funding Rate Dislocation

## Claim boundary

**No SFRD-1 post-entry BTC outcome has been opened.** This document freezes a
source-only-screened exploratory two-sided singleton before reading
entry-to-exit OHLC, return, funding PnL, CAGR, or drawdown.

- official source: New York Fed SOFR median rate;
- signal: extreme daily SOFR change relative to exactly 120 prior changes;
- action: tightening shock → short BTC, easing shock → long BTC;
- entry: one complete five-minute bar after the rate's conservative timestamp;
- hold: 1,440 five-minute bars / five calendar days;
- leverage: 0.5x;
- base cost: 6 bp/notional/side;
- stress cost: 10 bp/notional/side;
- Stage 1: 2021-2022;
- conditional Stage 2: 2023;
- 2024, 2025, and 2026 YTD remain sealed.

The repository has seen these BTC years in unrelated research, so this is not a
global clean room. The exact SFRD-1 post-entry outcome remains unopened. A
source-only preflight saw the frozen SOFR feature values, event timestamps, and
counts through 2023. Binary-float arithmetic was discarded because ties were
not reproducible. Under exact decimal arithmetic, tail thresholds `0.80`,
`0.825`, `0.85`, `0.875`, `0.90`, `0.925`, and `0.95` were inspected for event
density and concentration. The final 15% tail clock has 48 non-overlapping
events in 2021-2022 and 40 in 2023. No crypto market field, outcome, action
direction, or hold was compared. This is therefore **not** a clean source-OOS
support claim. Event density through 2023 is in-sample; 2023 market outcomes
remain sealed and may only pass or reject the frozen singleton.

## Economic mechanism and independent axis

SOFR measures overnight cash borrowing secured by U.S. Treasury securities.
SFRD-1 asks whether an unusually large day-to-day change in that secured-dollar
funding rate transmits to crypto risk appetite over several days. A tightening
shock is treated as deleveraging pressure and fixes a short; an easing shock is
treated as restored funding capacity and fixes a long.

The signal uses no BTC price, volume, taker flow, open interest, perpetual
funding, basis, premium, Kimchi, FX, DXY, REX prediction, existing alpha state,
or portfolio PnL. This makes its input source independent of the occupied alpha
families even though trade-level orthogonality remains unproven until standalone
economics pass.

The source also contains percentile and volume summary fields, but SFRD-1
**forbids** them. New York Fed policy allows quarterly-updated summary values,
so the frozen historical table is not a safe same-day vintage for those fields.
SFRD-1 consumes only `sofr_percent` at `sofr_available_at_utc`, whose timestamp
is after the possible same-day rate revision window.

## Frozen feature and event

For SOFR observation index `t`, parse `SOFR_percent` as a base-10 decimal,
multiply by 100, and require an exact integer basis point. Binary floating-point
rate comparison is forbidden.

```text
D[t] = SOFR_integer_bp[t] - SOFR_integer_bp[t-1]
N[t] = 2 * count(D[t-120:t] < D[t])
       + count(D[t-120:t] == D[t])
R[t] = N[t] / 240
```

The rank window contains exactly 120 prior finite changes and excludes `t`.
No calendar interpolation, weekend row, expanding fallback, imputation, or
current-row inclusion is allowed.

```text
state[t] = +1  if N[t] >= 204    # R >= 0.85, tightening
           -1  if N[t] <= 36     # R <= 0.15, easing
            0  otherwise
```

An event occurs only if `state[t]` is nonzero and differs from the immediately
prior SOFR row's state. A direct `+1 -> -1` or `-1 -> +1` switch is a new event.
The side is fixed before market outcomes:

- `state=+1`: short BTCUSDT USD-M perpetual;
- `state=-1`: long BTCUSDT USD-M perpetual.

At `sofr_available_at_utc`, the evaluator waits through one full five-minute
bucket and enters at the following open. This produces 19:05 UTC entries during
EDT and 20:05 UTC entries during EST. The position exits after exactly 1,440
held bars. Global reservation ignores rather than queues any signal whose entry
would occur before the current scheduled exit.

The complete admitted source clock is frozen at:

`results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz`

It contains 158 pre-2024 source events and has SHA-256
`391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69`.
Every later support or performance stage must reproduce all 158 rows exactly,
not merely their count, before opening a market outcome.

## Outcome-blind support contract

There is no **remaining** threshold grid after the disclosed pre-freeze
source-only screen. The next support stage is an exact implementation replay,
not an independent generalization test; it must reject drift and cannot choose
a fallback:

| Window | Minimum events | Preflight count |
|---|---:|---:|
| 2021-2022 | 45 | 48 |
| 2021 | 10 | 12 |
| 2022 | 35 | 36 |
| 2023 | 35 | 40 |
| 2023 H1 | 15 | 18 |
| 2023 H2 | 18 | 21 |

Train must contain at least 15 events per side; the replay has 31 long and 17
short. 2023 must contain at least 18 per side; the replay has 20 and 20. No
single UTC entry month may exceed 15% in either window. Actual maximum shares
are 5/48 = 10.42% in train and 5/40 = 12.50% in 2023. Support
may inspect source values, timestamps, sides, concentration, and control-clock
overlap only. It may not load market OHLC or performance.

## Frozen controls

1. exact direction flip on the primary clock;
2. a 120-prior 15% tail rank of the SOFR level instead of its daily change,
   with high level → short and low level → long;
3. a 120-prior 15% tail rank of the exact-integer five-observation SOFR change,
   with high change → short and low change → long;
4. first/last-SOFR-business-day month-turn events using the sign of `D[t]`;
5. primary state and side delayed one SOFR observation;
6. deterministic random side on the primary clock: compute
   `SHA256("SFRD-1-random-side-20260717|" + entry_time)` per event in ascending
   ledger order; first digest byte below 128 means long, otherwise short.

Controls can falsify the mechanism and can never replace SFRD-1.

## Strict staged evaluation

Stage 1 may read BTCUSDT five-minute execution OHLC and exact funding only
through 2022-12-31. Stage 2 may open 2023 only after an unchanged Stage 1 passes
every gate. Every result reports **absolute return, full-clock CAGR, strict MDD,
CAGR/strict-MDD, and trade count**.

Both Stage 1 and 2023 must have:

- positive absolute return;
- CAGR/strict-MDD at least 3.0;
- strict MDD at most 15%;
- weekly-cluster one-sided sign-flip `p <= 0.10` with 20,000 draws and seed
  `20260717`;
- positive absolute return at 10 bp/notional/side;
- mean gross underlying move at least 35 bp per trade.

2021, 2022, and both 2023 halves must each have positive absolute return and
their frozen minimum counts. Stage 1's primary ratio must strictly beat the
level-tail, five-observation-change, and month-turn control ratios. After Stage
2, the primary's minimum train/2023 ratio must strictly beat the corresponding
minimum for every mechanism control. Equality rejects. A fully qualifying
one-observation delay or random-side control rejects the claim.

Strict MDD uses global/pre-entry high water, entry cost, favorable-before-
adverse held OHLC, exact funding, hypothetical liquidation cost, and exit cost.
CAGR uses the full wall-clock split, including idle cash and warm-up.

## Orthogonality and RLLM boundary

Only a standalone pass unlocks comparison with the frozen existing-alpha
universe. SFRD-1 must satisfy exact entry Jaccard `<=0.02`, near-six-hour entry
fraction `<=0.25`, occupied-time Jaccard `<=0.15`, and absolute daily-PnL
Pearson `<=0.30`, with at least 20 nonzero PnL days, then improve a synchronized
portfolio marginally.

Only after deterministic standalone and orthogonality passage may a compact
RLLM receive symbolic SOFR delta/rank/state, source gap, current position, and
time-to-exit tokens. It may abstain or size the frozen side. It may not reverse
the side or redesign the event.

Any support or staged-performance failure retires SFRD-1. No threshold, side,
feature, delay, hold, or gate may be repaired after outcomes open.
