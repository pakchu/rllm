# DEHR-72 — source-support preregistration

## Purpose and unopened outcomes

This stage asks only whether the **Deribit Expiry Hedge Release** mechanism has
enough pre-2023 source coverage, calendar dispersion, and both directional
signs to justify a strict BTC market evaluation. It opens no Binance bar,
funding row, post-expiry return, held path, PnL, or 2023+ source row.

The complete Deribit delivery incidence had not been downloaded or counted
when this contract was written. The thresholds below are a singleton. Failure
rejects DEHR-72 without changing its lookback, quantiles, side map, latency,
hold, or support requirements.

## Frozen source and causal clock

The source is Deribit's public BTC `delivery` history:

- [official endpoint documentation](https://docs.deribit.com/api-reference/market-data/public-get_last_settlements_by_currency);
- `currency=BTC`, `type=delivery`, `count=1000`;
- source interval `[2019-01-01, 2023-01-01)`; and
- expiry-level aggregates only, with raw responses neither committed nor
  redistributed.

The loader must prove that continuation pagination crossed the lower boundary,
validate the option instrument/expiry clock and reported positions, exclude
futures, reject duplicate instruments, and bind each response page and the
deterministic aggregate by SHA-256. It may not load a BTC market or outcome
table.

The API event timestamp is not treated as a publication timestamp. Historical
availability is synthetically delayed to `expiry + 65 minutes`, matching two
identical live canonical delivery sets five minutes apart after an initial
60-minute embargo. Entry is one further completed five-minute bar later
(`expiry + 70 minutes`). Late live data delays or cancels the event and is
never backdated. The later frozen hold is 72 five-minute bars (six hours).

## Exact singleton source feature

For each expiry:

```text
put_release   = sum(reported position for terminally ITM puts)
call_release  = sum(reported position for terminally ITM calls)
net_release   = put_release - call_release
release_share = abs(net_release) / total reported option position
```

The tentative side is long for positive `net_release`, short for negative
`net_release`, and absent for zero. This is a hypothesis about aggregate hedge
release, not a claim that public open interest identifies dealer inventory.

For each current expiry, calculate two references from **strictly earlier**
expiries in the preceding 365 calendar days, requiring at least 20 prior
expiries:

- total reported option position `q0.50`; and
- release share `q0.70`.

Select exactly when the side is nonzero and both current values are at or above
their references. The current row is excluded from both references. There is
no threshold grid, rank search, sign search, cadence search, or outcome-based
repair. Eligibility is entry time in `[2020-07-01, 2023-01-01)`.

## Frozen source-quality gate

All conditions must pass:

- first source expiry no later than 2019-03-01;
- last source expiry no earlier than 2022-12-25;
- at least 500 source expiries in the eligible interval;
- at least 10 source expiries in every eligible month from 2020-07 through
  2022-12; and
- no gap between eligible source expiries greater than 14 calendar days.

These checks distinguish missing source history from a sparse candidate rule.

## Frozen candidate-support gate

All conditions must also pass:

- at least 120 events in 2020H2–2022;
- at least 70 in train (2020H2–2021), including 15 in 2020H2 and 45 in 2021;
- at least 50 in calendar 2022, including 20 in each half;
- at least 8 in every quarter from 2020Q3 through 2022Q4;
- events in at least 27 of the 30 eligible calendar months;
- both long and short shares at least 25% separately in all, train, and test;
  and
- no calendar month above 15% of all candidate events.

No event clock is written if one check fails. Source incidence may then be
reported, but market outcomes remain unopened and the candidate is rejected.

## Conditional later evaluation

Only a passing source-support artifact may authorize a separately committed
strict evaluator. That evaluator must freeze the following before any matching
post-entry BTC path is read:

- train `2020-07-01` through 2021 and test calendar 2022;
- sealed sequential 2023, 2024, 2025, and 2026 YTD openings;
- 0.5 base leverage;
- 6 bp base and 10 bp stress notional cost per side;
- full-calendar CAGR including warmup and idle cash;
- strict MDD from global/pre-entry HWM through entry cost, exact funding, every
  held five-minute path, virtual adverse exit fee, and actual exit; and
- interior exact-time symmetric funding, dropping exact entry/exit credits and
  retaining exact entry/exit debits.

Each train and test must have positive absolute return, CAGR/strict-MDD at
least 3, strict MDD no greater than 15%, positive stress-cost return, positive
one-extra-bar-delayed return, mean gross underlying edge at least 20 bp, and a
weekly-cluster sign-flip p-value no greater than 0.10.

Mandatory falsification controls are expiry-time-only random side on every
eligible expiry, exact direction flip, equal-position strike ablation
(`sign(ITM put count - ITM call count)`), call/put-type ablation using a fixed
alternating side at exact candidate clocks, deterministic timestamp-seeded
random side at exact candidate clocks, release-share-gate ablation,
total-position-gate ablation, and one additional five-minute entry delay. ITM
call/put instrument counts are retained in the source aggregate specifically
so the position-size ablation is reproducible without persisted raw rows.
Candidate mean gross edge must exceed the best nondirectional control by at
least 5 bp. Controls may reject the mechanism but cannot replace it.

## Research-history boundary

The repository has already seen broad BTC history. This protocol can establish
only a candidate-level frozen sequence. It cannot turn 2022 or later years into
a globally pristine holdout. At preregistration, complete DEHR incidence and
all matching post-entry outcomes remained unopened.
