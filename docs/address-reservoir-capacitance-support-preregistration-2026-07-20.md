# ARCR-864 — source-support preregistration

## Purpose and unopened boundary

This stage asks only whether the **Address Reservoir Capacitance Release**
mechanism has enough source coverage, calendar dispersion, and both directional
signs to justify opening BTC market outcomes. It reads no Binance bar, funding
row, future return, held path, PnL, or source observation at or after
2024-01-01.

The complete 2019–2023 source prefix and candidate incidence have not been
downloaded or counted when this contract is written. A previously disclosed
bounded schema probe read three source rows around each of 2019-01-01,
2021-06-01, and 2023-12-28 without calculating a feature or event. Failure of
any gate below rejects this singleton without changing its feature, sign,
threshold, cadence, hold, or support requirement.

## Frozen source and availability

The source is the Coin Metrics Community API daily BTC asset-metric endpoint:

- exact fields `AdrBalCnt`, `AdrActCnt`, and `AssetEODCompletionTime`;
- exact source interval `[2019-01-01, 2024-01-01)`;
- `AdrBalCnt` and `AdrActCnt` must be positive integers on a unique contiguous
  UTC-midnight daily grid; and
- each row is unavailable until its exact `AssetEODCompletionTime`, which may
  not precede observation midnight plus one day.

The loader rejects row-schema drift, status/review fields, duplicate or missing
days, out-of-range rows, pagination scope changes, non-progress, and
sub-microsecond boundary ambiguity. It binds every parsed response page, the
deterministic gzip, and the deterministic manifest by SHA-256. Raw API pages
are not persisted.

Historical revisions remain a limitation: the file hash freezes one current
vintage, not a point-in-time revision archive. The known 2019 backfill may seed
a reference only after its recorded 2021 availability; it can never emit a
backdated event. Live promotion requires a forward vintage-parity audit.

## Exact singleton feature and side

For observation day `t` and seven-day lag `t-7`:

```text
reservoir_flux_7d = log(AdrBalCnt_t / AdrBalCnt_t-7)
turnover_t         = log(AdrActCnt_t / AdrBalCnt_t)
turnover_shift_7d  = turnover_t - turnover_t-7
activity_flux_7d   = log(AdrActCnt_t / AdrActCnt_t-7)
```

Algebraically, `turnover_shift_7d = activity_flux_7d - reservoir_flux_7d`.
The two gated terms are therefore an intentional stock-versus-flow contrast,
not independent confirmations. `activity_flux_7d` is standardized with the
same causal reference solely for a mandatory component control; it does not
enter the candidate direction gate, although a finite activity score is
required so every accepted candidate has an auditable component control.

The complete feature becomes available at the later availability of rows `t`
and `t-7`. A reference observation is admissible only when all of its own
inputs were available **strictly before** the current feature availability.

Each change is standardized using the sample mean and sample standard
deviation of finite feature observations that:

1. have a strictly earlier observation date;
2. fall in the preceding 365 calendar days; and
3. satisfy the strict availability rule above.

At least 180 reference observations are required. The current observation is
never in its own reference and no clipping, winsorization, rank fit, expanding
future fit, or threshold grid is allowed. Reference rows are not subject to the
three-day current-signal freshness rule: a historical backfill may enter a
reference only after its own complete feature availability. A nonfinite or
zero sample standard deviation, or fewer than 180 admissible references, emits
a neutral score.

Let the scores be `reservoir_z` and `turnover_z`, and define
`spread_z = reservoir_z - turnover_z`:

- **long** when `reservoir_z >= 0.75`, `turnover_z <= -0.75`, and
  `spread_z >= 1.75`;
- **short** when `reservoir_z <= -0.75`, `turnover_z >= 0.75`, and
  `spread_z <= -1.75`; and
- neutral otherwise.

A stale **current candidate row** cannot signal: feature availability must be
no more than three calendar days after observation midnight. An event is only
the first nonzero state after a neutral prior observation; persistent states do
not re-enter and a direct long-to-short or short-to-long jump does not count
until a neutral observation intervenes.

The earliest observable five-minute open is `ceil(feature_available_at, 5m)`.
Entry is one additional completed five-minute latency bar later. The hold is
fixed at 864 five-minute bars (three days). Events are greedily nonoverlapping
within each split, and a trade is admissible only when its complete hold is
contained in that split:

- train-support entry: `[2021-07-01T00:00:00Z, 2023-01-01T00:00:00Z)`; and
- test-support entry: `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`.

Every total, year, half, quarter, month, and side-support count below is bucketed
by the accepted **entry timestamp**, after current-row freshness, transition,
latency, complete split containment, and greedy nonoverlap have all been
applied. Raw state days and rejected overlapping events never count.

There is no sign, lookback, threshold, transition, latency, or hold search.

## Frozen source-quality gate

All checks must pass:

- exactly 1,826 daily rows;
- first observation exactly 2019-01-01 and last exactly 2023-12-31;
- no duplicate or missing date and maximum gap exactly one day;
- exact requested columns and no market, funding, return, or PnL field; and
- every availability at least D+1, with no 2024+ source row.

## Frozen candidate-support gate

All checks must also pass after split-contained nonoverlap:

- at least 90 events in 2021H2–2023;
- at least 55 train events in 2021H2–2022, including at least 15 in
  2021H2 and 30 in calendar 2022;
- at least 30 test events in calendar 2023, including at least 12 in each
  half;
- at least 5 events in every quarter from 2021Q3 through 2023Q4;
- events in at least 25 of the 30 eligible months;
- both long and short shares at least 25% separately in all, train, and test;
  and
- no month above 15% of all events.

No event clock is written if one check fails. Source and candidate incidence
may then be reported, but market outcomes remain unopened and ARCR-864 is
rejected without repair.

## Conditional later evaluation

Only a passing support artifact authorizes a separately committed strict
evaluator. It must open train first and calendar 2023 only after train passes;
2024, 2025, and 2026 YTD are then opened sequentially, stopping at the first
failed sealed gate. Later-year source rows and clocks are fetched and built
only when their preceding gate authorizes them. Every evaluated trade remains
fully contained in its split.

The evaluator must freeze:

- 0.5 base leverage;
- 6 bp base and 10 bp stress notional cost per side;
- full-split wall-clock CAGR including warmup and idle cash;
- strict MDD from the global/pre-entry HWM through entry cost, exact funding,
  every held five-minute path, virtual adverse exit fee, and actual exit; and
- interior exact-time symmetric funding, dropping exact entry/exit credits
  and retaining exact entry/exit debits.

Each opened train/test split must have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD no greater than 15%, positive
stress-cost return, positive one-extra-bar-delayed return, mean gross
underlying edge at least 20 bp, and weekly-cluster sign-flip `p <= 0.10`.

Mandatory frozen controls are exact side flip; reservoir-only state (long for
`reservoir_z >= 0.75`, short for `reservoir_z <= -0.75`); turnover-only state
(long for `turnover_z <= -0.75`, short for `turnover_z >= 0.75`);
active-address-flux-only state (long for `activity_z <= -0.75`, short for
`activity_z >= 0.75`); the exact candidate event and side delayed seven
calendar days before split containment and nonoverlap; one extra five-minute
entry delay; constant-long and constant-short sides on exact candidate clocks;
and a deterministic within-year source permutation. The permutation shuffles
the paired `(reservoir_flux_7d, turnover_shift_7d, activity_flux_7d,
source_lag_days)` tuples across observation days inside each UTC calendar year,
uses NumPy `default_rng` seeded by the first eight bytes of
`SHA256("ARCR-864|<year>")` interpreted as one unsigned big-endian integer,
then recomputes references, states, and clocks.

The comparison metric is mean gross underlying basis points per accepted
trade, calculated separately in every opened split. The candidate must exceed
the best reservoir, turnover, activity-flux, or stale control by at least 5 bp.
Controls may reject the mechanism but cannot replace or tune it.

## Research-history boundary

The repository has already seen broad BTC history. This protocol establishes
only a candidate-level freeze, not a globally pristine holdout. At this freeze,
the disclosed bounded source probe is open; complete source incidence,
candidate incidence, every matching market outcome, and every 2024+ source row
remain unopened.
