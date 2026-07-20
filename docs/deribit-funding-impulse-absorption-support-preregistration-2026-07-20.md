# DFIA-72 — source-support preregistration

## Purpose and unopened boundary

This stage asks only whether **Deribit Funding-Impulse Absorption** has enough
pre-2024 source coverage, split-contained event density, calendar dispersion,
and both directional signs to justify a strict BTC market evaluation. It opens
no Binance bar, Binance funding row, future return, held path, PnL, or Deribit
source row at or after 2024-01-01.

The complete 2019–2023 hourly source and candidate incidence have not been
downloaded or counted when this contract is written. The earlier mechanism
decision disclosed only bounded schema/coverage probes and the observed
one-hour/eight-hour memory relationship. Failure of any gate below rejects
this singleton without changing its reference, thresholds, side map, latency,
hold, scheduler, or support requirements.

## Frozen source and causal clock

The only source is Deribit's public BTC perpetual funding-history endpoint:

- `public/get_funding_rate_history`, `instrument_name=BTC-PERPETUAL`;
- exact fields `timestamp`, `interest_1h`, `interest_8h`, `index_price`, and
  `prev_index_price`;
- exact source interval
  `[2019-04-30T10:00:00Z, 2024-01-01T00:00:00Z)`; and
- deterministic request windows no wider than 28 days, with exact boundary
  deduplication and no persisted raw response.

Official references are the [funding-history endpoint](https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history),
the [inverse perpetual specification](https://support.deribit.com/hc/en-us/articles/31424954847133-Inverse-Perpetual),
and the [API usage policy](https://docs.deribit.com/articles/api-usage-policy).

The loader must preserve exact JSON decimals, reject schema/environment drift,
bind every response result and the deterministic source by SHA-256, verify the
contiguous index-price chain, report missing hours without filling them, and
verify that `interest_8h` remains within `0.00005` of the sum of the latest
eight contiguous `interest_1h` observations.

Historical source availability is conservatively assigned to `T + 5 minutes`
for a completed hourly row stamped `T`. The earliest observable five-minute
open is that boundary; entry is one additional completed five-minute latency
bar later, at `T + 10 minutes`. Live collection uses the actual first
successful observation when later and never backdates. A candidate requires
the current eight hourly source rows to be contiguous, so a gap breaks the
signal chain.

## Exact singleton feature and side

For hourly row `t`:

```text
trailing_hourly_mean_t = interest_8h_t / 8
funding_impulse_t      = interest_1h_t - trailing_hourly_mean_t
index_return_1h_t      = log(index_price_t / prev_index_price_t)
```

Each raw feature is standardized independently against finite observations
whose source availability is **strictly earlier** than the current row, whose
timestamp lies in `[T-720h, T)`, and whose own eight-hour source chain is
contiguous and memory-valid. The current row is excluded. At least 360 such
prior observations are required. The reference uses the population mean and
population standard deviation (`ddof=0`); a nonfinite or zero deviation emits
a neutral score. There is no clipping, winsorization, rank fit, expanding
future fit, or threshold grid.

Let the scores be `funding_impulse_z` and `index_return_z`:

- **short** when `funding_impulse_z >= 1.25`, raw
  `index_return_1h <= 0`, and `index_return_z <= 0`; and
- **long** when `funding_impulse_z <= -1.25`, raw
  `index_return_1h >= 0`, and `index_return_z >= 0`.

Thus the index must oppose or fail to confirm the newly crowded funding side
both in raw sign and relative to its causal reference. Exact zero is allowed
on the corresponding side. A row satisfying neither rule is neutral.

Every qualifying hourly row is eligible while flat; no transition-onset rule
is used. Within train and test independently, rows are processed
chronologically and accepted greedily only when their entry is at or after the
previous accepted exit. Back-to-back entries are allowed. The hold is exactly
72 five-minute bars (six hours), and a trade counts only if its entire hold is
contained in its split:

- train: `[2020-01-01T00:00:00Z, 2023-01-01T00:00:00Z)`; and
- test: `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`.

All support counts use the accepted entry clock after reference readiness,
eight-hour continuity, latency, split containment, and greedy nonoverlap.

## Frozen source-quality gate

All checks must pass:

- first row exactly `2019-04-30T10:00:00Z` and last row exactly
  `2023-12-31T23:00:00Z`;
- at least 98% of the exact 40,958 requested hourly timestamps;
- coverage at least 95% in every eligible calendar month from 2020-01 through
  2023-12;
- no adjacent observed-timestamp delta greater than 24 hours (therefore no
  run longer than 23 missing hourly timestamps);
- exact source schema, causal availability, index-price chain, and memory
  invariant; and
- no duplicate, conflicting boundary, market, return, PnL, or 2024+ source
  row.

## Frozen candidate-support gate

All checks must also pass after split-contained nonoverlap:

- at least 300 events in 2020–2023;
- at least 200 train events in 2020–2022 and at least 50 in each train year;
- at least 80 test events in 2023, including at least 30 in each half;
- at least 10 events in every quarter from 2020Q1 through 2023Q4;
- events in at least 44 of the 48 eligible calendar months;
- both long and short shares at least 25% separately in all, train, and test;
  and
- no calendar month above 8% of all accepted events.

No event clock is written if one check fails. Source and candidate incidence
may then be reported, but market outcomes remain unopened and DFIA-72 is
rejected without repair.

## Conditional later evaluation

Only a passing support artifact authorizes a separately committed strict
evaluator. It must open train first and calendar 2023 only after train passes;
2024, 2025, and 2026 YTD source/outcome prefixes are then opened sequentially,
stopping at the first failed sealed gate. Every trade remains split-contained.

The evaluator must freeze 0.5 base leverage; 6 bp base and 10 bp stress
notional cost per side; exact Binance funding; full-split wall-clock CAGR
including warmup and idle cash; and strict MDD from the global/pre-entry HWM
through entry cost, every held five-minute path, exact funding, virtual adverse
exit fee, and actual exit.

Each opened train/test split must have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD no greater than 15%, positive
stress-cost return, positive one-extra-bar-delayed return, mean gross
underlying edge at least 20 bp, and weekly-cluster sign-flip `p <= 0.10`.

Mandatory frozen controls are listed below. Every control uses the same
six-hour hold, costs, exact funding rule, complete split containment, and
chronological greedy nonoverlap. Component/stale controls build and nonoverlap
their own clocks; exact-clock controls retain the candidate clock unless their
stated delay requires rescheduling. Every component/stale control must have at
least 30 accepted trades in each opened split or the mechanism is rejected as
unfalsifiable.

1. exact side flip on the candidate clock;
2. impulse-only events using `abs(funding_impulse_z) >= 1.25` and side opposite
   the impulse, followed by the same split scheduler;
3. index-only events using `abs(index_return_z) >= 1.25` and side opposite the
   index return, followed by the same split scheduler;
4. an eight-hour-stale interaction using the already-causal
   `funding_impulse_z` from exact timestamp `T-8h` while retaining the current
   raw and standardized index response and all singleton thresholds; the row
   is ineligible if that exact stale score is absent;
5. one additional five-minute entry delay on the exact candidate rows before
   split containment and nonoverlap;
6. constant-long and constant-short sides on the exact candidate clock; and
7. a deterministic within-year paired source permutation. For each UTC year,
   permute paired `(funding_impulse, index_return_1h)` tuples across hourly
   timestamps with NumPy `default_rng` seeded by the first eight bytes of
   `SHA256("DFIA-72|<year>")` interpreted as an unsigned big-endian integer,
   then recompute causal references, candidates, and clocks.

The comparison metric is mean gross underlying basis points per accepted
trade in every opened split. The candidate must exceed the best impulse-only,
index-only, or stale interaction control by at least 5 bp. Controls may reject
the mechanism but cannot replace or tune it.

## Research-history boundary

The repository has already seen broad BTC history. This protocol establishes
only a candidate-level frozen sequence, not a globally pristine holdout. At
this freeze, bounded source probes are disclosed as open; complete source
incidence, candidate incidence, every matching market outcome, and every
2024+ source row remain unopened. Calendar 2023 is explicitly an
**outcome-blind support-screened test**, not a pristine global holdout. No
threshold, sign, hold, or support repair is allowed after its candidate
incidence is opened.
