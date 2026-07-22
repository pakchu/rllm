# AFDR-864 address–funding divergence relay mechanism decision

## Decision

The next standalone BTC candidate is **AFDR-864 — Address–Funding Divergence
Relay, 72-hour hold**.

AFDR tests one cross-domain disagreement that is available without BTC price:

1. Bitcoin funded-address stock and active-address flow broaden or contract over
   seven calendar days;
2. completed BTCUSDT perpetual funding settlements express the opposite
   leveraged-positioning state over the preceding 72 elapsed hours; and
3. only the first daily onset of that disagreement creates a candidate clock.

The fixed economic direction is:

- network breadth high and completed funding pressure low -> **LONG BTC**;
- network breadth low and completed funding pressure high -> **SHORT BTC**.

This document is frozen before AFDR source incidence or any matching BTC
market outcome is computed. The repository has already seen broad historical
BTC results and the two source files were used by earlier, different
mechanisms, so this is a candidate-level freeze rather than a pristine global
holdout. Only a future shadow period beginning after the final policy freeze
can provide pristine forward evidence.

## Why this is a new mechanism

AFDR is not a repair of ARCR-864, CDLTR-72A, BFMWD-144, or DLPD-12:

- ARCR used opposing tails of address-stock growth and address turnover, never
  derivatives positioning. It failed its frozen incidence floor and remains
  retired; AFDR does not relax an ARCR threshold or reuse its event state.
- CDLTR used active-address, transfer, and transaction votes only as the first
  report after agreement between ON RRP and Cboe term structure. AFDR has no
  macro gate, no ordered three-source relay, and uses funded-address stock plus
  active-address flow as one slow network breadth state.
- BFMWD used Bitfinex margin-funding inventory deployment. DLPD used BTC
  dominance versus BTC premium. Neither combined daily on-chain breadth with
  completed Binance BTCUSDT funding.

Repository search before this decision found no existing candidate that uses
`AdrBalCnt` or `AdrActCnt` together with realized BTCUSDT funding rates.

## Frozen source contract

### Coin Metrics daily address source

- file: `data/coinmetrics_btc_address_reservoir_2019_2023.csv.gz`;
- SHA-256:
  `15550072f954d29ae4c9ffe16e11f07c492ee5b6b956e54654b14b9a7af5170a`;
- allowed fields:
  `observation_date,available_at,AdrBalCnt,AdrActCnt`;
- causal clock: exact `AssetEODCompletionTime` represented by `available_at`,
  never earlier than observation midnight plus one UTC day;
- current-candidate freshness: availability may be no later than observation
  midnight plus three UTC days.

`AdrBalCnt` is funded-address stock and `AdrActCnt` is daily active-address
flow. Addresses are not entities: custodial aggregation, change addresses, and
one owner controlling multiple addresses limit the economic interpretation.
The downloaded history is a frozen current vintage, not a historical archive
of every Coin Metrics revision. Live promotion requires forward vintage-parity
monitoring.

### Binance completed BTCUSDT funding source

- file: `data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz`;
- SHA-256:
  `3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6`;
- source-only signal fields:
  `funding_time_ms,funding_time_utc,symbol,funding_rate`;
- exact physical header additionally contains
  `settlement_mark_price,mark_open_time_ms,mark_open_time_utc,`
  `funding_time_offset_ms,mark_source`; source support may validate those names
  from a zero-row header read but may not parse their values;
- required symbol: `BTCUSDT`;
- causal clock for a settlement: `funding_time_utc + 5 minutes`.

The five-minute allowance prevents a historical settlement timestamp from
being treated as an instantaneous live publication. Settlement mark price and
all mark-price fields are forbidden during source support. They may be opened
later only by a separately frozen economic evaluator to calculate exact
funding cash.

## Frozen feature

For daily address observation `t`, using the exact observation seven calendar
days earlier:

```text
balance_growth_7d  = log(AdrBalCnt_t / AdrBalCnt_t-7)
activity_growth_7d = log(AdrActCnt_t / AdrActCnt_t-7)
```

The raw address feature availability is the later of the current and exact
seven-day-lag row availabilities. If that timestamp is later than the current
row's own `available_at`, the row is ineligible to signal and may not be
backdated. Its finite raw feature may enter a later rank reference only after
that complete feature availability.

At the current address row's causal `available_at`, take the nine most recent
funding events whose causal availability is not later than that timestamp.
Their canonicalized settlement timestamps must be nine consecutive eight-hour
slots, and the newest event's causal availability must be no more than eight
hours old. This avoids silently dropping the current 00:00 settlement when an
address report becomes available during its conservative five-minute
publication allowance, while still rejecting a missing or stale settlement:

```text
funding_pressure_72h = sum(funding_rate over those nine settlements)
```

No funding event is forward-filled, interpolated, or reused before its causal
availability. A missing, duplicate, non-BTCUSDT, nonfinite, nonconsecutive, or
noncanonical settlement makes that daily feature invalid.

Canonicalize each funding slot as `floor(funding_time_ms / 8h) * 8h`. The
reported timestamp must be between 0 and 60,000 milliseconds after that slot,
and `funding_time_utc` must exactly represent `funding_time_ms`. Consecutive
means adjacent canonical slots differ by exactly eight hours.

Each of the three raw features is transformed into a causal empirical rank
using only finite feature observations that:

1. have a strictly earlier observation date;
2. have complete feature availability strictly before the current address
   `available_at`; and
3. fall in the preceding 365 calendar days.

At least 180 prior observations are required. The rank is
`(count(prior < current) + 0.5 * count(prior == current)) / n`; the current
row is never included. No global fit, future fit, clipping, winsorization, or
threshold search is allowed.

```text
network_rank = (balance_growth_rank + activity_growth_rank) / 2
funding_rank = funding_pressure_rank
```

The singleton state is:

```text
LONG  when network_rank >= 0.75 and funding_rank <= 0.25
SHORT when network_rank <= 0.25 and funding_rank >= 0.75
FLAT  otherwise
```

The absolute rank disagreement is therefore at least 0.50 by construction.
The event is only a transition from a valid FLAT row to LONG or SHORT on the
exact immediately preceding daily observation grid. Persistent states do not
re-enter. A missing, invalid, or stale predecessor is not treated as FLAT and
cannot create an artificial onset; a later valid FLAT row is required before
another event.

## Execution contract

- decision time: current address availability, after proving the exact lag row
  and all nine funding events were already causally available;
- entry: `ceil_to_5m(decision_time) + 5 minutes`; an exact five-minute boundary
  still receives the additional full latency bar;
- exit: exactly 864 five-minute bars (72 elapsed hours) after entry;
- exposure: 0.5x notional;
- no stop, take-profit, trailing exit, leverage search, sign search, or hold
  search;
- events are accepted in global chronological order and skipped while a prior
  event is open;
- any event whose entry or exit crosses a split boundary is skipped rather
  than truncated.

The 7-day address change is one weekly network unit. The 72-hour completed
funding sum spans nine ordinary settlements and the hold matches that
cross-domain digestion horizon. These choices cannot be changed after source
incidence is opened.

## Research windows

- warm-up only: `[2019-01-01, 2021-01-01)` UTC;
- train: `[2021-01-01, 2023-01-01)` UTC;
- sealed selection: `[2023-01-01, 2024-01-01)` UTC;
- every source or outcome row at or after `2024-01-01T00:00:00Z` remains
  closed during source support and train evaluation.

## Source-only admission gates

After split containment and non-overlap, the primary must have:

- at least 50 train events and at least 20 in each train year;
- at least 25 selection events and at least 10 in each selection half;
- at least 15 events per side in train and 7 per side in selection;
- no UTC calendar month above 20% of either split;
- no UTC weekday above 35% of either split; and
- no rolling 30-day window above 25% of either split.

Complete containment means `entry >= split_start` and
`exit <= split_end_exclusive`; a trade is never truncated. Scheduling is
chronological greedy independently for each control: skip an entry strictly
before the previous accepted exit, while an entry exactly at that exit is
admissible. Calendar month and weekday use entry UTC. For rolling
concentration, anchor at every accepted entry `t`, count entries in
`[t,t+30 elapsed days)`, take the maximum, and divide by that split's event
count. Every concentration check is applied separately to train and selection.

Novelty is checked against comparator artifacts frozen no later than this
decision: NTB-7, CVTR-1, ORFR-1, every primary FLCC-1 candidate, every named
prior-microstructure event list, every primary BFMWD-144 variant, and DLPD-12
primary. Grouped variants are independent comparator members; they are never
unioned to make a more favorable result.

For NTB, CVTR, ORFR, FLCC, and BFMWD use the exact common interval
`[2021-01-01,2024-01-01)`. For DLPD use `[2022-01-01,2024-01-01)` because its
frozen source begins in 2022. For each named prior-microstructure list, use
`[2021-01-01,2024-01-01)` intersected with its explicit
`coverage_start_inclusive,coverage_end_exclusive` fields. Coverage is never
inferred from the first or last observed event. Parse exact timezone-aware UTC
entry timestamps without rounding or date coercion.
At least 10 candidate and 5 comparator events must remain in the common scope;
otherwise novelty fails closed rather than becoming not-applicable.

Let both event sets contain unique exact entry timestamps:

```text
exact_jaccard = |candidate intersection comparator| / |candidate union comparator|
candidate_near_share = fraction(candidate with any comparator within +/-6h)
comparator_near_share = fraction(comparator with any candidate within +/-6h)
```

The member passes timestamps only when exact Jaccard is at most 0.10 and the
larger of the two near shares is at most 0.35. Timestamp distance is absolute
elapsed UTC time and includes the exact six-hour boundary.

For directional-interval members, additionally build a complete five-minute
UTC-open grid over the same common scope. Intervals are entry-inclusive and
exit-exclusive, with LONG `+1`, SHORT `-1`, and flat `0`. Entry and exit must
lie exactly on five-minute opens; any within-member interval overlap, unknown
side, invalid interval, or zero-variance exposure fails closed. Ordinary
Pearson correlation is computed on the complete grid and its absolute value
must be at most 0.40. Timestamp-only prior-microstructure members do not gain
an inferred side, exit, or hold.

An empty, malformed, hash-drifted, low-common-support, or unknown-capability
comparator fails closed.

Failure retires AFDR-864 before BTC outcomes. No sign flip, threshold change,
feature deletion, hold change, calendar gate, LLM, or RL policy may rescue it.

## Frozen controls

Source support retains clocks for diagnostics only:

1. `balance_only`: identical causal rank and tail, without activity or funding;
2. `activity_only`: identical causal rank and tail, without balance or funding;
3. `funding_only`: opposite-direction funding tail on the same daily clock;
4. `one_address_report_delay`: exact primary side delayed to the next valid
   address report;
5. `direction_flip`: exact primary clock and opposite side;
6. `deterministic_random_side`: exact primary clock with SHA-256-fixed side.

Controls cannot replace a failed primary. A later economic evaluator must also
retain one extra five-minute latency bar and 10 bp-per-side cost stress.

The first three controls have exact sides: `balance_only` and `activity_only`
are LONG at rank at least 0.75 and SHORT at rank at most 0.25;
`funding_only` is LONG at funding rank at most 0.25 and SHORT at funding rank
at least 0.75. `one_address_report_delay` moves the frozen primary side to the
next valid address report and recomputes only execution/non-overlap.
`deterministic_random_side` uses the first SHA-256 byte of
`AFDR-864|20260720|<primary_entry_time_utc>`, LONG below 128 and SHORT
otherwise.

## Economic and RLLM boundary

Only an unchanged source-support and novelty pass may authorize a separate,
tested, committed, and hash-frozen economic evaluator. It opens train first
and may physically prepare 2023 BTC market bars and funding settlement-mark
values only after train passes every gate; the already source-qualified 2023
signal clock stays hash-bound. Required train and selection gates are positive
absolute return, full-calendar CAGR/strict-MDD at least 3, strict MDD at most
15%, stress CAGR/strict-MDD at least 2.5, mean gross side-adjusted edge at
least 30 bp, positive contained half-years, positive contribution from both
sides, positive one-extra-bar-delay return, and weekly-cluster one-sided
significance `p <= 0.10`.

The significance test is frozen separately in every opened split. Its inputs
are base-cost, exact-funding `net_return` values for accepted trades. Cluster
by UTC ISO year-week of entry. The statistic is the arithmetic mean trade
return. Under the null, multiply every return in one week by one shared
independent Rademacher sign. Draw 100,000 cluster-sign vectors with NumPy
`default_rng(20260720)`. Sort clusters by ascending UTC ISO `(year,week)` and
make one `integers(0,2,size=(100000,n_clusters))` call, mapping `0` to `-1`
and `1` to `+1`. The one-sided alternative is positive mean and:

```text
p = (1 + count(randomized_mean >= observed_mean)) / 100001
```

No component/control statistic may replace this primary test.

Strict MDD starts at equity 1.0 and includes the global and pre-entry
high-water mark, entry cost, every held five-minute adverse path, exact
realized funding, a virtual adverse exit cost at every held bar, and actual
exit cost. Exact entry/exit funding credits are excluded while exact
entry/exit debits are retained. Full-calendar CAGR includes idle cash.

Gemma/RLLM may be introduced only after deterministic train and selection both
pass. It may learn a constrained `TRADE`/`ABSTAIN` veto or bounded size within
a new preregistration; it may not create direction, retime events, alter the
hold, or rescue a failed deterministic mechanism.
