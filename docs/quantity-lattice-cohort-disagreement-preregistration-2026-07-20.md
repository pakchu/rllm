# QLCD-288 preregistration — quantity-lattice cohort disagreement

## Decision and evidence boundary

QLCD-288 tests a new raw BTCUSDT observable: disagreement between aggressive
trades whose exact base quantity lies on a coarse `0.1 BTC` lattice and trades
that remain on the fine `0.001 BTC` lattice.  The fixed direction follows the
coarse cohort and the fixed hold is 24 hours.

This protocol is frozen before reading real QLCD event incidence, a QLCD clock,
post-entry price, funding, return, PnL, CAGR, or MDD.  A bounded source-only
probe inspected quantity precision in 10,000 rows from `2020-01-01` and
`2023-12-31`; both samples used at most three decimal places.  It did not count
cohort membership or compute the feature.

## Why this is a new mechanism

Prior trade-size research used candle-level average trade size, aggregate-event
notional tails, buy/sell average-size ratios, or the number of underlying trade
IDs spanned by an aggTrade.  QLCD uses none of those quantities.  It retains
the exact base-quantity residue class of each raw aggregate event.

- AFCS tested high underlying-fill span plus price response; QLCD uses no fill
  span and no price.
- NETF tested event-count versus capital direction; QLCD compares two exact
  quantity-denomination cohorts.
- SMCC tested transaction-millisecond collisions and displacement; QLCD ignores
  collision timing and price.
- The old trade-size scan used `quote_volume / number_of_trades` from candles;
  it could not recover a `0.1 BTC` residue class.

The mechanism is behavioral, not an identity claim: an exact-denomination
aggregate-event cohort may encode a different execution convention from
fine-lot flow.  Binance `aggTrade.quantity` is not assumed to equal a submitted
parent-order size.  If the cohorts oppose, QLCD tests whether inventory resolves
toward the coarse side.  It does not label either cohort institutional or
informed.

## Exact causal feature

Convert each positive aggregate-trade quantity to integer milli-BTC and fail if
`quantity * 1000` differs from its nearest integer by more than `1e-9`.
Classify mutually exclusive cohorts:

```text
coarse: q_mbtc % 100 == 0
medium: q_mbtc % 10 == 0 and q_mbtc % 100 != 0
fine:   q_mbtc % 10 != 0
```

Medium is retained only for a later frozen control.  In completed five-minute
bar `t`:

```text
coarse_share     = coarse_qty / total_qty
coarse_coherence = abs(coarse_signed_qty) / coarse_qty
fine_signed_share = fine_signed_qty / fine_qty
coarse_side      = sign(coarse_signed_qty)
opposition       = max(-coarse_side * fine_signed_share, 0)
score            = coarse_share * coarse_coherence * opposition
```

Aggressive side is `+1` when `is_buyer_maker=false` and `-1` otherwise.  Price
and quote notional are forbidden from the source signal.

## Singleton clock

At `t`, require a complete source row, at least 64 aggregate events, at least
three coarse events, at least 16 fine events, nonzero coarse side, positive
opposition, and positive score.  Score must be at or above the strictly prior
30-day q99.75:

```text
score.where(source_complete).shift(1)
     .rolling(8640, min_periods=2016).quantile(0.9975)
```

There is no quantile, cohort, direction, hold, or regime grid.  The signal is
known after bar `t` closes, entry is the open at `t+2`, and exit is 288 held
five-minute bars later.  Positions do not overlap; re-entry at a scheduled exit
is allowed.  Future source validity after decision `t` cannot cancel a clock.

The source-independent severity definition is q99.75 of a bounded `[0,1]`
product that is large only when coarse share, coarse directional coherence, and
fine-cohort opposition are jointly exceptional.  It is not selected from an
event-count grid.  The 288-bar hold spans one full UTC day and three ordinary
eight-hour perpetual funding cycles, fixing one cross-session inventory
resolution horizon rather than a tested hold menu.

The six frozen source-gap UTC days, verified empty buckets, and following 24
bars are quarantined.  The added `2020-01-15` day contains the already audited
exact duplicated underlying event.  Raw ZIPs are streamed and not persisted.

## Support and novelty stopping gates

Before outcomes, require 200–800 non-overlapping events, at least 35 per year,
15 in each 2023 half, each side between 25% and 75%, and no month above 15%.
The clock must pass exact and +/-12-bar one-to-one overlap gates against MFIC,
AFCS, TAAR, RIFT, PCP, and the rejected-but-frozen SMCC clock.  Dense BAFR is
report-only; its absence or parse error is disclosed but cannot change the
decision.  Missing or changed sparse comparator artifacts fail closed.
NETF and the old candle-average trade-size scan have no canonical outcome-blind
clock.  RIFT is the frozen topology substitute for NETF; AFCS and MFIC are the
closest executable fill/size comparators.  These exclusions are frozen before
QLCD incidence.

Failure retires QLCD-288 without repair.  Only a support/novelty pass permits a
separately committed economic evaluator.  Later stages open sequentially:
2020–2022 train, 2023 selection, 2024 test, 2025 eval, then 2026 report.  Each
stage uses full-calendar CAGR, strict held-path MDD, 6 bp/side base and 10
bp/side stress costs.  Promotion requires positive return, CAGR/MDD at least
3, MDD at most 15%, positive stress return with stress ratio at least 2.5,
mean gross move above 24 bp, and weekly-cluster `p < 0.10`.

If the base alpha survives, its categorical cohort state can enter the RLLM as
an abstention/sizing observation.  The LLM may not reconstruct or mutate the
base event from sealed outcomes.
