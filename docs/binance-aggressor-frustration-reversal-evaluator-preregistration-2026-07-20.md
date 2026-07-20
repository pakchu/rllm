# BAFR-24F strict evaluator preregistration — 2026-07-20

## Status and irreversible order

This document freezes the only permitted pre-2024 profitability evaluation for
`BAFR-24F`. At this commit no BAFR post-entry OHLC value, funding cash flow,
return, PnL, absolute return, CAGR, strict MDD, or hit rate has been opened.

The sequence is write-once and irreversible:

1. commit the evaluator, freeze tool, this document, and unit tests;
2. on that clean commit, scan only physical timestamp columns and freeze every
   policy clock, input hash, outcome boundary, parameter, and source hash;
3. commit the evaluator-freeze artifact;
4. open only the `2020-01-01 <= t < 2023-01-01` train values;
5. reject immediately if train fails any gate; otherwise commit its write-once
   result; and
6. only after exact train replay may the evaluator open
   `2023-01-01 <= t < 2024-01-01` sealed-selection values.

No 2024, 2025, or 2026 value is permitted in this evaluator. Parameters,
thresholds, controls, side mapping, costs, accounting, and gates are immutable
after the first outcome is opened. A failed control may not replace the primary
policy under the BAFR name.

## Hash-bound source chain

The freeze must bind and verify all of the following before execution:

- BAFR feature source SHA256
  `e46dc9a4f5e4d4a93bc260d40c0a599ccd0e609d5cb8ebf438c716f7272f7275`;
- BAFR feature manifest SHA256
  `9fa1025c90fb8ad1729f2278236a73e94b0d20bcf9b79178610306cf3b85a28b`;
- BAFR support result SHA256
  `cf6edad6a4eb46c6630dbb5008c88da1ddd39f9ac5c1606785be02f2b323fb62`;
- BAFR primary clock SHA256
  `f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747`;
- BAFR novelty result SHA256
  `38ab5dbb1b36f14e32a4d7a09d94c37b84eaec5d1b75bbc5ef576660e05e3028`;
- official Binance USD-M five-minute kline SHA256
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`;
- its manifest SHA256
  `c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e`;
- exact realized-funding timestamps/rates plus frozen settlement-mark-proxy SHA256
  `3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6`;
- its manifest SHA256
  `a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b`;
- prior completed-bar aggregate-trade microstructure SHA256
  `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`;
  and
- its manifest SHA256
  `6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73`.

The support and novelty artifacts must replay their own canonical hashes,
declare outcomes unopened, and retain passing decisions. The market and
funding freeze may parse only the first physical timestamp field. It must not
construct a return, label, position, or execution result.

The older aggregate-trade file is explicitly classified as a **pre-entry
predictor source**, not as the evaluator's market outcome source. Its frozen
completed-bar `signed_quote_notional`, `quote_notional`, and
`micro_log_return` fields may be loaded solely to construct the
`completed_bar_rejection` clock before outcomes. They use only trades completed
by the signal-bar close. No official-kline OHLC field, post-signal bar, funding
field, or strategy return may enter that clock.

## Frozen primary and control clocks

Every score clock uses the original BAFR scheduler: prior-clean 90th percentile
of absolute score, an 8,640-clean-observation rolling reference, at least 2,016
prior clean observations, one-bar shift of the reference, next-five-minute-open
entry, exactly 24 held bars, chronological non-overlap, and re-entry permitted
at the preceding exit open. The original gap-day/missing-row quarantine and the
following 24 bars remain excluded.

The frozen policies are:

1. `primary`: original BAFR score and side;
2. `direction_flip`: primary clock with every side multiplied by `-1`;
3. `aggressor_flow_only`: score
   `-signed_quote_notional / quote_notional`, i.e. reverse raw aggressive flow;
4. `tick_direction_only`: score `tick_notional_imbalance`, i.e. follow only
   completed trade-price tick direction;
5. `strict_nonzero_tick_only`: score
   `(strict_sell_frustrated_notional - strict_buy_frustrated_notional) /
   quote_notional`;
6. `carried_zero_tick_only`: score
   `(carried_sell_frustrated_notional - carried_buy_frustrated_notional) /
   quote_notional`;
7. `completed_bar_rejection`: from the older verified aggregate-trade source,
   define `flow = signed_quote_notional / quote_notional`; score is `-flow`
   only when `sign(flow)` opposes `sign(micro_log_return)`, otherwise zero;
8. `stale_1h`: primary signal, entry, exit, and side delayed exactly 12 bars;
   and
9. `stale_24h`: primary signal, entry, exit, and side delayed exactly 288 bars.

The direct score controls receive no feature, threshold, hold, or scheduling
grid. Their q90 threshold is calculated from their own strictly prior clean
observations. The completed-bar source is observable at bar completion and
enters only at the following open. Stale clocks drop trades whose shifted exit
is not strictly before `2024-01-01`.

## Stage windows

Train is the full three-year wall clock `2020-01-01` through `2023-01-01`,
with separately reported and gated 2020, 2021, and 2022 calendar years.
Selection is the full one-year wall clock `2023-01-01` through `2024-01-01`,
with separately reported and gated 2023 H1 and H2. A trade crossing a split or
stage boundary is excluded from that particular simulation; no partial trade
is manufactured.

Idle periods remain cash and count in both absolute return and CAGR. CAGR uses
the declared full wall-clock interval, never active time or first-to-last trade
time.

## Frozen execution and strict-MDD accounting

- initial equity is `1.0` and leverage is `0.5x`;
- entry and exit execute at their frozen five-minute opens;
- quantity is fixed for the trade as
  `entry_equity * leverage / entry_open`;
- base cost is 6 bp of notional on each side;
- stress cost is 10 bp of notional on each side;
- exit cost uses exit notional, not entry notional;
- realized funding cash is
  `-side * quantity * settlement_mark_price * funding_rate`;
- all settlements in the closed interval `[entry_time, exit_time]` are
  considered; at an exact entry or exit boundary, a debit is included while a
  credit is discarded as a conservative ordering rule, with discarded boundary
  credits reported separately; and
- bankruptcy is floored at zero and produces 100% strict MDD.

Strict MDD uses a global high-water mark that carries through idle periods and
across trades. For every held trade the evaluator pessimistically orders the
path as: entry cost; all favorable held OHLC movement plus all admissible
funding credits; then all adverse held OHLC movement plus all funding debits
and a hypothetical exit cost at the adverse price; then realized exit price,
realized funding, and realized exit cost. This favorable-first/adverse-second
envelope is intentionally stricter than an unknown intrabar path.

## Frozen performance and falsification gates

The primary must satisfy every condition independently in train and selection:

- full-window absolute return is positive;
- full-window CAGR / strict MDD is at least `3.00`;
- full-window strict MDD is at most `15.00%`;
- 10 bp/side stress absolute return is positive and stress CAGR / strict MDD
  is at least `2.50`;
- mean gross underlying move per trade is at least `24 bp`;
- weekly-cluster one-sided sign-flip p-value is at most `0.10`;
- every calendar-year or half-year split has positive absolute return;
- both long-only and short-only full-calendar contributions have positive
  absolute return;
- train has at least 500 fully-contained trades and at least 100 in each year;
- selection has at least 150 fully-contained trades and at least 60 in each
  half-year;
- train has at least 100 trades on each side and selection at least 40 trades
  on each side;
- train has at least 26 nonempty UTC weekly clusters and selection at least 12;
- base and stress runs have identical trade counts and no qualifying ratio uses
  the zero-MDD cap;
- the primary base CAGR/strict-MDD exceeds every one of the eight frozen
  controls by at least `0.25`; and
- none of `aggressor_flow_only`, `tick_direction_only`,
  `strict_nonzero_tick_only`, `carried_zero_tick_only`, or
  `completed_bar_rejection` independently passes all primary performance gates.

Direction flip and the two stale clocks are superiority/falsification controls
but are not eligible replacement mechanisms. The five direct score controls
are additionally subject to the independent-pass rejection above.

The weekly test clusters realized trade returns by UTC `W-SUN` entry week,
preserves every within-week cluster sum, and sign-flips whole clusters. It uses
exact enumeration for at most 18 nonempty weeks, otherwise 100,000 deterministic
Monte Carlo permutations with seed `20260720` and the add-one p-value correction.

No gate may be weakened and no failed parameter may be repaired. Train failure
keeps 2023 sealed. Selection failure retires BAFR before any forward-window or
portfolio test. Even a selection pass is only permission for a separately
frozen forward/orthogonality study; it is not permission for live trading.
