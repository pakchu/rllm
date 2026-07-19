# PSR-30/6 — Premium Snapback Recenter preregistration

Date frozen: 2026-07-19  
Candidate count: **one**  
BTC execution outcomes opened for this candidate: **no**

## Hypothesis

A 30-minute premium-index path with unusually high intrapath energy, poor net
efficiency, repeated sign changes and a large one-sided excursion that has
already returned near its strictly-prior center represents failed derivatives
pressure. A failed positive excursion is traded short; a failed negative
excursion is traded long.

This is not a single-candle wick/range rule, a funding/premium level rule, or a
repair of the rejected liquidation relay candidates. BTC price action, volume,
funding, OI, regime, external assets, trees, LLMs and prior alpha signals are
excluded from clock construction.

## Frozen source

Official Binance Vision `BTCUSDT` USD-M `premiumIndexKlines/1m`:

- data SHA-256:
  `7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9`;
- source manifest SHA-256:
  `821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8`;
- range `[2020-01-01, 2026-07-01)`;
- missing exchange rows remain invalid and may not be filled.

The source contains only premium OHLC and source timestamps. BTC execution
price, return, funding and PnL are physically absent.

## Frozen path features

Evaluate only five-minute decision boundaries. For decision time `T`, use the
30 completed one-minute premium bars in `[T-30m, T)`. Convert premium OHLC to
basis points by multiplying by 10,000.

The path center is the median one-minute premium close in the 30 calendar days
ending strictly before `T-30m`. Require at least 95% source-valid rows in that
reference and all 30 current path rows source-valid.

For the current path:

- `path_range = sum(high_i - low_i)`;
- `efficiency = abs(last_close - first_open) / path_range`;
- `turns = count(delta_close_i * delta_close_(i-1) < 0)`;
- `up_excursion = max(high_i) - prior_center`;
- `down_excursion = prior_center - min(low_i)`;
- `max_excursion = max(up_excursion, down_excursion)`;
- `terminal_deviation = abs(last_close - prior_center)`.

Thresholds use analogous 30-minute windows ending every five minutes in the
strictly prior 30 calendar days. Shift the reference by six decisions so no
reference path overlaps the current path; require 95% valid reference paths.

Frozen quantiles:

- path range q90;
- efficiency q35;
- turns q70;
- max excursion q85;
- terminal deviation q40.

## Frozen signal

Require all of:

1. `path_range >= prior q90`;
2. `efficiency <= prior q35`;
3. `turns >= prior q70`;
4. `max_excursion >= prior q85`;
5. `terminal_deviation <= prior q40`;
6. exactly one excursion side reaches q85.

Direction:

- upper excursion alone reaches q85 → short BTC;
- lower excursion alone reaches q85 → long BTC;
- both or neither → no trade.

There is no parameter grid, direction flip selection or hold search.

## Execution boundary

The final one-minute source row is conservatively available at `T+1s`. Leave
the complete five-minute bucket `[T,T+5m)` empty, then enter at `T+10m`.
Hold exactly 30 minutes / six five-minute bars. Positions do not overlap.
There is no stop, take-profit, regime gate or dynamic exit.

## Sequential splits

| stage | start inclusive | end exclusive |
|---|---|---|
| train | 2020-02-01 | 2023-01-01 |
| test | 2023-01-01 | 2024-01-01 |
| eval | 2024-01-01 | 2026-07-01 |

All clocks and controls may be built from premium-only data. BTC outcomes must
later open train → test → eval, stopping permanently at the first failed gate.

## Outcome-blind support gates

- minimum clocks: train 120, test 30, eval 80;
- minimum per side: train 30, test 8, eval 20;
- each side at least 25% in every split;
- maximum one-month share: train 15%, test 25%, eval 15%;
- subperiod minimums:
  - 2020 Feb-Dec 20, 2021 30, 2022 30;
  - 2023 H1/H2 10 each;
  - 2024 25, 2025 25, 2026 H1 12.

Exact entry Jaccard must be at most 0.10 and the share of PSR clocks within 30
minutes of a comparator must be at most 0.20 against:

- the two previously selected single-bar premium intrabar-shape rules;
- CLBR-24, ICLA-60 and EBLR-60/30 source clocks.

Failing support or novelty rejects the candidate before any BTC outcome opens.

## Frozen controls

1. same clocks, direction flipped;
2. simple one-sided premium excursion without path/recenter requirements;
3. high-energy alternating path without terminal recenter;
4. prior PSI-2016 single-bar comparator;
5. prior PSI-8640 single-bar comparator;
6. primary delayed another five minutes;
7. future-premium placebo shifted 40 minutes earlier so the decisive path is
   noncausal and can only veto;
8. deterministic random clocks preserving split, month, side and count.

Controls must be frozen before train outcome parsing. No control may replace a
failed primary.

## Later strict outcome gate

Freeze a separate evaluator first. Base accounting is 1x notional, 6bp per
side; stress is 10bp per side. Use exact conservative funding boundaries,
full-calendar CAGR, and strict MDD from the global/pre-entry HWM, entry cost,
every held five-minute favorable-then-adverse OHLC envelope, settlement marks,
virtual adverse-mark exit cost and actual exit cost. Use a fixed-seed circular
stationary trade-block bootstrap.

Every opened stage requires positive base and stress return, strict MDD at most
15%, CAGR/strict-MDD at least 3, one-sided bootstrap `p <= 0.10`, and the frozen
minimum trade/side counts. The eval stage additionally reports 2024, 2025 and
2026 H1 separately. Because adjacent history has been inspected elsewhere in
the repository, any pass is retrospective evidence requiring forward shadow
validation, not production proof.
