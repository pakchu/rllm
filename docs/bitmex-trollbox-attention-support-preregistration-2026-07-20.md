# TBASR-24 — text-free attention support preregistration

## Purpose and unopened data

This first TBASR-24 gate asks only whether unusual, multi-participant BitMEX
English Trollbox attention bursts occur often enough across train and test
calendars to justify opening private message text. It loads no message text,
LLM output, character-count proxy, BTC price, funding, execution path, return,
PnL, or 2023+ row. Failure rejects the candidate before semantic or market
outcomes are opened.

## Frozen source and causal clock

The official [BitMEX Chat endpoint](https://docs.bitmex.com/api-explorer/chat-get)
is downloaded chronologically for English channel `1`. Raw sender fields are
replaced locally by study-specific pseudonymous hashes. Raw text and aggregate
source data remain ignored; only a hash-bound manifest and research results are
committed.

An operational-only correction lowered the REST request pause from 0.55 to
0.25 seconds after observing download throughput and the official anonymous
limit of 180 requests per minute. At that point only resume-state page/message
totals and date progress had been monitored; no attention threshold, event
incidence, text semantic, or market outcome had been calculated. Pagination,
selected bytes, hashes, availability time, and every support rule are
unchanged.

Before completion, final aggregation was also changed from retaining every
five-minute participant counter in memory to a single-bucket streaming writer.
The writer validates and hashes the same increasing-ID private stream, emits
the same complete five-minute grid, and atomically replaces the aggregate only
after success. This changes resource usage and crash safety only, not source
membership or any research rule.

Increasing IDs can contain small raw-date regressions. The downloader therefore
assigns `available_date = cumulative_max(raw_date)` in increasing-ID order. A
late row is delayed, never advanced. Five-minute bars and this support clock use
only `available_date`. The aggregate exposes message count, unique participant
count, maximum participant share, and character count, but this stage explicitly
does not load character count.

## Exact singleton attention rule

For each completed five-minute availability bar:

1. identify its exact five-minute slot of the 7-day UTC week;
2. calculate the message-count `q0.98` and unique-participant `q0.95` from the
   same slot in the previous eight weeks, excluding the current week and
   requiring all eight prior slots;
3. require at least five messages and three independent participants;
4. require the busiest participant to contribute at most 50% of messages;
5. require current message and participant counts to meet both strictly prior
   thresholds; and
6. select the first qualifying bar, then require 12 five-minute bars (one hour)
   of separation before another selection.

There is no threshold grid. Eligibility is `[2020-07-01, 2023-01-01)`. The
observation is known at the bar end. If later stages pass, the earliest entry is
one additional completed five-minute bar later and the frozen hold is 24 bars
(two hours). No side exists at this stage.

## Frozen support gate

All conditions must pass:

- at least 240 events in 2020H2–2022;
- at least 150 in train (2020H2–2021), including 40 in 2020H2 and 80 in 2021;
- at least 90 in calendar 2022, including 35 in each half;
- at least 10 in each quarter from 2020Q3 through 2022Q4;
- events in at least 100 distinct weeks overall, 65 train weeks, and 35 test
  weeks; and
- no quarter above 18% of all events.

If any condition fails, no lookback, quantile, minimum count, participant cap,
cooldown, eligibility date, or calendar gate may be repaired after incidence.
No attention clock is written on failure.

## If and only if support passes

The passing timestamps are hash-frozen without counts, text, labels, or market
outcomes. A separate committed stage must then freeze the exact small Gemma2
revision, quantization, prompt, decoding, participant/message cap, synthetic
semantic controls, and directional-support gate before reading any message
semantics. `BULLISH`, `BEARISH`, and `UNCLEAR` are semantic compression only;
the LLM will not calculate returns, thresholds, position size, or trade policy.

Only after directional support passes may one strict pre-2023 market evaluator
be frozen and opened. This remains a candidate-level freeze on a repository
whose broader BTC history has already been researched, not a pristine global
human holdout.
