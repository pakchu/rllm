# AFDR-864 outcome-blind preregistration

## Frozen identity

This document preregisters **AFDR-864 — Address–Funding Divergence Relay**
exactly as defined in the mechanism decision. The candidate is a singleton,
not a search grid. Its network feature, completed-funding window, empirical
rank rule, tails, direction, latency, hold, support floors, controls, novelty
comparators, economic gates, and stopping rule are immutable after this
artifact is written.

The source files are already hash-frozen from earlier independent work, but no
AFDR incidence or matching BTC outcome has been calculated at this freeze.
Broad historical BTC results are research-seen; therefore historical stages
are falsification/development evidence and not a pristine global holdout.

## Bound files and source allowlists

The machine-readable preregistration binds by SHA-256:

- the mechanism decision and this document;
- the Coin Metrics source, source manifest, and immutable downloader;
- the Binance completed-funding source, source manifest, and freezer;
- NTB-7, CVTR-1, ORFR-1, primary FLCC-1, and the sanitized
  prior-microstructure comparator bundle, all dated no later than 2026-07-20;
- BFMWD-144 source-support clocks; and
- DLPD-12 source-support clocks.

The preregistration hashes bytes only. It parses zero source or comparator data
rows and opens no BTC market, return, PnL, or post-2023 row.

The later source-support evaluator may read exactly:

```text
Coin Metrics:
  observation_date,available_at,AdrBalCnt,AdrActCnt

Binance funding signal:
  funding_time_ms,funding_time_utc,symbol,funding_rate
```

It must validate the exact known physical funding header with a zero-row read,
reject any column outside that frozen header, and then parse only the four
signal columns. It must not parse `settlement_mark_price`, mark-price
timestamps, funding-offset, or mark-source values during source support.

## Frozen policy

- address transforms: exact seven-calendar-day log changes in `AdrBalCnt` and
  `AdrActCnt`;
- address feature availability: max(current, exact-lag availability); a value
  later than current availability is reference-only after that time and can
  never emit a backdated signal;
- funding transform: sum of the nine most recent already-available funding
  rates, requiring consecutive canonical eight-hour slots and a newest-event
  age no greater than eight hours; each reported timestamp must lie in the
  first 60 seconds of its canonical slot and agree with its millisecond form;
- normalization: tie-midrank empirical percentile over strictly prior finite
  observations in the preceding 365 calendar days, minimum 180;
- network rank: equal-weight mean of balance-growth and activity-growth ranks;
- LONG: network rank at least 0.75 and funding rank at most 0.25;
- SHORT: network rank at most 0.25 and funding rank at least 0.75;
- event: valid-FLAT-to-nonzero onset only on immediately adjacent daily rows;
  a missing/stale/invalid predecessor cannot act as FLAT;
- entry: one complete five-minute latency bar after the first five-minute open
  at or after decision availability;
- hold: 864 five-minute bars;
- leverage: 0.5x;
- greedy global non-overlap and complete split containment.

## Frozen windows and gates

- warm-up: 2019–2020;
- train: calendar 2021–2022;
- sealed selection: calendar 2023;
- 2024+ remains closed.

The exact support, concentration, novelty algorithms, control formulas,
weekly-cluster randomization, strict-MDD accounting, and economic gates are
serialized in the preregistration JSON and match the mechanism decision. Any
failed gate retires AFDR-864 without repair. Controls are diagnostic only and
cannot become replacement candidates.

## One-way sequence

1. Commit this preregistration before AFDR source values are combined.
2. Implement, test, commit, and hash-freeze a source-only evaluator before
   deriving incidence.
3. Stop permanently on source-support or novelty failure.
4. On support pass only, implement, test, commit, and hash-freeze a strict
   economic evaluator.
5. Open train 2021–2022 market/settlement-mark transport first. The already
   source-qualified 2023 signal clock may remain hash-bound, but 2023 BTC bars
   and funding settlement-mark values remain physically absent until train
   passes.
6. Stop permanently on train failure. Open 2023 only after an exact pass.
7. No 2024+ period is opened until the preceding stage authorizes it.

LLM/RL is downstream of deterministic evidence. It may abstain or size only
under a new freeze after deterministic train and selection pass; it cannot
retime, reverse, or repair AFDR.
