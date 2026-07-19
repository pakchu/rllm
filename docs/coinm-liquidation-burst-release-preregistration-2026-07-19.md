# CLBR-24: COIN-M liquidation burst release preregistration — 2026-07-19

## Hypothesis

`CLBR-24` trades a direct forced-flow exhaustion mechanism rather than another
price, funding, OI, REX, or regime gate.  A large one-sided COIN-M liquidation
snapshot burst is mechanical flow.  If the following completed five-minute bar
shows an immediate collapse in that flow without a meaningful opposite-side
cascade, the mechanical pressure is treated as exhausted and faded.

The source is Binance's censored 1Hz force-order snapshot stream, not a complete
liquidation tape.  Its construction and limitations are frozen in
`docs/binance-coinm-liquidation-snapshot-source-design-2026-07-19.md`.

## Single fixed source rule

1. On each source-valid bar, compute the 97.5th percentile of **positive**
   liquidation-snapshot notional over the strictly prior 14 calendar days.
   Require at least 200 prior positive bars.
2. A burst needs at least three snapshots, notional at or above that threshold,
   and absolute side imbalance at least 0.8.
3. The immediately following bar is a release only when total snapshot notional
   falls to at most 25% of the burst and opposite-side snapshot notional is at
   most 10% of the burst.
4. Fade the forced side: dominant `SELL`/long-liquidation flow enters long;
   dominant `BUY`/short-liquidation flow enters short.
5. Feature availability is release-bar end plus one second.  Entry is the first
   USD-M `BTCUSDT` five-minute open at or after availability, normally one full
   bar later.
6. Hold 24 five-minute bars (two hours), with no overlapping positions.
7. Structural stop is 25bp outside the burst's COIN-M snapshot-price extreme.
   Long uses the burst minimum; short uses the burst maximum.  If the USD-M
   entry has already crossed the stop, the event is invalid and skipped.

There is one candidate, no threshold battery, no LLM/model selection, and no
repair after train/test/eval outcomes are opened.

## Frozen chronological split

| stage | start inclusive | end exclusive | source support floor |
|---|---|---|---:|
| train | 2023-06-25 | 2023-10-15 | 40 |
| test | 2023-10-15 | 2024-04-15 | 100 |
| eval | 2024-04-15 | 2024-10-15 | 100 |

Test and eval are each 183 days.  Clocks for all splits are generated from the
frozen liquidation source before any executable market path is loaded.

Outcome-blind support passed exactly as frozen:

| stage | total | long | short |
|---|---:|---:|---:|
| train | 40 | 29 | 11 |
| test | 128 | 82 | 46 |
| eval | 109 | 77 | 32 |

- clock SHA-256:
  `df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0`;
- support artifact SHA-256:
  `362c1b45fd52b278e2c7f3f06214812fd02b5a1a311aae716ad3c8621852ead3`;
- support manifest hash:
  `114071968cfba0bb40cf7fa44b283a3b0312d5de6551dec549999d80b8cbb27b`.

## Strict execution and promotion contract

- executable market: official Binance USD-M `BTCUSDT` 5m OHLC;
- leverage for discovery: 1.0x;
- base cost: 6bp per side; stress: 12bp per side;
- realized funding applied at exact settlement timestamps;
- stops use adverse gap handling and held-bar high/low;
- strict MDD includes pre-entry/global HWM and every held-bar adverse extreme;
- CAGR uses the complete calendar window including idle periods;
- every report includes absolute return, CAGR, strict MDD, CAGR/MDD, trades,
  long/short counts, win rate, and exposure time.

Train must be positive with CAGR/strict-MDD at least 2.0, strict MDD at most
15%, at least 30 executable trades, and both sides represented.  Failure seals
test.  Test must independently meet the same return/MDD gates, at least 60
trades, positive doubled-cost return, and a one-sided stationary-block bootstrap
`p <= 0.10`; otherwise eval stays sealed.  Eval promotion requires positive
return, strict MDD at most 15%, CAGR/strict-MDD at least 3.0, at least 60 trades,
both sides represented, and positive doubled-cost return.

## Integrity boundary

The archive ends on 2024-10-14, so even a passing eval is only a historical
candidate for forward live shadow, not production proof.  The repository has
seen 2024 market outcomes in other research; this experiment prevents local
reranking but cannot claim a pristine human holdout.  No result may be promoted
without fresh, versioned live force-order snapshots and execution parity.
