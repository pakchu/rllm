# AFDR-864 source-support rejection

## Decision

`AFDR-864` is permanently retired as `REJECT_NO_REPAIR` at the source-support
stage. No BTC market price, settlement mark, return, PnL, 2024+ row, economic
gate, LLM rescue, or RL rescue was opened.

Frozen result:

- result SHA-256:
  `ea95077fb16ed367f06f204c729525e699ff1af89bd8af63df1b7acd7e656c09`
- clock SHA-256:
  `d688c4e4d845cf0a4daaf14b7ecfa6bb4c990bde59602eb9d55ffc7088c6d7b9`
- result manifest:
  `3084691e08a90325bfcb3e0d60e102427afdcc34ec1fda46195864c8161e464d`

## Fatal support result

The primary policy emitted only 14 accepted train events and 13 accepted
selection events, versus frozen minima of 50 and 25. Annual train counts were
8 in 2021 and 6 in 2022, versus 20 required in each year. Selection-half
counts were 7 and 6, versus 10 required in each half.

Side support also failed: train had 10 LONG and 4 SHORT events; selection had
7 LONG and 6 SHORT events. Several month, weekday, and rolling-30-day
concentration gates failed. The mechanism is therefore too sparse and too
concentrated for a statistically defensible economic test.

## Novelty result

The members that could be parsed were highly distinct from AFDR-864. Their
exact entry Jaccard was zero and signed-exposure correlations were small.
However, frozen comparator contracts also failed closed for timezone-less
timestamps and textual side encodings in several legacy artifacts. This is an
additional novelty-stage failure, not the reason to rescue the candidate:
AFDR-864 already fails the source-support gates by a wide margin.

## Stopping rule

Do not change AFDR-864 signs, thresholds, lookback, hold, onset rule, source
availability, or comparator parsing after observing this result. Do not open
its economic outcomes. Future research must start from a separately named and
preregistered mechanism with materially broader causal support.
