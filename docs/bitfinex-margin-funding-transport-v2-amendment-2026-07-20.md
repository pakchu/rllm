# Bitfinex margin-funding transport v2 amendment — 2026-07-20

## Reason

The frozen v1 transport assumed every official hourly observation would be
available by `floor(observation_time, 1h) + 15m`. The complete download reached
source validation but refused publication because some historical observation
timestamps occurred after that clock. Publishing v1 would therefore assign an
availability timestamp before the provider timestamp.

A timestamp-only audit of the cached official responses opened field `MTS`
only. It found 70,116 total symbol-hours and 100 observations after HH:15. No
funding amount, funding used, tenor, FRR, candidate incidence, comparator
incidence, BTC price, return, label, position, funding paid, or PnL value was
inspected to make this amendment. No source artifact was published by v1.

## Frozen repair

Transport v2 changes only `available_at`:

```text
available_at = max(
    floor(observation_time, 1 hour) + 15 minutes,
    ceil(observation_time, 5 minutes),
)
```

Thus ordinary HH:05 observations retain HH:15 availability. A delayed row is
not available before its official timestamp and is aligned to the first
five-minute boundary at or after that timestamp. The already-preregistered
entry delay remains a separate full five-minute bar after `available_at`.

All other frozen properties remain unchanged:

- endpoint, symbols, fields, raw rows, page order and page size;
- `[2020-01-01, 2024-01-01)` physical boundary;
- no market/outcome fields;
- exact source-hour joins with no fill;
- BFMWD feature algebra, directions, thresholds, variants, controls, hold,
  support gates, novelty gates and stop rules.

This is a fail-closed transport correction, not candidate repair. It is frozen
before any BFMWD feature value, event incidence, comparator incidence, BTC
outcome, or performance statistic is opened.
