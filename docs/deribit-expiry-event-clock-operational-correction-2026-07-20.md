# Deribit expiry event-clock operational correction — 2026-07-20

## Trigger

The frozen v2 source loader failed closed before writing an aggregate because
one historical BTC option delivery timestamp was more than five seconds after
the scheduled 08:00 UTC expiry. No Binance market row, funding row,
post-delivery return, candidate event count, signal threshold, trade, or PnL
was loaded.

A source-only clock diagnostic repeated the same bounded pre-2023 Deribit
pagination and inspected only instrument expiry codes and delivery timestamps.
It observed:

- 31 response pages;
- 30,357 BTC option delivery rows in 2019–2022;
- 30,332 rows stamped at 08:00 UTC; and
- 25 rows, all belonging to `BTC-27AUG20-*`, stamped at
  `2020-08-27 09:18:27.676 UTC`.

The maximum delay from the scheduled 08:00 clock was 4,707.676 seconds. These
are source-quality statistics, not outcome or candidate-incidence statistics.

## Defect

The v2 parser conflated two different clocks:

1. the scheduled expiry encoded in the instrument name; and
2. the API row timestamp for the actual delivery event.

Rejecting the late event is fail-closed, but using scheduled expiry plus 65
minutes would be worse: on the anomalous date it would assign historical
availability before the reported delivery event.

## Outcome-blind correction

Loader v3 keeps both clocks:

- `expiry_time`: scheduled 08:00 UTC from the instrument name;
- `delivery_event_time`: the latest common API delivery timestamp for all rows
  in that expiry; and
- `source_observation_earliest`: `delivery_event_time + 65 minutes`.

Rows must not precede scheduled 08:00, must remain on the same UTC calendar
date as the instrument expiry, and must agree with each other within five
seconds. The latest row timestamp is deliberately conservative. The live rule
still requires two identical canonical delivery sets five minutes apart after
the 60-minute embargo; late data delays or cancels and is never backdated.

No source-support threshold, 365-day reference, q0.50/q0.70 rule, side map,
entry latency, six-hour hold, calendar split, performance target, or control is
changed. The original preregistration commit remains the evidence that these
rules were fixed before the clock diagnostic. A successor artifact must bind
this correction and explicitly disclose that source-clock diagnostics were
opened while candidate incidence and every market outcome remained unopened.
