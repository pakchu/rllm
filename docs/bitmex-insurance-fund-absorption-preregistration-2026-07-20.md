# IFAR-288 — outcome-blind support preregistration

## Frozen mechanism

`IFAR-288` tests whether a realized loss in BitMEX's BTC-denominated default
fund confirms that a completed directional BTC move contained costly forced
position transfer, after which part of the move reverses.

The official [insurance history endpoint](https://docs.bitmex.com/api-explorer/get-insurances)
provides a daily `XBt` `walletBalance`. The current
[Exchange Rules](https://www.bitmex.com/legal/exchange-rules) state that profit
or loss from liquidation trades is booked to the insurance fund. This is a
venue-level stress proxy, not a complete liquidation tape and not proof that
every change came from `XBTUSD`.

## Exact causal clock

- source snapshot: daily `12:00 UTC` `XBt` wallet balance;
- directional observation: close of the completed Binance BTCUSDT
  `11:55–12:00 UTC` five-minute bar versus the prior daily snapshot close;
- publication embargo: one full calendar day after the insurance timestamp;
- decision: `D+1 12:00 UTC`;
- latency: one complete five-minute bar;
- entry: `D+1 12:05 UTC`;
- hold: 288 five-minute bars, exactly 24 hours;
- live rule: fail closed unless the expected insurance timestamp was actually
  observed before the decision.

The same-day historical row is never used for a same-day decision. The source
support builder parses no Binance price after the snapshot close, no funding,
no execution bar, and no post-decision outcome.

## Exact singleton rule

For each daily source row:

1. `fund_return = log(balance_D / balance_D-1)`;
2. `fund_loss = max(-fund_return, 0)`;
3. compare fund loss with the median positive loss among the previous 365
   source rows, excluding the current row and requiring 20 prior losses;
4. compute the completed daily BTC return ending at the same 12:00 snapshot;
5. compare its absolute value with the median absolute return among the
   previous 126 observations, excluding the current row and requiring 90;
6. signal only when the fund return is negative and both magnitudes meet their
   strictly prior medians; and
7. trade opposite the completed BTC move.

There is no threshold, embargo, side, or hold grid. Entry eligibility is
`[2020-07-01, 2023-01-01)`, fixed before source incidence so the price baseline
does not create an accidental first-quarter support failure.

## Frozen support gate

Support is outcome-blind and must satisfy all conditions:

- at least 50 events in 2020H2–2022;
- at least 30 in train (2020H2–2021), including at least 8 in 2020H2 and 20 in
  2021;
- at least 20 in calendar 2022, including at least 8 in each half;
- at least 2 in each of the ten eligible quarters;
- at least 25% long and 25% short in all/train/test separately; and
- no quarter above 20% of all events.

Failure rejects IFAR before any post-decision return. The gate may not be
repaired after incidence by changing the source interval, loss threshold,
price threshold, embargo, side, hold, or calendar requirements.

## Source and data-use boundary

The private raw insurance prefix is fixed to `XBt` daily rows from
`2018-01-01` through `2022-12-31`. It is downloaded only after this code and
document are committed. The raw file remains ignored; the committed source
manifest records the request contract, range audit, and SHA-256.

The already audited Binance five-minute prefix is physically pre-2023. Source
support reads only one completed price per day. The repository does not grant
a license to redistribute BitMEX data; current
[Terms of Service](https://www.bitmex.com/terms) remain applicable.

## Evaluation boundary if support passes

Before any outcome is loaded, one strict evaluator must be hash-frozen. It
uses train `[2020-07-01, 2022-01-01)` and test calendar 2022, 0.5x notional,
6 bp base cost per side, 10 bp stress cost per side, exact realized funding,
and full five-minute held paths.

Train and test must each have positive absolute return, CAGR/strict-MDD at
least 3, strict MDD at most 15%, positive stress-cost and one-bar-delayed
return, mean gross underlying edge at least 20 bp, and weekly clustered
sign-flip `p <= 0.10`. Controls are exact side flip, the same price rule
without insurance loss, a seven-day-stale insurance gate, and within-year
permutation of fund-loss magnitudes.

Only a complete pre-2023 pass can request 2023+ insurance data. Sealed years
open once in order: 2023, 2024, 2025, then 2026 YTD. Stop on the first failure.

This is a candidate-level freeze on a repository whose broader market history
has already been researched; it is not a pristine global human holdout.
