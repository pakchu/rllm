# UCBR-12 outcome-blind preregistration — 2026-07-20

## Boundary

This document freezes one **USDT Collateral Breadth Relay** policy before
calculating its real incidence. No BTCUSDT perpetual price, funding, future
return, label, PnL, absolute return, CAGR, or MDD has been opened.

UCBR-12 cannot be repaired under the same name after source support or any
outcome is observed. Threshold, breadth, direction, hold, controls, and gates
below are singleton choices, not a grid.

## Strictly prior issuer state

For each completed hourly direct pair `USDCUSDT`, `TUSDUSDT`, `USDPUSDT`, and
`FDUSDUSDT`, use `log(close)` only when that current hour has positive base
volume, quote notional, and trade count. For each issuer separately:

```text
history = at most 720 calendar hours ending at t-1
minimum = 672 valid historical observations
scale   = (prior q75 - prior q25) / 1.349
z       = (current log close - prior median) / scale
```

All rolling quantiles use linear interpolation and `shift(1)`. Nonpositive
scale, insufficient valid history, or a current invalid member yields no z for
that issuer. No cross-issuer raw-level pooling is allowed.

## Singleton clock and direction

Count current issuers with `z >= +1.25` and with `z <= -1.25`. The state is
active when either count is at least three. Only a `false -> true` transition
emits an event.

```text
source_sign = +1 for broad alternative-stablecoin strength / USDT weakness
source_sign = -1 for broad alternative-stablecoin weakness / USDT strength
trade_side  = -source_sign
```

Thus broad USDT strength maps `LONG BTCUSDT`; broad USDT weakness maps `SHORT`.
The rule requires three same-sign strong issuers, not merely a basket median.

## Execution clock

- source hour `[h,h+1h)` is final at `h+1h`;
- entry is the BTCUSDT USD-M perpetual open at `h+1h+5m`;
- exit is the scheduled open exactly 144 five-minute bars later (12 hours);
- candidates are processed chronologically, and entry must be at or after the
  prior accepted exit within each clock;
- exit must be no later than `2024-01-01 00:00 UTC` in initial support.

## Source-only controls

Controls keep the same normalization, onset, and scheduler:

- `all_four`: all four issuers must be current-valid, strong, and same-sign;
- four leave-one-issuer-out clocks: all three remaining fixed issuers must be
  current-valid, strong, and same-sign;
- `median_only`: at least three valid z-scores and `abs(median z) >= 1.25`,
  without the three-strong-members requirement;
- `stale_1h`: delay the complete primary source state and side by one hour,
  with provenance retained at the original feature hour.

Controls are falsification diagnostics and cannot replace the primary after
support is observed.

## Frozen support gates

The primary initial-source clock must satisfy every condition:

- at least 30 accepted events;
- at least 5 accepted events in each of September–December 2023;
- LONG and SHORT each at least 30%;
- no UTC entry month more than 45%;
- against SDDR `primary` and SQFD `primary`, `no_usdt_lag`, and
  `no_participation`: exact-entry Jaccard at most 0.10 and maximum
  bidirectional event containment within plus/minus six hours at most 0.35.

The complete source must also retain at least three current-valid members in
every hour. Failure of any check rejects UCBR-12 before outcomes. Real event
counts and overlap values have not been calculated at this freeze.

## Conditional later evaluator

Only a support pass may authorize a separately implemented, tested, committed,
and hash-frozen evaluator. It must open stages sequentially—train 2023, test
2024, evaluation 2025, final 2026H1—and stop at the first failure.

Future accounting is fixed at 0.5 leverage; 6 bp base and 10 bp stress cost per
notional side; exact next-open execution; realized funding; full-calendar
CAGR; and strict position-path MDD. Required gates include at least 30 trades
per stage, positive absolute return, positive predeclared half-periods, base
`CAGR/MDD >= 3.0`, stress ratio at least 2.5, strict MDD at most 15%, mean gross
move at least 25 bp, weekly clustered sign-flip `p <= 0.10`, and at least 0.25
ratio advantage over direction, random-side, latency, issuer-removal,
median-only, and matched BTC-direction controls.

Gemma/RLLM remains outside the source clock. It may later abstain or route risk
only after deterministic gross edge exists; it cannot repair missing issuer
data, timing, direction, breadth, or threshold.
