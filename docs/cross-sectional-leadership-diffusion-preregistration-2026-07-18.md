# CLD-72 cross-sectional leadership diffusion — preregistration

## Decision boundary

- This artifact used signal-time incidence only; post-entry returns and PnL remain unopened.
- The mechanism is a transition from concentrated alt leadership to broad aligned participation, not a static breadth level.
- Support status: **PASS**.

## Frozen support cell

- Move quantile: `0.6`
- Prior residual-HHI quantile: `0.6`
- Maximum current/prior HHI ratio: `0.9`
- Minimum return participation: `0.8333333333333334`
- Minimum taker-flow alignment: `0.6666666666666666`
- Rank-turnover quantile: `0.5`
- Prior-leader decline quantile: `0.6`
- Non-overlapping events: `106` (41 long / 65 short)
- Quarter counts: `{'q1': 15, 'q2': 22, 'q3': 29, 'q4': 40}`

## Causal contract

- Every feature uses a completed UTC hour and strictly prior rolling thresholds.
- Entry is delayed one full five-minute bar after the hour boundary; hold is six hours.
- Signals are non-overlapping and contained inside each calendar quarter.
- No entry/later OHLC, funding cash flow, PnL, CAGR, MDD, win rate, or payoff was computed.
- A failed first strict 2023 evaluation retires this exact clock without threshold or direction repair.
