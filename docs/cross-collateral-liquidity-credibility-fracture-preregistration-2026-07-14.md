# PDF-10 preregistration — 2026-07-14

## Decision and claim boundary

**PDF-10 (Phantom Display Fracture, 10-minute hold)** tests whether displayed
percentage-band depth points one way while the within-bar firmness of that
depth points the other way. The action follows firmness and fades display.

“Phantom” is shorthand, not a spoofing claim. Binance Vision provides nominal
30-second cumulative snapshots, not order IDs or cancel/replace messages.
Accordingly, this experiment may claim only a **display/firmness divergence**
derived from net band-mass change and instability proxies.

This preregistration, its support program, and the frozen input panel contain
no future BTC return, PnL, CAGR, MDD, 2024+, or label access. Calendar 2023 is
development/selection data. It is not clean OOS.

## Why this is a new mechanism

The failed CLV experiment traded a price-shock/liquidity-response impulse. The
failed CCLH experiment traded a 12-hour persistent cross-contract median-depth
geometry state. PDF-10 instead trades a short-lived contradiction between:

1. the direction of displayed cumulative depth; and
2. signed net replenishment after penalizing within-bar instability.

PDF-10 does not use a prior price shock, taker flow, CCLH state, or either
failed strategy as an entry gate. Its intended economic horizon is minutes,
matching the archive's snapshot cadence and the microstructure literature
summarized in the credibility-data audit.

## Frozen data boundary

- input manifest:
  `results/binance_cross_collateral_book_credibility_btc_2023_manifest.json`
- manifest SHA256:
  `f530f472765c8cb56bf564efd346c734e2404e072b87e2ff8dc3b84e303c30f7`
- input panel:
  `data/binance_cross_collateral_book_credibility_btc_2023/BTC_cross_collateral_book_credibility_5m_2023.csv.gz`
- panel SHA256:
  `45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429`
- inclusive start: `2023-01-01 00:00:00` UTC
- exclusive end: `2024-01-01 00:00:00` UTC
- required markets: USD-M `BTCUSDT` and COIN-M `BTCUSD_PERP`
- required bands: cumulative bid/ask depth at every `1..5%` distance
- required path statistics: `log_net`, `log_mad`, and `log_step`
- current and previous confirmation bars must both be `source_complete`
- no fill, interpolation, nearest-time join, or post-2023 row is allowed

The panel and its limitations are audited in
`docs/binance-cross-collateral-book-credibility-data-audit-2026-07-14.md`.
Official source documentation is linked there, including:

- <https://github.com/binance/binance-public-data/issues/437>
- <https://github.com/binance/binance-public-data/issues/447>
- <https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

## Causal normalization

For every scalar input series `x_t`, use only prior clean observations:

```text
prior_t  = x shifted by one bar
center_t = rolling_median(prior_t, window=8640, min_periods=2016)
mad_t    = rolling_median(abs(prior_t - center_t),
                          window=8640, min_periods=2016)
Z(x_t)   = clip((x_t - center_t) / (1.4826 * mad_t), -12, 12)
```

Zero, missing, or non-finite scale fails closed. A value at `t+1` can never
change a feature or action at `t`.

## Frozen signal

Let venue `v` be `um` or `cm`, and distance `k` be `1..5`. Suffix `m` denotes
the bid-side negative percentage band and `p` the ask-side positive band.

### Displayed direction

```text
Disp_vk,t = Z(log(depth_v,mk,t / depth_v,pk,t))
Disp_v,t  = mean_k(Disp_vk,t)
Disp_t    = 0.5 * (Disp_um,t + Disp_cm,t)
```

Positive `Disp` is unusually bid-heavy displayed depth; negative is unusually
ask-heavy displayed depth. The five cumulative bands are aggregated, not
treated as five independent votes.

### Firmness direction

Only `log_net` is signed. `log_mad` and `log_step` are unsigned instability
measures and may only penalize a side:

```text
Net_vk,t = Z(log_net_v,mk,t) - Z(log_net_v,pk,t)

Churn_vk,t =
    0.5 * (Z(log_mad_v,mk,t) + Z(log_step_v,mk,t))
  - 0.5 * (Z(log_mad_v,pk,t) + Z(log_step_v,pk,t))

Cred_v,t = mean_k(Net_vk,t - 0.5 * Churn_vk,t)
Cred_t   = 0.5 * (Cred_um,t + Cred_cm,t)
```

Positive `Cred` means bid-side band mass is net-replenishing and/or less
unstable than the ask side; negative means the reverse. These are snapshot
proxies, not literal order additions or cancellations.

### Raw state and confirmation

At completed bar `t`, bullish raw state `+1` requires all of:

```text
source_complete_t
all required standardized values finite
Cred_um,t > 0 and Cred_cm,t > 0
Cred_t >= +0.75
Disp_t <= -1.00
```

Bearish raw state `-1` is the exact sign reverse:

```text
Cred_um,t < 0 and Cred_cm,t < 0
Cred_t <= -0.75
Disp_t >= +1.00
```

Otherwise raw state is zero. A bar is tradeable only when `raw_t` and
`raw_(t-1)` are the same nonzero side. Every such confirmed evidence bar is a
candidate. The non-overlap scheduler may therefore re-enter at a prior exit
open while an unusually long divergence run persists. This behavior is
explicitly part of v1 and may not be changed after returns are opened.

## Clock and support scheduler

For a signal row indexed `t`:

- information: completed five-minute bar `t` only;
- entry: open of row `t+1`;
- held bars: `t+1` and `t+2`;
- exit: open of row `t+3`;
- position: `+1` long or `-1` short according to `Cred`;
- scheduler: reset flat at each calendar-quarter boundary, then greedily
  accept chronological candidates within that quarter when entry is at or
  after the prior scheduled exit; every trade must exit inside its quarter.

The frozen annual event clock is the concatenation of those four
quarter-contained schedules. Quarter counts, H1/H2 counts, side shares,
independence overlap, and the later return evaluator must all derive from this
same clock; no separate half-year rescheduling is allowed.

Future credibility availability cannot cancel a scheduled trade. The eventual
return evaluator must require a complete execution-price path independently.

## Outcome-blind support gates

There is no threshold grid and no support repair. Reject PDF-10 without opening
returns unless all gates pass:

- at least 500 non-overlapping scheduled trades;
- at least 180 trades in each half of 2023;
- at least 75 trades in each calendar quarter;
- long and short share each at least 35%;
- no quarter contributes more than 40% of trades;
- all signal inputs finite and both confirmation bars source-complete.

## Independence gates

CCLH is replayed from the exact depth columns in the new panel; no price file is
loaded. PDF-10 is rejected as insufficiently distinct if any condition fails:

- CCLH's 167 scheduled positions and sides must reproduce canonical SHA256
  `e90079d95b111f95ce64459c42d17e4286636a1a2854ed948e8ada497a13dfa7`;
- greedy event Jaccard against CCLH within `±2` bars must be at most `0.15`;
- event Jaccard within `±12` bars must be at most `0.30`;
- maximum absolute Spearman correlation of PDF `Cred`/`Disp` with CCLH
  cross-pressure/cross-elasticity must be at most `0.60`.

CLV overlap is not a support gate because replaying CLV requires its historical
price-shock clock. PDF-10 support deliberately loads no prices or returns.

## Frozen 2023 return evaluation, if support passes

The evaluator must be written, tested, committed, and separately hash-frozen
before any PDF-10 return is read. It must replay the support artifact exactly.

Execution contract:

- 0.5x exposure;
- fee 5 bp plus slippage 1 bp per notional side;
- account cost 3 bp at entry and 3 bp at exit;
- trade multiplier
  `(1 - 0.0003) * (1 + 0.5 * raw_return) * (1 - 0.0003)`;
- CAGR uses the complete split clock, including idle cash;
- strict MDD includes pre-entry equity and every held-bar adverse extreme;
- favorable extreme is applied before adverse extreme within a held bar;
- exit-bar later high/low is excluded because exit occurs at its open.

Qualification gates, with no parameter change:

- H1 and H2 absolute return both positive;
- H1 and H2 CAGR / strict MDD both at least `3.0`;
- H1 and H2 strict MDD both at most `15%`;
- every quarter absolute return positive and retains at least 75 trades;
- one-sided weekly-cluster sign-flip p-value below `0.10` in both halves,
  using 100,000 permutations and seed `20260714`;
- PDF-10 must beat exact reverse, always-long, always-short, causal 5m price
  momentum side, and frozen sign-permutation controls on the minimum H1/H2
  CAGR/MDD, all evaluated on the reserved PDF-10 candidate clock.

Firmness-only and display-only clocks may be reported only as structural
diagnostics. They cannot replace PDF-10 after outcomes open.

## OOS stopping rule

If any support, independence, or 2023 return gate fails, PDF-10 v1 is rejected
and 2024+ credibility data remains sealed. If every gate passes, the same
builder and frozen signal may then construct calendar 2024 for first genuine
OOS evaluation. No Gemma, RL, or LLM policy is introduced before deterministic
2024 qualification.
