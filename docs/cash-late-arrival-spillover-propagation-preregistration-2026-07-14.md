# CLASP-24 preregistration — 2026-07-14

## Status and evidence boundary

**CLASP-specific support only; CLASP outcomes unopened.** This document freezes
the mechanism, direction, hold, causal baselines, support grid, controls, and
return gate before any CLASP post-entry return or held path is calculated.
Calendar 2024+ remains sealed.

- name: **CLASP-24 — Cash Late-Arrival Spillover Propagation**
- feature source: frozen Binance Spot/USD-M one-minute ordering aggregated into
  completed five-minute descriptors;
- inspected horizon: strictly before `2024-01-01`;
- signal availability: only after completed five-minute bar `t`;
- entry: next Binance USD-M five-minute open;
- exit: fixed USD-M open 24 bars / two hours later.

Feature-incidence counts, side balance, era balance, and frozen-clock overlap
were inspected without opening a future price, return, PnL, CAGR, or MDD. Those
outcome-blind checks set the support floors below. They did not select direction,
hold, or a return threshold.

## Economic hypothesis

Most cross-venue leadership rules ask which venue moved first. CLASP asks a
different question: **did an unusually large and efficient Spot impulse arrive
late enough in the completed bar that USD-M showed only a partial response and
closed with a remaining catch-up debt?**

The signal requires five independently interpretable pieces:

1. Spot activity, signed-flow intensity, and price-path intensity all occur
   later inside the five-minute bar than their USD-M counterparts;
2. Spot's net move is more path-efficient than USD-M's move;
3. the average Spot ticket is unusually large relative to the contemporaneous
   USD-M average ticket and its strictly prior local baseline;
4. the USD-M/Spot basis still says Spot moved farther in the Spot-flow side;
5. within-bar Spot-flow-to-next-minute-USD-M response is positive and stronger
   than both the reverse venue ordering and the reverse-time reconstruction.

“Large ticket,” “response,” and “catch-up debt” are inferences from public
aggregates. They do not claim trader identity, private information, or observed
inventory.

## Frozen formula on completed bar `t`

Let `d = sign(spot_flow_fraction_t)`. Require a clean source row and the frozen
24-bar post-defect quarantine. Define directed acceptance:

```text
spot_accept = d * spot_log_return_5m * 10000 > 0
um_accept   = d * um_log_return_5m   * 10000 >= 0
basis_lag   = -d * basis_change_bp > 0
```

The basis sign is symmetric. If Spot rises farther than USD-M, basis falls; if
Spot falls farther, basis rises. Both produce positive `basis_lag` in direction
`d`.

### Late-arrival shape

```text
closing_latency = min(
  spot_activity_time_centroid - um_activity_time_centroid,
  spot_flow_time_centroid     - um_flow_time_centroid,
  spot_return_time_centroid   - um_return_time_centroid,
)
```

Require `closing_latency > 0`. Each centroid is computed only from the five
one-minute observations inside the already completed bar. This is the opposite
temporal orientation from CATCH-12, which requires Spot activity before USD-M.

### Path efficiency advantage

```text
spot_eff = abs(spot_log_return_5m) * 10000 / spot_abs_path_return_bp
um_eff   = abs(um_log_return_5m)   * 10000 / um_abs_path_return_bp
efficiency_advantage = spot_eff - um_eff
```

Require `efficiency_advantage > 0`. This distinguishes a coherent cash impulse
from a high-path-length but low-net noisy bar.

### Relative average-ticket surprise

```text
ticket_ratio = log(
  (spot_quote_notional / spot_trade_count)
  / (um_quote_notional / um_trade_count)
)

ticket_baseline_t = median(ticket_ratio.shift(1))
                    over 8,640 bars, minimum 2,016 clean observations

ticket_surprise = ticket_ratio - ticket_baseline_t
```

Require `ticket_surprise > 0`. The absolute Spot ticket need not exceed the
USD-M ticket. Only a positive deviation from the strictly prior local relative
baseline is required, avoiding a permanent venue-scale rule.

### Forward response advantage

```text
spot_forward = min(
  spot_to_um_lagged_flow_response_bp,
  lagged_flow_response_diff_bp,
)

spot_reverse = min(
  reverse_spot_to_um_lagged_flow_response_bp,
  reverse_lagged_flow_response_diff_bp,
)

response_advantage = spot_forward - max(spot_reverse, 0)
```

Require `response_advantage > 0`. This simultaneously requires a positive
Spot-to-USD-M next-minute response, superiority over USD-M-to-Spot response,
and superiority over a later-flow-to-earlier-return reconstruction.

### Composite score and causal support grid

For eligible rows only:

```text
score = geometric_mean(
  closing_latency,
  efficiency_advantage,
  log1p(basis_lag),
  log1p(ticket_surprise),
  log1p(response_advantage),
)
```

The score threshold is the quantile of prior eligible events only:

```text
threshold_q(t) = rolling_quantile_q(
  score.where(eligible).shift(1),
  window=17,280 bars,
  min_periods=64 eligible events,
)
```

The frozen support grid is `0.50`, `0.65`, `0.75`, `0.80`, `0.85`. Select the
highest quantile passing every frozen support and novelty floor; otherwise
reject before returns. No return is available to this choice.

The action is `side = d`, entry is the next USD-M open, and the hold is fixed at
24 bars. There is no direction grid and no hold grid.

## Why this is not a prior alpha re-sign

- **CSPR-12** uses Spot sponsorship against same-bar perp rejection and a
  different Spot/agg-trade frame; CLASP requires late cash arrival, relative
  ticket surprise, path-efficiency advantage, and flow-response magnitude.
- **RIFT-96** is a long-only two-bar refill/pressure continuation state; CLASP
  is bidirectional and event-local.
- **CATCH-12** requires Spot activity earlier than USD-M and an already observed
  directional handoff; CLASP requires all three Spot centroids later and trades
  the uncompleted propagation state.
- **LURI-48** uses a 36-bar inferred USD-M inventory formation and reverse
  release; CLASP has no trailing inventory state and trades with the current
  late Spot impulse.

## Frozen controls

Every non-flip control reserves its own global non-overlapping 24-bar clock.
Component ablations are diagnostics and return-comparison controls, not support
gates: a composite of weak, complementary components is allowed.

1. `direction_flip`: exact primary clock, side `-d`.
2. `no_timing`: remove all three late-arrival conditions and their score term.
3. `no_ticket_surprise`: remove relative average-ticket surprise.
4. `no_efficiency_advantage`: remove Spot-over-USD-M path efficiency.
5. `no_basis_lag`: remove the remaining cross-venue price-gap condition.
6. `no_response_advantage`: remove forward-over-reverse response magnitude.
7. `early_cash`: require the full mirror of all three centroid signs while
   retaining Spot direction and every other primary component.
8. `venue_swap`: fully mirror venue direction, timing, efficiency, ticket,
   basis, and forward/reverse response signs.
9. `aggregate_only`: retain same-direction Spot/USD-M acceptance and basis lag,
   scoring only Spot flow magnitude, flow coherence, and basis lag.
10. `stale_1h`: primary event and side shifted 12 completed bars.
11. `stale_24h`: primary event and side shifted 288 completed bars.
12. `signal_delay_1bar`: primary event and side shifted one completed bar.
13. frozen CSPR-12, RIFT-96, CATCH-12, and LURI-48 primary clocks for overlap.

## Frozen support and novelty floors

- non-overlapping total at least `600`;
- each calendar year 2020–2023 at least `130`;
- each 2023 half at least `65` and each 2023 quarter at least `28`;
- each side at least 40% overall and at least `50` events per year;
- at least `42` months with at least `5` scheduled events;
- `early_cash` and `venue_swap`, raw and independently scheduled: Jaccard at
  most `0.01`, primary containment at most `0.02`;
- stale one-hour, stale one-day, and one-bar delay, raw and independently
  scheduled: Jaccard at most `0.01`, containment at most `0.02`;
- each frozen prior primary clock: Jaccard at most `0.02`, CLASP containment at
  most `0.05`.

Failure rejects CLASP before returns. Component-ablation retention is reported
but deliberately does not decide support, preventing support-gate optimization
from being mistaken for alpha evidence.

## Frozen return gate

- train: `2020-01-01 <= t < 2023-01-01`;
- selection: full 2023 and fixed H1/H2;
- 2024 test, 2025 eval, and 2026 YTD remain sealed;
- leverage `0.5x`, fee `5 bp`, slippage `1 bp` per notional side;
- realized Binance USD-M funding applied at every settlement satisfying
  `entry_time <= funding_time <= exit_time`;
- full-clock CAGR includes all idle time;
- strict held-path MDD uses favorable-first then adverse ordering and excludes
  the scheduled exit bar's later high/low;
- weekly entry-cluster Rademacher test, 100,000 draws, seed `20260714`.

CLASP advances only if train and full 2023 each have positive absolute return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, one-sided cluster `p<0.10`,
and mean gross underlying move strictly above 12 bp. Each 2023 half must be
positive with at least 65 trades. The primary train/selection ratio must beat
the exact direction flip and every frozen score-bearing control. Failure
rejects v1 without threshold, direction, component, or hold repair.
