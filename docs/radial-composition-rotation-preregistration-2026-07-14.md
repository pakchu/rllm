# RCR-144 preregistration — 2026-07-14

## Decision and claim boundary

**RCR-144 (Radial Composition Rotation, 12-hour hold)** tests whether the
normalized visible-depth composition rotates closer to the reference on the bid
side relative to the ask side, consistently across USD-M and COIN-M.

“Inward” and “rotation” describe a change in the five-shell composition. They
do not prove that an order moved, that a cancellation occurred, or that mass was
conserved. Binance Vision supplies nominal 30-second snapshots of cumulative
depth in percentage bands whose reference moves with the book. Hidden
liquidity, update IDs, order IDs, and sub-snapshot activity are not observed.

This preregistration and its support program contain no BTC price, future
return, PnL, CAGR, MDD, label, or 2024+ access. Calendar 2023 is
development/selection data, not clean OOS.

## Why this is a new predictive object

RCR-144 is not a repair of failed RLWC-144:

- RLWC required a rare ordered outer-to-middle-to-inner flow event and
  near-synchronous UM/CM terminal waves. RCR has no wave, six-bar order,
  synchronization window, addition/withdrawal classification, or RLWC
  threshold.
- CCLH uses persistent cross-contract cumulative-depth level and elasticity.
  RCR uses one-bar change in each side's internally normalized shell
  composition.
- PDF-10 uses displayed direction versus net firmness/churn. RCR uses neither
  display/firmness contradiction nor churn.

The exact bid/ask antisymmetry supplies direction: a bid composition rotating
inward relative to asks is positive; swapping bid and ask must negate the score.

## Outcome-blind mechanism screen

No returns or prices were opened during design. Two continuous candidates were
checked against the already frozen support and independence boundaries:

| candidate | finite rows | strong bars | 144-bar clock | max prior-feature Spearman | decision |
|---|---:|---:|---:|---:|---|
| direction-symmetric radial flow first moment (DS-RTP) | 93,587 | 18,786 | 618 | 0.610049 | rejected above 0.60 |
| radial composition rotation (RCR-144) | 97,414 | 18,884 | 646 | 0.499388 | preregister |

A scratch continuous flow-torque state produced only 168 scheduled events at a
weak `0.1` level threshold after an incidence grid. It was discarded before
registration rather than turning support calibration into a return search.

This screen establishes observability and distinction only. It is not evidence
of profitability. RCR's equations and threshold are now fixed; no support
repair is allowed.

## Frozen data boundary

- input manifest:
  `results/binance_cross_collateral_book_shells_btc_2023_manifest.json`
- manifest SHA256:
  `1b5519143d58f62ef3e8b6d9e22f012f80197a59903509041aca24252ed04521`
- input panel:
  `data/binance_cross_collateral_book_shells_btc_2023/BTC_cross_collateral_book_shells_5m_2023.csv.gz`
- panel SHA256:
  `ead931ec8ce2bbd73c946b8660e16d7750ce73051e60ce4989467a7c5bc68342`
- inclusive start: `2023-01-01 00:00:00` UTC
- exclusive end: `2024-01-01 00:00:00` UTC
- markets: USD-M `BTCUSDT` and COIN-M `BTCUSD_PERP`
- suffix `m`: bid/negative percentage side
- suffix `p`: ask/positive percentage side
- shells: `1=0–1%`, `2=1–2%`, `3=2–3%`, `4=3–4%`, `5=4–5%`
- feature input: only `shell_share_median` for both venues, sides, and shells
- no fill, interpolation, nearest-time join, or post-2023 row is allowed

Official source semantics and shell construction are frozen in
`docs/binance-cross-collateral-book-shell-data-audit-2026-07-14.md`:

- <https://github.com/binance/binance-public-data>
- <https://github.com/binance/binance-public-data/issues/437>
- <https://github.com/binance/binance-public-data/issues/447>
- <https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

## Frozen radial composition

Let `S_(v,s,k,t)` be the median share of shell `k` within the 0–5% visible
side-total for venue `v`, side `s`, and completed five-minute bar `t`.

Each shell is median-aggregated independently, so the five stored medians need
not sum exactly to one. RCR renormalizes them within venue/side/bar:

```text
p_(v,s,k,t) = S_(v,s,k,t) / sum_j S_(v,s,j,t)
B_(v,s,t)   = sum_(k=1..5) k * p_(v,s,k,t)
```

`B` lies from 1 to 5 when available. A lower barycenter means more of the
visible composition is concentrated in nearer shells. Non-finite, negative, or
zero-total shares fail closed.

One-bar inward composition change is:

```text
I_(v,s,t) = B_(v,s,t-1) - B_(v,s,t)
```

Positive `I` means the normalized composition became nearer. Both `t-1` and
`t` must be `source_complete`.

Venue-local bid/ask polarization is:

```text
P_(v,t) = (I_(v,bid,t) - I_(v,ask,t)) / sqrt(2)
```

Swapping bid and ask negates `P`; multiplying all five shares on one side by a
positive constant leaves it unchanged.

## Causal normalization and cross-venue score

Each `P_v` is standardized independently using only prior clean observations.
For `x_t=P_(v,t)`, `p_t=x_(t-1)`, window `W=8640`, and minimum prior history
`N=2016`:

```text
center_t   = rolling_median(p_t, W, N)
residual_t = abs(p_t - center_t)
scale_t    = 1.4826 * rolling_median(residual_t, W, N)
Q_(v,t)    = clip((x_t - center_t) / scale_t, -12, 12)
```

Every historical residual uses the center available at its own time. Missing
or zero scale fails closed. Future rows cannot revise an earlier score.

The venue-symmetric final score is:

```text
RCR_t = (Q_(um,t) + Q_(cm,t)) / sqrt(2)
```

Swapping UM and CM leaves `RCR` unchanged. Swapping bid and ask over the full
history negates it.

## Frozen action and clock

At every completed bar `t`:

```text
RCR_t >= +2.0  -> long candidate
RCR_t <= -2.0  -> short candidate
otherwise      -> flat/no candidate
```

Every strong bar is a candidate; this is a maintained-state policy, not only a
threshold-crossing event. The non-overlap scheduler may re-enter after a prior
scheduled exit if the score is still strong.

For an accepted signal row `t`:

- entry: next five-minute open at row `t+1`;
- hold: exactly 144 completed five-minute bars, or 12 hours;
- exit: scheduled open at row `t+145`;
- later evaluation exposure: `+0.5x` long or `-0.5x` short;
- reset flat at each calendar-quarter boundary;
- greedily accept chronological candidates whose entry is at or after the
  prior scheduled exit;
- every trade must exit inside its quarter.

The annual event clock is the concatenation of four quarter-contained
schedules. A future shell-data gap cannot cancel an entered trade; the later
evaluator must separately reject any incomplete execution-price path.

## Outcome-blind availability and support gates

Reject RCR-144 without opening returns unless every gate passes:

- at least 90,000 finite score rows;
- at least 15,000 finite score rows in every quarter;
- at least 500 strong-score bars in every quarter;
- strong-bar long and short share each at least 35%;
- at least 120 non-overlapping scheduled trades;
- at least 45 trades in each half of 2023;
- at least 20 trades in every quarter;
- scheduled long and short share each at least 35%;
- no quarter contributes more than 40% of scheduled trades.

## Independence gates

CCLH and PDF-10 are replayed from their frozen no-outcome panels. RCR-144 is
rejected unless:

- CCLH positions/sides reproduce canonical SHA256
  `e90079d95b111f95ce64459c42d17e4286636a1a2854ed948e8ada497a13dfa7`;
- PDF-10's six-field clock reproduces canonical SHA256
  `ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2`;
- event Jaccard within `±12` bars against each clock is at most `0.35`;
- no more than 35% of current RCR events match either prior clock within
  `±12` bars;
- maximum absolute Spearman correlation of RCR with CCLH cross-pressure,
  CCLH cross-elasticity, PDF credibility, and PDF display is at most `0.60`.

CLV and RLWC are not feature-correlation gates: CLV needs a price-shock clock,
while RLWC produced zero candidates. No price is loaded merely for support.

## Frozen 2023 return evaluation, only if support passes

The evaluator must be written, tested, committed, and separately hash-frozen
before opening RCR outcomes. It must reproduce the support clock exactly and
must not silently drop an accepted event.

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

Qualification gates, with no parameter or side change:

- H1 and H2 absolute return both positive;
- H1 and H2 CAGR / strict MDD both at least `3.0`;
- H1 and H2 strict MDD both at most `15%`;
- every quarter absolute return positive;
- one-sided weekly-cluster sign-flip p-value below `0.10` in both halves,
  using 100,000 permutations and seed `20260714`;
- RCR-144 must beat exact reverse, always-long, always-short, causal five-minute
  price-momentum side, and frozen sign permutation on minimum H1/H2 CAGR/MDD,
  all on the same reserved RCR clock.

Single-venue, static-barycenter, absolute-rotation, DS-RTP, flow-torque, and
threshold variants may be structural diagnostics only. They cannot replace
RCR-144 after outcomes open.

## OOS stopping rule

If any support, independence, or 2023 return gate fails, RCR-144 v1 is closed
and 2024+ shell outcomes remain sealed. If every gate passes, the unchanged
builder and frozen action may construct calendar 2024 for first genuine OOS
evaluation. No threshold relaxation, side flip, holding-period search, LLM
gate, RL rescue, or portfolio optimization is allowed for this candidate.
