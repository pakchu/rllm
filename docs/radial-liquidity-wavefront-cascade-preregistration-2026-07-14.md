# RLWC-144 preregistration — 2026-07-14

## Decision and claim boundary

**RLWC-144 (Radial Liquidity Wavefront Cascade, 12-hour hold)** tests whether
unusually directional visible depth change propagates from the 3–5% shells
toward the 0–2% shells, while the opposite side develops a supporting addition
wave. A long requires ask withdrawal plus bid addition; a short is the exact
bid/ask symmetry.

“Wavefront,” “addition,” and “withdrawal” are compact feature names, not claims
that individual orders or cancellations were observed. Binance Vision provides
nominal 30-second cumulative percentage-band snapshots, without order IDs,
update IDs, or sub-snapshot events. The stored flow is only signed visible shell
mass change between adjacent snapshots.

The outer-to-inner mechanism is a falsifiable inference. Primary literature
supports non-flat depth shape, deeper-book relevance, and state-dependent
resiliency, but does not establish this exact cascade as a market law:

- <https://arxiv.org/abs/cond-mat/0203511>
- <https://arxiv.org/abs/0801.3712>
- <https://arxiv.org/abs/1708.02715>
- <https://arxiv.org/abs/1602.00731>
- <https://arxiv.org/pdf/1003.3796>

This preregistration and its support program contain no BTC price, return, PnL,
CAGR, MDD, label, or 2024+ access. Calendar 2023 is development/selection data,
not clean OOS.

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
- sides: suffix `m` is bid/negative percentage; suffix `p` is ask/positive
  percentage
- shells: `1=0–1%`, `2=1–2%`, `3=2–3%`, `4=3–4%`, `5=4–5%`
- no fill, interpolation, nearest-time join, or post-2023 row is allowed

Official source semantics and the snapshot-to-shell transform are frozen in
`docs/binance-cross-collateral-book-shell-data-audit-2026-07-14.md`. Relevant
official references are:

- <https://github.com/binance/binance-public-data>
- <https://github.com/binance/binance-public-data/issues/437>
- <https://github.com/binance/binance-public-data/issues/447>
- <https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

## Frozen shell flow

For visible shell mass `H_(k,i)` and total 0–5% side depth `T_i` at adjacent
raw snapshots `i-1,i` inside one accepted five-minute bar:

```text
denom_i       = 0.5 * (T_i + T_(i-1))
flow_(k,i)    = (H_(k,i) - H_(k,i-1)) / denom_i
flow_net_k,t  = sum_i(flow_(k,i))
efficiency_k,t = abs(flow_net_k,t) / sum_i(abs(flow_(k,i)))
```

Efficiency is zero when the denominator is zero. A positive `flow_net` is
visible shell-mass addition; a negative value is visible withdrawal.

## Causal normalization

Each venue/side/shell `flow_net` is standardized independently. Let `x_t` be
the current clean value, `p_t=x_(t-1)`, window `W=8640`, and minimum prior
history `N=2016`:

```text
center_t   = rolling_median(p_t, W, N)
residual_t = abs(p_t - center_t)
scale_t    = 1.4826 * rolling_median(residual_t, W, N)
Z(x_t)     = clip((x_t - center_t) / scale_t, -12, 12)
```

Every historical residual uses the center that was available at its own time.
Missing or zero scale fails closed. A future row cannot revise an earlier
score.

## Frozen venue-side wave detector

For venue `v`, side `s`, and wave kind `q`, define sign `d_q=+1` for addition
and `d_q=-1` for withdrawal:

```text
z_vk,t = d_q * Z(flow_net_vk,t)
O_t    = 0.5 * (z_v4,t + z_v5,t)   # outer 3–5%
M_t    = z_v3,t                     # middle 2–3%
I_t    = 0.5 * (z_v1,t + z_v2,t)   # inner 0–2%
```

At terminal completed bar `t`, use exactly six bars `t-5…t`:

1. `o` is the earliest maximum of `O` over `t-5…t-3`, and `O_o >= 1.25`;
2. `m` is the earliest maximum of `M` over `o+1…t-1`, and `M_m >= 1.00`;
3. `I_t >= 1.25`, so `o < m < t`;
4. every shell in the selected stage has matching raw sign: shells 4 and 5
   at `o`, shell 3 at `m`, and shells 1 and 2 at `t`;
5. the minimum participating-shell efficiency is at least `0.35` at each
   selected stage;
6. `I_j < 1.00` for every `j` from `t-5` through `m-1`, rejecting an inner
   wave that arrived before the middle stage;
7. `O_t < 1.00`, rejecting a terminal state that is still outer-dominated;
8. every required value is finite and all six bars are `source_complete`.

There is no threshold grid and no alternate ordering.

## Frozen cross-venue action

A venue wave is considered recent only on its terminal bar or the immediately
following bar. This two-bar tolerance permits nominal 30-second UM/CM sampling
to aggregate into adjacent five-minute bars without permitting a wider search.

At completed bar `t`, long requires all of:

```text
source_complete_t
recent UM ask-withdrawal wave
recent CM ask-withdrawal wave
recent bid-addition wave on UM or CM
not (recent bid-withdrawal wave on both venues)
not (recent ask-addition wave on both venues)
```

Short is the exact bid/ask symmetry:

```text
source_complete_t
recent UM bid-withdrawal wave
recent CM bid-withdrawal wave
recent ask-addition wave on UM or CM
not (recent ask-withdrawal wave on both venues)
not (recent bid-addition wave on both venues)
```

A simultaneous long/short conflict is flat.

## Clock and support scheduler

For a signal row indexed `t`:

- information: completed five-minute bars through `t` only;
- entry: next five-minute open at row `t+1`;
- hold: exactly 144 completed five-minute bars, or 12 hours;
- exit: scheduled open at row `t+145`;
- exposure in later evaluation: `+0.5x` long or `-0.5x` short;
- scheduler: reset flat at each calendar-quarter boundary, then greedily
  accept chronological candidates whose entry is at or after the prior exit;
- every accepted trade must exit inside its calendar quarter.

The annual event clock is the concatenation of the four quarter-contained
schedules. A source gap on the decision row or inside any participating
six-bar wave window fails closed. A future shell-data gap cannot cancel an
already scheduled trade; the later evaluator must separately reject any
incomplete execution-price path.

## Outcome-blind support gates

Reject RLWC-144 without opening returns unless every gate passes:

- at least 120 non-overlapping scheduled trades;
- at least 45 trades in each half of 2023;
- at least 20 trades in each calendar quarter;
- long and short share each at least 35%;
- no quarter contributes more than 40% of trades;
- all signal inputs obey the decision-row and six-bar finite/source-complete
  rule.

## Independence gates

The prior CCLH and PDF-10 clocks are replayed without loading prices. RLWC-144
is rejected as insufficiently distinct unless:

- CCLH positions/sides reproduce canonical SHA256
  `e90079d95b111f95ce64459c42d17e4286636a1a2854ed948e8ada497a13dfa7`;
- PDF-10's full six-field clock reproduces canonical SHA256
  `ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2`;
- tolerant event Jaccard against each prior clock within `±12` five-minute
  bars is at most `0.35`.

CLV is not replayed because its event clock requires a price shock. Loading a
price series merely to calculate support overlap would violate this phase's
outcome-blind boundary.

## Frozen 2023 return evaluation, only if support passes

The return evaluator must be written, tested, committed, and separately
hash-frozen before opening RLWC-144 outcomes. It must reproduce the frozen
support clock exactly and must not drop an accepted event silently.

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

Qualification gates, with no parameter or direction change:

- H1 and H2 absolute return both positive;
- H1 and H2 CAGR / strict MDD both at least `3.0`;
- H1 and H2 strict MDD both at most `15%`;
- every calendar quarter absolute return positive;
- one-sided weekly-cluster sign-flip p-value below `0.10` in both halves,
  using 100,000 permutations and seed `20260714`;
- RLWC-144 must beat exact reverse, always-long, always-short, causal five-minute
  price-momentum side, and a frozen sign-permutation control on minimum H1/H2
  CAGR/MDD, all on the same reserved RLWC event clock.

Wave incidence, addition-only, or withdrawal-only variants may be reported as
structural diagnostics only. They cannot replace RLWC-144 after outcomes open.

## OOS stopping rule

If any support, independence, or 2023 return gate fails, RLWC-144 v1 is closed
and 2024+ shell outcomes remain sealed. If every gate passes, the unchanged
builder and frozen action may construct calendar 2024 for the first genuine OOS
evaluation. No threshold relaxation, side reversal, holding-period search,
Gemma/LLM gate, RL update, or portfolio rescue is allowed for this candidate.
