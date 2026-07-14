# Visibility-layer irreversibility alpha — frozen pre-outcome design

## Hypothesis

A directed horizontal visibility graph (HVG) converts the order topology of a
time series into a network. A mismatch between its in-degree and out-degree
distributions is a model-free time-asymmetry descriptor. The fixed trading
hypothesis is that, when price and aggressive flow have materially different
HVG irreversibility, the more irreversible layer carries the locally relevant
direction:

- if flow irreversibility is larger, follow current completed flow;
- if price irreversibility is larger, follow current completed price return.

This is **not** a claim that the layer causes future price or that
irreversibility guarantees predictability. Directed-HVG/KL time irreversibility
was introduced by Lacasa et al.; its financial application reports information
complementary to volatility and associates high irreversibility with stressed
periods, but does not validate this trading rule:

- https://doi.org/10.1140/epjb/e2012-20809-8
- https://arxiv.org/abs/1601.01980
- https://doi.org/10.1016/j.physleta.2016.03.011

## Frozen feature construction

- Source: project BTCUSDT five-minute OHLC, quote volume and taker-buy quote
  cache. The returned analysis frame is strictly before `2024-01-01`.
- A six-hour block is `[T-6h,T)`, contains exactly 72 complete five-minute
  rows, and ends on the minute-55 source bar before UTC boundary `T`.
- Block price is `log(last close / first open)`.
- Block flow is `sum(2*taker_buy_quote - quote_volume) / sum(quote_volume)`.
- At boundary `T`, each layer's HVG contains the 168 completed blocks ending
  with the current `[T-6h,T)` block. No later block enters the graph.
- Directed edge `i -> j` exists iff `i < j` and every intermediate value is
  strictly below `min(x_i,x_j)`.
- For each layer, estimate both in/out degree distributions on one shared
  support `0..max(max_in_degree,max_out_degree)` with Jeffreys pseudocount
  `0.5`.
- Irreversibility is the symmetric mismatch
  `0.5 * (KL(Pin||Pout) + KL(Pout||Pin))`. It is invariant to reversing the
  series and to strictly monotone value transforms; it measures the magnitude,
  not an arrow sign.
- `layer_log_ratio = log((flow_irrev+1e-9)/(price_irrev+1e-9))` and
  `score = abs(layer_log_ratio)`.
- Gate: `score_t > rolling_q80(score[t-120:t], min_periods=60)`. The threshold
  input is shifted one state; equality is inactive.
- Side: `sign(current flow)` for positive layer ratio, otherwise
  `sign(current price return)`.

## Frozen replay and support gate

- Signal boundary: minute 00; executable entry: minute-05 next open.
- Hold: 144 five-minute bars / 12 hours; leverage `0.5x`; cost `6 bp/side`.
- Strict MDD: conservative favorable-first/adverse-second OHLC high-water path.
- Fit: `2020-06-01 <= T < 2023-01-01`.
- One-shot internal selection: calendar 2023 with fixed H1/H2 diagnostics.
- No 2024+ feature, outcome or source hash may be opened.
- Support uses simulator-identical chronological non-overlap: enter at
  `signal+1`, exit at `entry+144`, require split-contained exit, and accept the
  next signal only after the prior exit.
- Pre-outcome floors: fit >=200, 2023 >=60, each 2023 half >=25, fit >=50 per
  side and 2023 >=15 per side.
- Outcome admission: positive fit and 2023 absolute return,
  CAGR/strict-MDD >=3 in both, and positive 2023 H1/H2 absolute return.

## Frozen controls

- exact direction flip;
- same primary event set always following current flow;
- same primary event set always following current price return;
- price-HVG-only prior-q80 events following price;
- flow-HVG-only prior-q80 events following flow;
- six-hour and seven-day signal delays;
- 0/1/3/6/10/15 bp per-side cost stress;
- minute-05/10/15 entry diagnostics;
- fixed 6/12/24-hour hold diagnostics.

Before outcomes, novelty must pass against cross-map dominance/event membership,
rolling price volatility, mean absolute flow, 28-block price trend, and an
order-3 price permutation-entropy reference computed over the same 168 blocks.
Entry and hold diagnostics are report-only and cannot replace the primary after
2023 is opened.
