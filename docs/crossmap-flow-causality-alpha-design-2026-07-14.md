# Nonlinear price/flow cross-map alpha — frozen pre-outcome design

## Hypothesis

A nonlinear state-space reconstruction may distinguish price/flow coupling
states even when ordinary correlation and one-lag linear lead/lag do not.

- `flow_to_price_skill`: the price/effect manifold reconstructs the putative
  flow/cause, following the CCM convention.
- `price_to_flow_skill`: the flow/effect manifold reconstructs the putative
  price/cause.
- `crossmap_dominance = flow_to_price_skill - price_to_flow_skill`.
- Positive dominance follows current completed aggressive flow as a fixed
  policy rule.
- Negative dominance fades current completed aggressive flow as a fixed policy
  rule.

These signs are labels for a preregistered trading policy, not validated
"information-leading" or "reactive" regimes. This is a **cross-map asymmetry
descriptor**, not proof of causality. CCM was
introduced for nonlinear dynamical systems by Sugihara et al.; published
limitations show that noise, synchrony and some coupling regimes can make
direction unreliable. Sources:

- https://doi.org/10.1126/science.1227079
- https://pubmed.ncbi.nlm.nih.gov/25615160/
- https://pure.au.dk/portal/en/publications/inferring-causality-from-noisy-time-series-data-a-test-of-converg/

## Frozen feature construction

- Source: project BTCUSDT five-minute OHLC/quote-volume/taker-buy-quote cache.
- Returned source frame is strictly before `2024-01-01`.
- Six-hour blocks end at UTC 00/06/12/18 boundaries and require all 72 prior
  five-minute bars. The newest feature source is the minute-55 completed bar.
- Block price return: log(last close / first open).
- Block flow: sum(`2*taker_buy_quote - quote_volume`) / sum quote volume.
- At decision block `t`, cross-map state uses blocks `[t-120, t)` and therefore
  excludes current block `t`; current block contributes only the completed flow
  sign used by the policy.
- Price and flow are standardized inside that preceding library.
- Embedding dimension `E=3`, lag one block, `E+1=4` neighbors.
- Leave-one-out reconstruction excludes temporal neighbors within Theiler
  radius one. Simplex weights are `exp(-distance / nearest_distance)`.
- Reconstruction skill is Pearson correlation between reconstructed and actual
  library values.
- Gate: `abs(dominance_t) > rolling_q80(abs(dominance[t-120:t]),
  min_periods=60)`. The rolling input is shifted by one state, so the current
  dominance never estimates its own threshold. Equality is inactive.
- No return label, trade PnL or 2024+ value estimates the feature, side or gate.

## Frozen replay

- Signal: UTC boundary minute 00.
- Entry: minute-05 next open.
- Primary hold: 144 five-minute bars / 12 hours.
- Leverage: 0.5x.
- Cost: 6 bp per side.
- Strict MDD: favorable-first/adverse-second OHLC high-water convention.
- Fit: `2020-06-01 <= t < 2023-01-01`.
- One-shot internal selection: calendar 2023 with fixed H1/H2 diagnostics.
- No 2024+ outcome may be opened.
- Before any return is opened, support must pass using the simulator-identical
  execution rule: scan signal positions chronologically; enter at `position+1`,
  exit at `entry+144`; require the exit to remain inside the same split; reject
  overlaps through that exit; accept the next signal only after `exit+1`.
- Pre-outcome floors: at least 200 executable fit trades, 60 executable 2023
  trades, 25 executable trades in each 2023 half, at least 50 fit longs and 50
  fit shorts, and at least 15 2023 longs and 15 2023 shorts.
- Return admission: positive fit and 2023 absolute return,
  CAGR/strict-MDD >=3 on both, and positive absolute return in 2023 H1/H2.

## Frozen controls

- exact direction flip;
- the primary event set following/fading `sign(current completed flow)`;
- the same primary event set following/fading `sign(current completed block
  price return)`;
- ordinary one-lag linear lead/lag asymmetry, sided as
  `sign(linear_asymmetry) * sign(current flow)`, with its own strict prior-only
  rolling q80 of absolute linear asymmetry;
- six-hour and seven-day signal delays;
- 0–15 bp/side cost stress;
- fixed minute-05/10/15 entry diagnostics;
- fixed 6/12/24-hour hold diagnostics;
- pre-outcome Spearman and event-Jaccard novelty audit against linear
  lead/lag and same-time correlation.

Entry and hold diagnostics are report-only and cannot replace the primary after
2023 is opened.
