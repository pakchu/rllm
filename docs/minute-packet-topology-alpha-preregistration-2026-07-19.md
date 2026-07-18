# Minute Packet Topology alpha battery — preregistration

## Status

**Outcome-blind.** No post-signal return, execution path, funding PnL, CAGR, or
MDD is used by the support stage. The source is physically truncated before
`2024-01-01` and every rolling reference excludes the current bar.

This battery tests a new source axis: the topology of five completed one-minute
aggregates inside a five-minute Spot/USD-M bar. It is not individual-fill HHI
and does not reuse the failed MFIC event-level fragmentation formula.

## Frozen mechanisms

### 1. USD-M small-ticket swarm absorption

The USD-M quote-notional HHI is unusually low relative to trade-count HHI,
which indicates that trade arrivals concentrate more than capital. The flow
must be persistent, have at most one sign switch across adjacent minutes, fail
to receive Spot flow confirmation, and show weak flow-normalized price impact.
The fixed action fades USD-M net taker flow.

Grid:

- HHI-gap prior quantile: `q10`, `q20` (lower tail);
- flow-normalized impact prior quantile: `q20`, `q35` (lower tail);
- hold: `24`, `48`, `96` five-minute bars.

### 2. Cross-venue churn breakout

USD-M flow changes sign in at least three of four adjacent-minute transitions,
minute-average ticket dispersion is high, net flow is small, but absolute price
impact is high. Spot and USD-M current-bar return signs must agree. This is a
liquidity-displacement hypothesis: price moved despite cancelling directional
flow, so the fixed action follows the completed-bar move.

Grid:

- ticket log-dispersion prior quantile: `q70`, `q80` (upper tail);
- absolute net-flow prior quantile: `q20`, `q35` (lower tail);
- absolute signed impact fixed at prior `q80` or above;
- hold: `24`, `48`, `96` five-minute bars.

The two direction rules are frozen. A failed direction is not inverted after
returns are observed.

## Shared causal contract

- rolling window: 8,640 bars / 30 days;
- minimum prior history: 2,016 bars / 7 days;
- signal only on inactive-to-active onset;
- next-open entry and fixed-open exit;
- global non-overlap within each candidate;
- 2020–2023 source only at support and selection time;
- 2024 test, 2025 eval, and 2026 holdout stay sealed until one policy is frozen.

Outcome-blind support requires at least 150 events, 30 in every year, 15 in
each 2023 half, at least 25% on each side, and no more than 15% of events in one
month. Unsupported cells are removed without opening returns.

The frozen support run admitted **21 of 24** cells. The three rejected cells are
the `cross_venue_churn_breakout` `p80/s20` variants; each had only 14 events in
2023 H2 versus the fixed floor of 15. No direction or threshold was changed.

- source rows / valid rows: `420,768 / 420,205`
- source SHA-256: `5ea9f5075171c255732cc6eed003736c1beed211a0e6fd7797ab02f31a917aaa`
- support artifact: `results/minute_packet_topology_support_2026-07-19.json`
- support artifact SHA-256:
  `3ba017cbd1145b09b0bc3cc58b74a732fd57445304b7d884b52f9d50bed03f7c`

## Frozen return protocol

Supported cells are evaluated with:

- train: full 2020–2022;
- selection: full 2023, plus fixed H1/H2 diagnostics;
- leverage: `0.5x`;
- base fee plus slippage: `6 bp` per notional side;
- stress cost: `10 bp` per notional side;
- exact realized funding over the held interval;
- full-calendar CAGR including idle time;
- strict MDD from global/pre-entry HWM and favorable-before-adverse held 5m
  high/low path.

Train admission requires positive absolute return, ratio at least `1.5`, MDD at
most `20%`, at least 100 trades, and weekly-cluster one-sided `p < 0.10`.
Selection requires positive absolute return, ratio at least `3`, MDD at most
`15%`, positive H1/H2, at least 20 trades in each half, 10bp stress profitability,
and weekly-cluster `p < 0.10`. The winner maximizes the minimum train/selection/
half-year ratio, then lower selection MDD, then lexical name.

Only the frozen winner may open 2024. A promotable alpha must independently
achieve positive return, ratio at least `3`, MDD at most `15%`, at least 40
trades, and weekly-cluster `p < 0.10` in both 2024 and 2025. Combined 2024–2025
must have `p < 0.05`; 2026 is a report-only holdout and cannot repair failure.
