# PIVOT-72 source-support decision

## Decision

Retire `PIVOT-72` unchanged before opening market prices, funding, comparator
rows, labels, rewards, or any post-entry outcome.

The preregistered source-only gate failed. The target fraction, anchor cutoff,
prior length, token schema, support floors, latency, and hold period therefore
remain frozen and are not repaired.

## Frozen result

| Item | Value |
|---|---:|
| Source rows decoded | 420,768 |
| Base paired states | 1,104 |
| Token-ready states | 1,014 |
| Globally reserved states | 1,013 |
| Split-contained opportunities | 1,012 |
| Train opportunities | 546 |
| Selection 2022 opportunities | 316 |
| Eval 2023 opportunities | 150 |
| Market rows decoded | 0 |
| Funding rows decoded | 0 |
| Comparator rows decoded | 0 |
| Future-return rows decoded | 0 |

The source-support report manifest is
`b7690b1d2a6bc864c9a16b162917407b98011f6c00c5428a4b658732bb63d186`.

## Failed gates

All train and 2022 selection gates passed. The failures were confined to 2023:

- only `150` opportunities, below the frozen `200` minimum;
- half-year counts `109 / 41`, so H2 missed the frozen `85` minimum;
- quarter counts `49 / 60 / 3 / 38`, so Q3 missed the frozen `35` minimum;
- `9` active months instead of `12`;
- maximum month share `16.67%`, above `15%`;
- maximum entry gap above `125` days, above `14` days; and
- 2023 `spot_late_abs_flow_q=Q3` share `50%`, above the frozen `40%` ceiling.

The first failing check in canonical order was `eval_opportunities`.

## Interpretation

PIVOT had ample global, train, and 2022 incidence, but the predictor source
produced a severe 2023 coverage discontinuity and a shifted spot late-flow
distribution. Passing the gate would require changing a preregistered
mechanism or accepting an eval regime with insufficient temporal coverage.
Both are prohibited.

No profitability claim was tested. This is a source-support retirement, not a
negative backtest.

## Bound artifacts

| Artifact | SHA-256 |
|---|---|
| `results/paired_intrinsic_venue_orderflow_topology_support_2026-07-24.json` | `d20a2647017bce5b6c8a8c8993d3b5aca9307aae457e433a59128f1c6dd2db5b` |
| `data/paired_intrinsic_venue_orderflow_topology_states_2020_2023.csv.gz` | `2828d9f0092ada7578e7297420cc95c6f6fa76c050458dfd9b5b3ed55ec5ae3e` |
