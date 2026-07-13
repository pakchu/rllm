# Closed-Excursion Inventory Holonomy Search — 2026-07-13

## Thesis

A price path can close while aggressive inventory does not. The experiment
starts an excursion when a completed 5-minute return exceeds `2σ` or `3σ`,
freezes the previous close as an anchor, and waits for the first causal recross.
If price returns to the anchor but cumulative taker flow still has the departure
sign, those aggressive traders paid for displacement that did not persist.

That nonzero inventory around a closed price loop is called **inventory
holonomy** here. The fixed direction is to fade the trapped aggressive side at
the next open.

Three representations were tested:

- cumulative same-side flow divided by square-root episode age;
- dissipative work, `Σ(flow × anchor-normalized displacement) / sqrt(age)`;
- flow elasticity, absolute cumulative flow divided by square-root episode age
  and maximum excursion.

This is an event-defined first-return loop, not a fixed-window path signature,
price tail or generic flow gate.

## Causal protocol

- The market source is physically cut before `2024-01-01`.
- Departure volatility and flow activity scales use history through `t-1`.
- The anchor and start volatility are frozen at departure.
- Only completed bars update age, work, maximum displacement and flow.
- A loop emits once, on its first anchor recross, after at least one hour and
  before its fixed 12-hour/24-hour expiry.
- Fit-only positive-score quantiles use `2020-06-01..2022-12-31`; 2023 is
  inspected internal selection; 2024+ stays sealed.
- Entry is next 5-minute open, leverage `0.5x`, cost `6bp/side`.
- Strict MDD uses favorable-first/adverse-second OHLC high-water.

## Bounded grid

Forty-eight policies:

- departure: `2σ`, `3σ`;
- maximum loop age: 12h, 24h;
- score: cumulative flow, dissipative work, flow elasticity;
- fit tail: q80, q90;
- hold: 6h, 12h.

For the top departure/expiry pair, 2,100 price loops completed; 1,150 retained
same-side net taker inventory. Median completed age was 33 bars.

## Best ranked policy

`2σ` departure, 12-hour expiry, q90 cumulative-flow holonomy, 6-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +9.21% | +3.47% | 8.49% | 0.41 | 78 | 59/19 |
| 2023 | -1.07% | -1.08% | 3.75% | -0.29 | 24 | 15/9 |
| 2023H1 | +0.70% | +1.41% | 2.55% | 0.55 | 11 | 8/3 |
| 2023H2 | -1.76% | -3.46% | 3.45% | -1.00 | 13 | 7/6 |

Five of seven pre-2024 half-year segments were positive, but the earliest fit
segment and final selection half lost. Zero of 48 policies passed admission.

## Structural controls

- exact direction flip: fit `-17.03%`, ratio `-0.35`; 2023 `-1.87%`, ratio
  `-0.42`;
- price first-return loops without inventory selection: fit `-36.94%`, ratio
  `-0.37`; 2023 `-20.19%`, ratio `-0.90`;
- one-hour signal lag: fit `+7.73%`, ratio `0.50`; 2023 `-4.07%`, ratio `-0.80`.

The inventory object and direction add information relative to price loops, but
timing and economics remain weak.

At zero cost the top policy reaches fit `+14.44%`, CAGR `5.36%`, strict MDD
`6.94%`, ratio `0.77`; 2023 is only `+0.36%`, CAGR `0.36%`, strict MDD `3.16%`,
ratio `0.11`, and 2023H2 still loses `0.99%`. Costs do not create the temporal
failure.

## Decision

**Not alpha.** Standard-cost 2023 and its second half fail, support is modest,
and the long/short distribution is imbalanced. OOS remains sealed.

The continuous closed-loop representation is retained as **weak beta** because
the exact flip and price-only control are materially worse and five segments
were positive. The inspected fixed thresholds/expiries/holds are gamma failure
provenance. Do not tune nearby departure widths, tails or holds on this sample.

Artifacts:

- `training/search_closed_excursion_holonomy_alpha.py`
- `tests/test_search_closed_excursion_holonomy_alpha.py`
- `results/closed_excursion_holonomy_alpha_scan_2026-07-13.json`
