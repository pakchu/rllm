# OI Round-Trip Geometric-Phase Alpha Search — 2026-07-13

## Thesis

Open interest is a stock of leveraged claims. A large positive OI shock starts
an inventory episode; the experiment freezes the pre-shock OI and price and
waits for the **first causal return of OI to its anchor**. At that point the
incremental leveraged inventory has round-tripped.

If price remains displaced after the leverage stock returns to origin, the
residual price phase may represent cash demand, passive absorption or a broader
repricing that survived the leveraged round trip. The predeclared primary
mapping therefore continues the residual price direction. A second branch
measures the signed line integral `Σ(price displacement × ΔOI)`, a geometric
work term intended to identify forced inventory relaxation.

This is the OI/price dual of a price-loop flow-holonomy experiment. It is not a
rolling OI threshold, fixed-window return gate or positioning-ratio model.

## Causal protocol

- Market/OI source rows are physically cut before `2024-01-01`.
- OI departures use the current completed OI change divided by a scale built
  from changes through `t-1`.
- Price volatility, OI anchor, price anchor and both starting scales are frozen
  before or at departure without future bars.
- One episode is tracked at a time and closes only on its first completed OI
  recross, after at least one hour and before its fixed 12h/24h expiry.
- Fit score quantiles use `2020-10-15..2022-12-31`; 2023 is internal selection;
  2024+ remains sealed.
- Entry is the next 5-minute open, leverage is `0.5x`, and implementation cost
  is `6bp/side`.
- Strict MDD uses favorable-first/adverse-second OHLC high-water accounting.

## Bounded grid

Forty-eight policies:

- positive OI departure: `2σ`, `3σ`;
- maximum episode age: 12h, 24h;
- score: residual price phase, terminal persistence, inventory work;
- fit tail: q80, q90;
- hold: 6h, 12h.

For the top `2σ`/12h state, 2,200 episodes began and 924 completed an OI
round trip. Median closure age was 39 bars; residual directions were balanced
at 457 up and 467 down.

## Best ranked policy

`2σ` positive OI shock, first reclosure within 12h, q90 residual price phase,
12h continuation hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +22.22% | +9.49% | 13.65% | 0.70 | 62 | 26/36 |
| 2023 | +6.19% | +6.20% | 6.03% | 1.03 | 42 | 19/23 |
| 2023H1 | +4.90% | +10.14% | 4.88% | 2.08 | 19 | 8/11 |
| 2023H2 | +2.17% | +4.36% | 6.03% | 0.72 | 22 | 11/11 |

Six of seven pre-2024 half-year segments were positive, but 2022H2 lost
`4.35%`; 2021H1 was nearly flat and contained no long trade. Zero of 48
policies passed the required fit-and-selection CAGR/MDD `>=3` admission gate.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -25.01% / -0.42 / 62 | -10.78% / -0.80 / 42 |
| Ignore OI reclosure; emit at fixed age | +13.95% / 0.36 / 95 | +5.33% / 0.63 / 43 |
| Every OI round trip, no phase score | -64.44% / -0.57 / 469 | +0.35% / 0.04 / 183 |
| Delay selected signal by one hour | +32.56% / 0.93 / 62 | +3.27% / 0.68 / 42 |

The exact direction, first reclosure and residual-tail selection all contribute:
the flip loses, fixed-age is weaker and unfiltered loops have no usable edge.
However, a one-hour lag improves fit and remains mildly positive in 2023, so
the event is a broad state rather than a sharply timed execution edge.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | +26.85% / 0.83 | +8.90% / 1.50 |
| 1bp | +26.07% / 0.81 | +8.45% / 1.42 |
| 3bp | +24.52% / 0.76 | +7.54% / 1.26 |
| 6bp | +22.22% / 0.70 | +6.19% / 1.03 |

The edge survives standard costs; cost is not the rejection reason. The failure
is insufficient risk efficiency and temporal stability.

## Decision

**Do not promote to alpha and do not open OOS.** Preserve continuous OI-loop
age, residual phase, terminal persistence and signed inventory work as weak
beta context. Record the exact 48 static threshold/hold policies as gamma
failure provenance. Any reuse must employ genuinely fresh forward data and a
materially different learner rather than retune these inspected settings.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_oi_roundtrip_geometric_phase_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_oi_roundtrip_geometric_phase_alpha.py
```

Artifacts:

- `training/search_oi_roundtrip_geometric_phase_alpha.py`
- `tests/test_search_oi_roundtrip_geometric_phase_alpha.py`
- `results/oi_roundtrip_geometric_phase_alpha_scan_2026-07-13.json`
