# Dual Intrinsic-Clock Impact-Elasticity Alpha Search — 2026-07-14

## Thesis

Calendar time hides whether price or aggressive order flow is doing the work.
This experiment therefore builds two independent online event clocks:

- a directional-change clock on log price;
- a directional-change clock on cumulative signed taker quote flow.

The flow increment is `(2 * taker_buy_quote - quote_volume)` divided by the
previous completed hour's quote volume. Each directional-change threshold is
frozen when its state starts, so later volatility cannot rewrite event history.

The proposed impact-elasticity interpretation is:

- **flow clock fast, price path flat:** aggressive flow is being absorbed, so
  fade the flow direction;
- **price clock fast, flow non-opposing:** price is crossing liquidity with
  unusually little flow confirmation, so continue the price direction.

This is an event-time state, not a conventional return/volume threshold rule.

## Causal protocol

- Market rows are physically cut before `2024-01-01`.
- Open-time-labelled 5-minute row `:55` is the last completed bar at the next
  hour boundary; its signal enters the following `:00` row's open.
- Price and flow scales use only prior completed observations.
- The flow denominator is the previous completed one-hour quote volume.
- Directional-change thresholds remain fixed until an online state transition.
- A state transition emits at most one event per bar.
- Only the first entry into an impact state may trade; persistent hourly
  re-entry is a separately reported control.
- Fit is 2020-10-15 through 2022. 2023 is inspected internal selection. 2024+
  remains sealed.
- Leverage is `0.5x`, cost is `6bp/side`, and strict MDD uses
  favorable-first/adverse-second OHLC high-water accounting.

## Bounded final grid and disclosed design history

The architect-reviewed grid contains 18 policies:

- directional-change width: `0.75`, `1.0`, `1.5` prior-scale units;
- event-count window: `6h`, `12h`, `24h`;
- clock dominance ratio: `1.5`, `2.0`;
- holding period tied to the window and capped at `12h`.

Before this final grid, more flexible pre-2024 quantile-tail, onset and
recoupling probes were inspected and were weak. Those semantics and every
setting above are now contaminated and frozen. The full design history is
reported rather than disguising the final 18 policies as the only search.

## Best ranked policy

Directional-change width `1.5`, 12-hour clock window, dominance ratio `2.0`,
12-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | -4.83% | -2.21% | 41.36% | -0.05 | 535 | 266/269 |
| 2023 | +18.84% | +18.86% | 8.57% | 2.20 | 225 | 117/108 |
| 2023H1 | +8.22% | +17.27% | 8.57% | 2.02 | 113 | 59/54 |
| 2023H2 | +9.82% | +20.44% | 8.37% | 2.44 | 112 | 58/54 |

The policy emitted 1,459 raw first-entry states: 445 flow-fast absorption
states and 1,014 price-fast vacuum states. Zero of 18 policies passed the
required fit-and-2023 admission gate.

The regime breakdown is decisive: 2020H2 and 2022H1 were profitable, whereas
both halves of 2021 lost and 2022H2 was nearly flat. The attractive 2023 result
is therefore regime-local rather than a stable pre-2024 alpha.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -50.83% / -0.50 / 535 | -36.94% / -0.96 / 225 |
| Persistent hourly re-entry | -1.73% / -0.02 / 581 | +31.89% / 3.57 / 255 |
| Magnitude only; no event clocks | -67.90% / -0.53 / 961 | +2.54% / 0.15 / 412 |
| Delay selected signal by one hour | -21.41% / -0.22 / 535 | +12.50% / 1.49 / 225 |

The exact mapping and event clocks contain directional information: the flip
and magnitude-only controls are materially worse. Persistent re-entry improves
2023 but not fit, so it is another expression of the same regime instability,
not a rescue. A one-hour lag remains positive only in 2023 and degrades fit.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | +31.20% / 0.42 | +36.02% / 4.31 |
| 1bp | +24.37% / 0.31 | +33.00% / 3.94 |
| 3bp | +11.74% / 0.14 | +27.14% / 3.24 |
| 6bp | -4.83% / -0.05 | +18.84% / 2.20 |
| 10bp | -23.17% / -0.24 | +8.61% / 0.88 |

The gross effect is real enough to preserve as a representation, but turnover
consumes it under the repository's executable 6bp/side assumption. Cost is not
the only problem: fit risk efficiency is poor even at zero cost.

## Decision

**Reject static trading and do not open 2024+.** Preserve the two event counts,
their ratio, normalized displacements and impact-state category only as a weak
beta representation for a materially different sequential learner. Record all
static widths, windows, dominance ratios, entry semantics and holds as gamma
failure provenance.

## Research context

Directional-change and intrinsic-time research motivates replacing uniform
calendar sampling with price-defined events. The paired price/flow clock and
impact-elasticity execution map are this repository's own falsifiable
construction:

- [Glattfelder et al., Patterns in high-frequency FX data](https://doi.org/10.3233/AF-160054)
- [Directional changes and intrinsic networks](https://arxiv.org/abs/2204.02682)

These sources motivate event time; they do not validate this trading rule.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_dual_intrinsic_clock_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_dual_intrinsic_clock_alpha.py
```

Artifacts:

- `training/search_dual_intrinsic_clock_alpha.py`
- `tests/test_search_dual_intrinsic_clock_alpha.py`
- `results/dual_intrinsic_clock_alpha_scan_2026-07-14.json`
