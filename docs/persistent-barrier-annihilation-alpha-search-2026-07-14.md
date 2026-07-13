# Persistent-Barrier Annihilation Alpha Search — 2026-07-14

## Thesis

A rolling maximum or minimum retains only one barrier. A one-dimensional
sublevel/superlevel persistence view retains **all local extrema and their
topographic prominence**. Prominence measures how much price relief surrounds
an extremum before it merges into an older component and is the practical 1D
component-persistence quantity used here.

At every hourly decision, the experiment freezes the previous window's full
set of persistent resistance peaks and support troughs. During the next hour,
price may cross and thereby annihilate several of those barriers. The proposed
state records:

- total normalized persistence mass crossed;
- largest persistence crossed;
- number of barriers crossed;
- persistence mass per unit price traversal.

Continuation interprets mass destruction as a structural breakout. Fade
interprets unusually dense mass destruction as transient over-extension and
liquidity consumption. Both interpretations were explicitly recorded before
opening any 2024+ data.

This is distinct from rolling global extrema, extrema ancestry and nested-range
touches: the state uses the full prior local-extrema persistence spectrum.

## Causal protocol

- Market rows are physically cut before `2024-01-01`.
- For decision `t`, the freeze point is `t-12` bars.
- The 2-day/7-day extrema window ends at `freeze-1`; the traversal starts at
  the completed `freeze` close and ends at the completed `t` close.
- The prominence floor and normalization scale use one-hour volatility ending
  at `freeze-1` and then remain fixed.
- Signals exist only on hourly completed bars and enter the next 5-minute open.
- Fit is 2020-10-15 through 2022; 2023 is inspected internal selection; 2024+
  remains sealed.
- Leverage is `0.5x`, implementation cost is `6bp/side`, and strict MDD uses
  favorable-first/adverse-second OHLC high-water accounting.

## Transparent bounded audit

The initial 24-policy continuation scan was weak. Before opening OOS, the audit
was expanded to record all already inspected researcher degrees of freedom:

- topology horizon: 2 days, 7 days;
- score: mass, largest barrier, mass density;
- fit tail: q80, q90, q95, q97.5;
- hold: 6h, 12h, 24h;
- fixed mapping: continuation or fade.

This is 144 policies. The expansion is contamination inside pre-2024 and is
reported rather than hidden; none of these settings may be tuned again on this
sample.

The two-day state emitted 22,424 hourly crossings and the seven-day state
26,640. Their median crossed-barrier counts were two and four, respectively.

## Best ranked policy

Seven-day topology, q97.5 persistence-mass density, fade, 24-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +14.95% | +6.50% | 23.04% | 0.28 | 205 | 101/104 |
| 2023 | +9.61% | +9.62% | 9.64% | 1.00 | 115 | 62/53 |
| 2023H1 | +7.37% | +15.42% | 8.24% | 1.87 | 45 | 21/24 |
| 2023H2 | +2.19% | +4.39% | 9.64% | 0.46 | 70 | 41/29 |

Six of seven half-year segments were positive. The exception was the short
2020Q4 segment at `-8.67%`; 2023H2 was positive but weak. Zero of 144 policies
passed the required fit-and-2023 CAGR/MDD `>=3` gate.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -36.90% / -0.43 / 205 | -21.35% / -0.86 / 115 |
| Every crossing; no tail | +6.35% / 0.06 / 765 | -17.59% / -0.57 / 345 |
| Barrier count only | +25.53% / 0.66 / 209 | +8.87% / 0.94 / 95 |
| Single frozen global max/min | -25.33% / -0.33 / 207 | -26.97% / -0.98 / 77 |
| Delay selected signal by one hour | +19.60% / 0.36 / 205 | +5.76% / 0.57 / 115 |

The full persistence spectrum is not reducible to a single rolling extremum,
and the exact direction and tail selection matter. Barrier count explains some
of the effect, while density adds modest 2023 efficiency. A one-hour lag remains
positive, so the state is not a sharp execution timestamp.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | +29.99% / 0.59 | +17.44% / 1.99 |
| 3bp | +22.24% / 0.43 | +13.46% / 1.46 |
| 6bp | +14.95% / 0.28 | +9.61% / 1.00 |
| 10bp | +5.89% / 0.11 | +4.68% / 0.46 |

The feature has gross directional information but is economically diluted by
turnover and regime variation. Cost is important but not the sole rejection
reason.

## Decision

**Do not promote the static strategy and do not open 2024+.** Preserve the
continuous persistence spectrum, crossed mass, density and barrier count as a
weak beta representation for a materially different sequential learner. The
144 inspected static combinations are gamma failure provenance.

## Research context

The transform is inspired by persistent-homology work on financial critical
transitions, but the frozen-barrier annihilation execution mapping is this
repository's own falsifiable construction:

- [Gidea & Katz, Landscapes of crashes](https://doi.org/10.1016/j.physa.2017.09.028)
- [Topological recognition of cryptocurrency transitions](https://arxiv.org/abs/1809.00695)

These papers motivate topology as a state representation; they do not validate
this trading rule.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_persistent_barrier_annihilation_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_persistent_barrier_annihilation_alpha.py
```

Artifacts:

- `training/search_persistent_barrier_annihilation_alpha.py`
- `tests/test_search_persistent_barrier_annihilation_alpha.py`
- `results/persistent_barrier_annihilation_alpha_scan_2026-07-14.json`
