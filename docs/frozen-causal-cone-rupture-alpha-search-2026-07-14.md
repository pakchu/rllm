# Frozen Causal-Cone Rupture Alpha Search — 2026-07-14

## Thesis

Every prior hourly price anchor projects a square-root-time diffusion envelope.
The envelope's volatility is frozen from information available strictly before
that anchor. A current completed price that exceeds many independently frozen
upper or lower envelopes simultaneously is interpreted as an information front
that invalidates many older local diffusion expectations at once.

For each current decision and prior hourly anchor `j`:

```text
z(j,t) = (log_price[t] - log_price[j]) /
         (prior_volatility[j] * sqrt(t-j))
```

Upper and lower rupture masses are the mean excess beyond `+2` and `-2` cones.
The larger mass determines direction; its magnitude determines activation.
This is a many-anchor causal geometry, not one rolling return z-score, rolling
maximum/minimum or one current-volatility band.

## Causal protocol

- Source rows are physically cut before `2024-01-01`.
- Anchor volatility is a seven-day rolling return standard deviation ending at
  the bar before each anchor.
- Anchors are sampled hourly and always precede the decision.
- The current completed minute-55 close enters only at the following minute-00
  open.
- Rupture thresholds use fit 2020-10-15 through 2022 only. 2023 is inspected
  internal selection; 2024+ remains sealed.
- `0.5x`, `6bp/side`, fixed hold and favorable-first/adverse-second OHLC strict
  MDD are used.

## Bounded grid

Exactly eight static policies:

- anchor history: two days (`48` hourly anchors) or seven days (`168`);
- fit-only rupture-mass tail: q80, q90;
- hold: six hours, 12 hours;
- fixed continuation in the dominant rupture direction.

The two-day state had 34,965 valid decisions and 8,409 nonzero rupture states.
The seven-day state had 34,905 valid decisions and 12,418 nonzero states.

## Best ranked policy

Seven-day anchors, q90 rupture mass and 12-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +2.54% | +1.14% | 25.24% | 0.05 | 226 | 99/127 |
| 2023 | +22.42% | +22.44% | 8.46% | 2.65 | 127 | 93/34 |
| 2023H1 | +20.47% | +45.62% | 8.46% | 5.39 | 72 | 58/14 |
| 2023H2 | +1.62% | +3.23% | 6.90% | 0.47 | 55 | 35/20 |

Six of seven half-year segments were positive. The exception was 2021H1 at
`-22.30%`; 2023H2 was positive but weak. Fit strict MDD and risk efficiency are
far outside the target, so zero of eight policies passed admission.

The q80/12-hour sibling had stronger aggregate fit (`+25.96%`, ratio `0.48`)
and 2023 (`+22.12%`, ratio `2.03`) but lost in 2022H2 and 2023H2. Tightening the
tail changes where the regime loss occurs rather than producing stability.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -29.92% / -0.42 / 226 | -31.03% / -0.96 / 127 |
| First rupture onset only | +13.99% / 0.39 / 123 | +5.71% / 0.74 / 51 |
| Breached-anchor fraction only | +1.85% / 0.03 / 209 | +28.71% / 3.65 / 108 |
| Rewrite every anchor with current volatility | +38.50% / 0.91 / 267 | +26.20% / 3.13 / 133 |
| One seven-day return z-score | -0.34% / -0.01 / 210 | +8.88% / 1.00 / 106 |
| Delay one hour | +4.38% / 0.08 / 226 | +22.89% / 3.17 / 127 |
| Delay six hours | +16.28% / 0.47 / 226 | +19.97% / 2.28 / 127 |
| Delay seven days | -27.07% / -0.37 / 231 | -4.35% / -0.32 / 127 |

The direction and multi-anchor ensemble are real: the exact flip loses badly,
and one-horizon z-score is much weaker. However, frozen anchor volatility is
**not** the source of the aggregate result—the current-volatility rewrite is
stronger, though still fails fit stability. Breach fraction also beats excess
mass in 2023 but retains almost no fit efficiency. A one-hour lag improves 2023
ratio, so exact timestamp identification is broad.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | +17.43% / 0.32 | +32.12% / 4.06 |
| 1bp | +14.81% / 0.27 | +30.45% / 3.81 |
| 3bp | +9.73% / 0.17 | +27.18% / 3.32 |
| 6bp | +2.54% / 0.05 | +22.42% / 2.65 |
| 10bp | -6.33% / -0.11 | +16.35% / 1.85 |
| 15bp | -16.34% / -0.25 | +9.19% / 0.82 |

Cost erodes the effect, but zero-cost fit ratio is still only `0.32`; cost is
not the fundamental rejection reason.

## Decision

**Do not promote the static policy and do not open 2024+.** Preserve the
multi-anchor upper/lower excess mass, breach fraction and side balance as weak
beta context. The frozen/current volatility interpretations, histories, tails,
direction map and holds are gamma failure provenance.

This is a promising representation rather than an alpha: it has strong exact
direction evidence, useful 2023 breadth and materially more information than a
single return z-score, but regime instability and fit MDD make it untradeable.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_frozen_causal_cone_rupture_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_frozen_causal_cone_rupture_alpha.py
```

Artifacts:

- `training/search_frozen_causal_cone_rupture_alpha.py`
- `tests/test_search_frozen_causal_cone_rupture_alpha.py`
- `results/frozen_causal_cone_rupture_alpha_scan_2026-07-14.json`
