# Barrier Constitution Alpha Search — 2026-07-13

## Question

Can a price barrier remember **how it was created**, not merely where it is?

The parent nested-barrier experiment retained the creation index of prior-only
12-hour, 2-day and 1-week extrema. This follow-up gives that ancestor a
"constitution":

- a high created by a close near the candle high is treated as accepted upward
  discovery;
- a high created by a close near the candle low is treated as rejected inventory;
- low barriers use the symmetric interpretation;
- depleted direction-specific taker work on revisit follows accepted discovery;
- reinforced work on revisit fades rejected inventory.

This is intentionally different from assigning one universal continuation/fade
meaning to every rolling extreme.

## Causal protocol

- Source rows are physically cut strictly before `2024-01-01`.
- Every rolling extreme excludes decision bar `t`.
- The longest touched horizon owns the ancestor candle and origin work.
- The origin candle close-location value and taker work are already completed
  history when a later revisit is evaluated.
- The current completed 5-minute bar creates a signal for the **next** open.
- Position size is `0.5x`; implementation cost is `6bp/side`.
- MDD uses favorable-first/adverse-second OHLC high-water accounting.
- `2024+` OOS was not opened.

## Frozen search

Eight policies were evaluated:

- minimum coalescence: `2`, `3` horizons;
- touch width: `10bp`, `20bp`;
- hold: `6h`, `12h`;
- fixed origin CLV threshold: `|CLV| >= 0.5`;
- fixed work ratios: depleted `<= 0.75`, reinforced `>= 1.25`.

The preflight admission rule remained CAGR/strict-MDD `>= 3` in both fit and
2023, non-negative 2023 halves, and sufficient trade support.

## Best ranked policy

Three-scale coalescence, `20bp` touch and `12h` hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +23.79% | +8.61% | 18.80% | 0.46 | 104 | 52/52 |
| 2023 | +5.70% | +5.71% | 7.22% | 0.79 | 26 | 16/10 |
| 2023H1 | +0.53% | +1.07% | 7.22% | 0.15 | 16 | 9/7 |
| 2023H2 | +5.14% | +10.47% | 6.95% | 1.51 | 10 | 7/3 |

Fit half-years were positive in four of five windows, but `2022H1` returned
`-11.41%` with `-1.32` CAGR/MDD. Aggregate positivity therefore does not imply
stable alpha.

## Falsification controls

At the selected policy and standard costs:

- exact direction flip: fit `-31.13%`, 2023 `-8.99%`;
- inverted ancestor-candle semantics: fit `-32.01%`, but 2023 `+14.61%`;
- ignored ancestor candle: fit `-6.67%`, 2023 `+5.53%`;
- zero-cost selected policy: fit `+31.76%`, ratio `0.63`; 2023 `+7.36%`, ratio
  `1.06`.

Direction and ancestor information matter in-sample, but the inverted-origin
control reversing from a large fit loss to a positive 2023 result exposes a
regime-dependent mapping. Costs are not the sole failure: even zero-cost risk
efficiency remains far below admission.

## Decision

**Rejected as alpha.** Zero of eight policies passed preflight, so OOS stayed
sealed. Do not tune nearby CLV thresholds, work ratios, widths or holds on this
sample. The exact constitution rule belongs in gamma failure provenance; the
parent continuous barrier ancestry remains only weak beta context.

Artifacts:

- `training/search_barrier_constitution_alpha.py`
- `tests/test_search_barrier_constitution_alpha.py`
- `results/barrier_constitution_alpha_scan_2026-07-13.json`
