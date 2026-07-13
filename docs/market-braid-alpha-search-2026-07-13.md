# Three-Strand Market-Braid Alpha Search — 2026-07-13

## Thesis

A directional market impulse can propagate through three distinct strands:

1. cash spot price;
2. perpetual-futures price;
3. leveraged participation, witnessed jointly by expanding delayed open
   interest and side-aligned premium.

The hypothesis was that the **first-passage order itself** distinguishes an
organic cash-led repricing from a derivatives-led crowded move. Spot crossing
before the leverage witness therefore continues the impulse, while leverage
crossing before spot fades it. A stricter branch accepts only the exact
`spot -> perp -> leverage` and `leverage -> perp -> spot` chains.

This differs from the prior two-strand phase-slip experiment by freezing a
post-impulse origin, requiring a third leverage strand and treating the full
crossing permutation as the state object.

## Causal protocol

- Market, spot and premium source rows are physically cut before `2024-01-01`.
- Only complete five-row spot and one-minute-premium aggregates are usable.
- OI values and availability are delayed one complete 5-minute source bar.
- The impulse volatility and all 1-hour price, OI and premium passage scales
  use history ending at `t-1` and freeze when the episode starts.
- A passage is recorded only on the completed bar where it is first observed.
- Multiple first passages on the same 5-minute bar invalidate the episode;
  intrabar order is never inferred.
- 2020-10-15 through 2022 is fit; 2023 is inspected internal selection;
  2024+ remains sealed.
- Entry is the next 5-minute open, leverage is `0.5x`, implementation cost is
  `6bp/side`, and strict MDD uses favorable-first/adverse-second OHLC ordering.

## Bounded grid

Thirty-two policies:

- common spot/perp impulse: `2σ`, `3σ`;
- price passage: `0.5σ`, `1.0σ` of frozen prior 1-hour displacement;
- episode expiry: 6h, 12h;
- topology: exact chain or relative spot-versus-leverage order;
- hold: 6h, 12h.

The OI and premium passage widths were fixed at one prior 1-hour sigma. They
were not additional search dimensions.

## Best ranked policy

`2σ` impulse, `0.5σ` passage, 12h expiry, relative-order mapping and 12h hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +9.89% | +4.36% | 18.09% | 0.24 | 72 | 34/38 |
| 2023 | +8.28% | +8.28% | 4.26% | 1.94 | 25 | 15/10 |
| 2023H1 | +6.40% | +13.33% | 3.58% | 3.72 | 14 | 8/6 |
| 2023H2 | +1.77% | +3.54% | 4.26% | 0.83 | 11 | 7/4 |

Six of seven half-year segments were positive, but 2021H1 returned `-0.48%`.
Zero of 32 policies passed the required fit and 2023 CAGR/MDD `>=3` gate.

The top state began 3,966 impulses. At five-minute resolution, 2,848 episodes
had an unknowable same-bar passage tie and were conservatively discarded. Only
106 completed ordered states remained, producing 72 non-overlapping fit trades
and 25 non-overlapping 2023 trades. This resolution bottleneck is itself a
material limit.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -18.21% / -0.42 / 72 | -10.61% / -0.87 / 25 |
| Ignore order; continue every completed impulse | -17.12% / -0.29 / 72 | +7.53% / 1.65 / 25 |
| Remove leverage strand | -6.47% / -0.13 / 194 | +3.60% / 0.53 / 69 |
| OI-only witness | +11.26% / 0.47 / 114 | +7.60% / 1.09 / 43 |
| Premium-only witness | -18.56% / -0.38 / 132 | +4.62% / 0.69 / 45 |
| Delay selected signal by one hour | +10.73% / 0.32 / 72 | +8.72% / 2.05 / 25 |

The exact flip and order-blind controls support some directional information,
and the joint witness improves 2023 risk efficiency over either single witness.
However, a one-hour delay improves both periods. The state is broad and slowly
resolved rather than a sharply timed first-passage execution edge.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | +14.75% / 0.38 | +9.91% / 2.33 |
| 3bp | +12.29% / 0.31 | +9.09% / 2.14 |
| 6bp | +9.89% / 0.24 | +8.28% / 1.94 |
| 10bp | +6.77% / 0.16 | +7.20% / 1.69 |

The effect is not erased by costs; insufficient risk efficiency, low event
count, 2023H2 weakness and coarse timestamp resolution are the rejection
reasons.

## Decision

**Do not promote to alpha and do not open 2024+.** Preserve episode age,
permutation and relative spot/leverage order only as weak beta context. Record
the exact static policies as gamma failure provenance. Do not tune passage
widths, expiries, sequence mappings or holds again on this inspected sample.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_market_braid_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_market_braid_alpha.py
```

Artifacts:

- `training/search_market_braid_alpha.py`
- `tests/test_search_market_braid_alpha.py`
- `results/market_braid_alpha_scan_2026-07-13.json`
