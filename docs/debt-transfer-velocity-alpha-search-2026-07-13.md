# Debt Transfer Velocity alpha preflight — 2026-07-13

## Hypothesis

A late leveraged receiver should be fragile when four causal conditions coincide:

1. new perpetual OI debt is created;
2. global-account positioning and aggressive futures flow rapidly change owner;
3. futures OI value expands faster than completed Binance spot notional liquidity;
4. price has already moved in the receiver's direction (acceptance).

The signed transfer impulse is `positive Δlog(OI) × Δownership`; its EWMA is
Debt Transfer Velocity (DTV). The policy fades the receiver only when transfer
intensity, spot cash gap and price acceptance are jointly positive. This is a
path product over debt creation and ownership change, not an OI/funding level.
The originally proposed revised stablecoin history was deliberately replaced by
strictly point-in-time Binance spot notional to avoid revision leakage.

## Causal protocol

- Futures, spot and metrics sources were physically truncated before 2024.
- Binance metrics were delayed by one complete 5-minute source bar.
- Only the global-account ratio was used; 2022 top-trader fields with 12.7%
  coverage were excluded. Required 2022 coverage was global ratio 94.91%, OI
  99.89%, OI value 99.89%, complete spot 100%.
- Spot inputs require five complete 1-minute rows per 5-minute bar.
- All z-scores use histories shifted through `t-1`; features at completed bar
  `t` execute at `open(t+1)`.
- Fit thresholds use 2020-10-15 through 2022 only; 2023/H1/H2 select. OOS was
  not opened.
- 0.5x, 6bp/side, fixed non-overlapping holds and conservative strict OHLC MDD.

Grid: transfer half-life `{6h,24h}` × price-acceptance horizon `{6h,24h}` ×
fit tail `{q90,q95}` × hold `{6h,12h,24h}` = 24 fixed candidates.

## Result

Best adequately populated candidate: 24h transfer memory, 24h acceptance, q90,
6h hold. Metric format is `absolute return / CAGR / strict MDD / ratio / trades`.

| Window | Result |
|---|---:|
| Fit through 2022 | `-3.71 / -1.69 / 12.48 / -0.14 / 80` |
| 2023 | `-2.39 / -2.39 / 5.85 / -0.41 / 31` |
| 2023 H1 | `-1.27 / -2.55 / 5.10 / -0.50 / 17` |
| 2023 H2 | `-1.13 / -2.23 / 1.96 / -1.14 / 14` |

Only three of six robustness half-years were positive. No sufficiently traded
candidate had a positive minimum core ratio.

## Matched controls for the selected timing

| Control | Fit result | 2023 result |
|---|---:|---:|
| Direction flip | `-6.32 / -2.91 / 15.15 / -0.19 / 80` | `-1.44 / -1.44 / 4.07 / -0.35 / 31` |
| Ownership level, no transfer derivative | `-6.82 / -3.14 / 8.60 / -0.37 / 65` | `-2.84 / -2.84 / 4.16 / -0.68 / 11` |
| Remove spot cash gap | `-27.77 / -13.67 / 36.82 / -0.37 / 184` | `-1.53 / -1.53 / 8.99 / -0.17 / 61` |
| Stale owner change by 24h | `-14.93 / -7.05 / 19.14 / -0.37 / 77` | `+0.85 / +0.85 / 2.33 / 0.37 / 32` |

Both the intended direction and its exact flip lose after costs, so the failure
is not just a sign mistake. Removing cash worsens the result but does not reveal
an edge.

## Decision

**Reject in preflight; do not open OOS or promote DTV as an alpha.** Preserve
only the experiment provenance. Any retry needs a different ownership observable
or actual cohort burden; tuning these tails/memories would be data mining.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_debt_transfer_velocity_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_debt_transfer_velocity_alpha.py
```
