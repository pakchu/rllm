# Cross-asset five-minute transfer preregistration — 2026-07-19

Manifest: `5cacbead33f2b66c5961e22666708cde72f9844233821002b421a17a658b6775`

## Scope

This is a five-minute OHLCV projection of three previously researched mechanisms. It is not an exact port of crypto-native gross-8 sleeves and cannot validate inputs such as taker flow, open interest, funding, premium, FX, or Kimchi premium.

- Instruments: QQQ, KODEX 200 (`069500`), and GLD. No KOSPI strategy is evaluated.
- Bar clock: completed regular-session five-minute bars; next-bar-open execution.
- Source is frozen before policy outcomes are computed.

## Frozen data and splits

- Primary research source: Investing.com TVC chart service (unofficial and unsupported for production).
- Raw period: 2024-03-05T00:00:00Z through 2026-07-19T00:00:00Z exclusive.
- Train: 2024-09-01 to 2025-07-01 exclusive.
- Test: 2025-07-01 to 2026-01-01 exclusive.
- Eval: 2026-01-01 to 2026-07-19 exclusive.
- Recent matching timestamps must pass the frozen Yahoo close-parity control.

## Frozen policies

- `rex_htf_pullback_reclaim_5m`: q=0.75, hold=144 five-minute bars.
- `rex_multiscale_extreme_fade_5m`: q=0.75, hold=144 five-minute bars.
- `persistent_barrier_mass_density_fade_5m`: q=0.975, hold=288 five-minute bars.

## Admission

A policy transfers only if it independently passes every frozen eval gate on QQQ, KODEX 200, and GLD.
Test/eval cannot tune thresholds, select a row, repair direction, or substitute a policy.
