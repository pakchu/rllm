# Cross-asset alpha transfer preregistration — 2026-07-19

Manifest: `3e89889f79215fc383fc3375119d774a62d6c765b7d53043384a5c2597d86056`

## Claim boundary

The gross-8 sleeves cannot be ported exactly because every sleeve uses at least one crypto-native input. 
This battery tests three fixed daily OHLCV translations only; it does not relabel them as the original BTC strategies.

## Instruments and splits

- QQQ (Nasdaq-100 ETF), 069500.KS (KODEX 200), GLD (gold ETF).
- Train: 2007-01-29 through 2016-12-31.
- Test: 2017-01-01 through 2021-12-31.
- Eval: 2022-01-01 through 2026-07-18.
- Thresholds are fit per instrument on train only. Test/eval cannot select or repair a policy.

## Frozen translated policies

- `rex_pullback_reclaim_session`: same multiscale location/pullback/reclaim algebra as REX, but vol_confirm uses only max(0, volume_zscore); no taker, OI, funding, premium, FX, or Kimchi input
- `rex_multiscale_extreme_fade_session`: max(0,abs(mean_range_pos)-0.55)*(1+abs(short_location-long_location))
- `persistent_barrier_mass_density_fade_session`: {'horizon_5m_bars': 2016, 'variant': 'mass_density', 'tail_quantile': 0.975, 'hold_5m_bars': 288, 'direction_mode': 'fade'}

## Admission

One policy must independently pass all QQQ/KODEX200/GLD eval gates: positive base and 10 bp/side stress return, 
CAGR/strict-MDD >= 3, strict MDD <= 15%, at least 20 trades, >=60% positive eval years, and a weaker direction flip.

No outcome field was downloaded or inspected when this manifest was generated.
