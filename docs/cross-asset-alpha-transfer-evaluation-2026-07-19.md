# QQQ / KODEX200 / GLD alpha-transfer evaluation — 2026-07-19

Preregistration: `3e89889f79215fc383fc3375119d774a62d6c765b7d53043384a5c2597d86056`

The original gross-8 sleeves remain nonportable. These are fixed daily OHLCV translations, not exact ports.

## Exact gross-8 portability

| Sleeve | Exact port | Blocking market-specific inputs |
|---|:---:|---|
| `fresh_kimchi_fx` | NO | BTC funding, Kimchi premium, USDKRW, BTC flow |
| `frozen_annual_rank7` | NO | BTC funding/premium event clock, BTC-fitted 40-feature model |
| `rex_taker_low_range_position` | NO | crypto aggressor-side taker imbalance, BTC-fitted threshold |
| `cand_rex_veto_7` | NO | BTC open interest, BTC-fitted threshold |
| `markov_transition_long` | NO | BTC funding/premium event clock, BTC-fitted transition states |

Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades / positive-year share`.

## Eval 2022–2026-07-18

| Policy | Asset | Base 5 bp/side | Stress 10 bp/side | Direction flip ratio | Gate |
|---|---|---:|---:|---:|:---:|
| `rex_pullback_reclaim_session` | QQQ | 19.07% / 3.92% / 14.09% / 0.28 / 35 / 20% | 14.97% / 3.12% / 15.70% / 0.20 / 35 / 20% | -0.18 | FAIL |
| `rex_pullback_reclaim_session` | 069500.KS | -13.41% / -3.12% / 26.08% / -0.12 / 31 / 40% | -16.05% / -3.78% / 27.54% / -0.14 / 31 / 40% | 0.06 | FAIL |
| `rex_pullback_reclaim_session` | GLD | 8.31% / 1.77% / 9.69% / 0.18 / 30 / 40% | 5.11% / 1.10% / 11.51% / 0.10 / 30 / 20% | -0.19 | FAIL |
| `rex_multiscale_extreme_fade_session` | QQQ | -23.89% / -5.83% / 38.55% / -0.15 / 81 / 20% | -29.82% / -7.49% / 42.88% / -0.17 / 81 / 20% | 0.09 | FAIL |
| `rex_multiscale_extreme_fade_session` | 069500.KS | -44.62% / -12.19% / 47.92% / -0.25 / 46 / 0% | -47.11% / -13.08% / 50.22% / -0.26 / 46 / 0% | 0.89 | FAIL |
| `rex_multiscale_extreme_fade_session` | GLD | -10.79% / -2.48% / 29.01% / -0.09 / 94 / 40% | -18.80% / -4.48% / 34.80% / -0.13 / 94 / 20% | -0.12 | FAIL |
| `persistent_barrier_mass_density_fade_session` | QQQ | -4.14% / -0.93% / 6.50% / -0.14 / 7 / 40% | -4.81% / -1.08% / 7.06% / -0.15 / 7 / 20% | 0.23 | FAIL |
| `persistent_barrier_mass_density_fade_session` | 069500.KS | -6.02% / -1.36% / 19.06% / -0.07 / 7 / 20% | -6.67% / -1.51% / 19.55% / -0.08 / 7 / 20% | 0.07 | FAIL |
| `persistent_barrier_mass_density_fade_session` | GLD | -3.60% / -0.80% / 8.36% / -0.10 / 8 / 20% | -4.36% / -0.98% / 8.63% / -0.11 / 8 / 20% | 0.07 | FAIL |

## Train / Test / Eval stability

Cells: `absolute return / CAGR-MDD / trades`.

| Policy | Asset | Train 2007–2016 | Test 2017–2021 | Eval 2022–2026 |
|---|---|---:|---:|---:|
| `rex_pullback_reclaim_session` | QQQ | 23.96% / 0.10 / 69 | 7.60% / 0.09 / 46 | 19.07% / 0.28 / 35 |
| `rex_pullback_reclaim_session` | 069500.KS | -13.32% / -0.07 / 55 | 4.40% / 0.11 / 35 | -13.41% / -0.12 / 31 |
| `rex_pullback_reclaim_session` | GLD | -1.00% / -0.01 / 86 | -8.16% / -0.19 / 19 | 8.31% / 0.18 / 30 |
| `rex_multiscale_extreme_fade_session` | QQQ | -23.74% / -0.07 / 171 | -31.06% / -0.21 / 126 | -23.89% / -0.15 / 81 |
| `rex_multiscale_extreme_fade_session` | 069500.KS | -4.83% / -0.04 / 89 | -25.00% / -0.17 / 47 | -44.62% / -0.25 / 46 |
| `rex_multiscale_extreme_fade_session` | GLD | -13.10% / -0.05 / 127 | -4.04% / -0.05 / 64 | -10.79% / -0.09 / 94 |
| `persistent_barrier_mass_density_fade_session` | QQQ | 28.59% / 0.19 / 16 | 4.57% / 0.25 / 5 | -4.14% / -0.14 / 7 |
| `persistent_barrier_mass_density_fade_session` | 069500.KS | -6.43% / -0.08 / 11 | -6.38% / -0.18 / 5 | -6.02% / -0.07 / 7 |
| `persistent_barrier_mass_density_fade_session` | GLD | -3.86% / -0.04 / 14 | -0.84% / -0.02 / 11 | -3.60% / -0.10 / 8 |

## Frozen decision

- `rex_pullback_reclaim_session`: **REJECT**
- `rex_multiscale_extreme_fade_session`: **REJECT**
- `persistent_barrier_mass_density_fade_session`: **REJECT**

No test/eval row selected or repaired another policy. Raw provider payloads remain local; source URL, SHA256, row range, and timezone are recorded in the JSON result.

## Data limitation

This battery uses the Yahoo Finance chart API as one common adjusted-OHLCV source. It is not an official exchange feed.
Before production claims, reproduce QQQ against [Nasdaq history](https://www.nasdaq.com/market-activity/etf/qqq/historical), KODEX 200 against [KRX Data Marketplace](https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?locale=en), and GLD against an independent exchange series.
[SSGA](https://www.ssga.com/us/en/intermediary/etfs/spdr-gold-shares-gld) provides official fund/NAV information but not the exchange OHLCV used here.
