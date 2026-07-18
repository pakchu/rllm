# Cross-asset five-minute transfer evaluation — 2026-07-19

Preregistration: `5cacbead33f2b66c5961e22666708cde72f9844233821002b421a17a658b6775`
Source audit: `0112d2b9e53fe684710af16345bb9674462070f62889451fd50310cfb068a415`
Result: `55a4ffa0fd71d8467c2d70f5ef5f3006f4cc5a4427ea927870a294062ca4f895`

This is a five-minute regular-session evaluation. It is separate from the earlier daily-bar transfer check.
Thresholds are fit on train only; test/eval never select, rerank, or repair a policy.
REX thresholds use all positive finite train strength rows before applying the frozen 12-bar decision clock, matching the source scanner.

## Eval headline (5 bp per side)

| Policy | Asset | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Positive months | Weekly p | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| rex_htf_pullback_reclaim_5m | QQQ | 2.37% | 4.39% | 6.75% | 0.65 | 22 | 3/7 | 0.3527 | REJECT |
| rex_htf_pullback_reclaim_5m | 069500 | 27.57% | 56.35% | 25.44% | 2.21 | 30 | 4/7 | 0.2341 | REJECT |
| rex_htf_pullback_reclaim_5m | GLD | 10.80% | 20.70% | 15.01% | 1.38 | 33 | 5/7 | 0.1923 | REJECT |
| rex_multiscale_extreme_fade_5m | QQQ | -5.96% | -10.66% | 13.76% | -0.77 | 22 | 2/7 | 0.7404 | REJECT |
| rex_multiscale_extreme_fade_5m | 069500 | -38.36% | -58.85% | 41.80% | -1.41 | 33 | 2/7 | 0.9821 | REJECT |
| rex_multiscale_extreme_fade_5m | GLD | -9.91% | -17.44% | 15.97% | -1.09 | 12 | 2/7 | 0.8125 | REJECT |
| persistent_barrier_mass_density_fade_5m | QQQ | 6.49% | 12.23% | 5.49% | 2.23 | 12 | 4/7 | 0.1174 | REJECT |
| persistent_barrier_mass_density_fade_5m | 069500 | 18.15% | 35.80% | 18.51% | 1.93 | 8 | 3/7 | 0.2070 | REJECT |
| persistent_barrier_mass_density_fade_5m | GLD | -7.63% | -13.56% | 14.44% | -0.94 | 11 | 2/7 | 0.7285 | REJECT |

## Train / test / eval

| Policy | Asset | Split | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---|---|---:|---:|---:|---:|---:|
| rex_htf_pullback_reclaim_5m | QQQ | train | 6.71% | 8.14% | 12.30% | 0.66 | 18 |
| rex_htf_pullback_reclaim_5m | QQQ | test | -0.31% | -0.61% | 5.01% | -0.12 | 13 |
| rex_htf_pullback_reclaim_5m | QQQ | eval | 2.37% | 4.39% | 6.75% | 0.65 | 22 |
| rex_htf_pullback_reclaim_5m | 069500 | train | -12.77% | -15.18% | 16.10% | -0.94 | 31 |
| rex_htf_pullback_reclaim_5m | 069500 | test | 5.68% | 11.59% | 5.97% | 1.94 | 23 |
| rex_htf_pullback_reclaim_5m | 069500 | eval | 27.57% | 56.35% | 25.44% | 2.21 | 30 |
| rex_htf_pullback_reclaim_5m | GLD | train | 4.17% | 5.05% | 5.66% | 0.89 | 21 |
| rex_htf_pullback_reclaim_5m | GLD | test | -6.64% | -12.75% | 10.85% | -1.17 | 21 |
| rex_htf_pullback_reclaim_5m | GLD | eval | 10.80% | 20.70% | 15.01% | 1.38 | 33 |
| rex_multiscale_extreme_fade_5m | QQQ | train | -2.52% | -3.03% | 9.02% | -0.34 | 21 |
| rex_multiscale_extreme_fade_5m | QQQ | test | -2.97% | -5.81% | 7.16% | -0.81 | 23 |
| rex_multiscale_extreme_fade_5m | QQQ | eval | -5.96% | -10.66% | 13.76% | -0.77 | 22 |
| rex_multiscale_extreme_fade_5m | 069500 | train | -10.00% | -11.93% | 14.99% | -0.80 | 23 |
| rex_multiscale_extreme_fade_5m | 069500 | test | -19.16% | -34.44% | 21.81% | -1.58 | 22 |
| rex_multiscale_extreme_fade_5m | 069500 | eval | -38.36% | -58.85% | 41.80% | -1.41 | 33 |
| rex_multiscale_extreme_fade_5m | GLD | train | -8.13% | -9.71% | 10.69% | -0.91 | 23 |
| rex_multiscale_extreme_fade_5m | GLD | test | -9.45% | -17.89% | 20.28% | -0.88 | 21 |
| rex_multiscale_extreme_fade_5m | GLD | eval | -9.91% | -17.44% | 15.97% | -1.09 | 12 |
| persistent_barrier_mass_density_fade_5m | QQQ | train | -0.52% | -0.63% | 6.48% | -0.10 | 11 |
| persistent_barrier_mass_density_fade_5m | QQQ | test | -0.66% | -1.30% | 6.75% | -0.19 | 7 |
| persistent_barrier_mass_density_fade_5m | QQQ | eval | 6.49% | 12.23% | 5.49% | 2.23 | 12 |
| persistent_barrier_mass_density_fade_5m | 069500 | train | 1.32% | 1.59% | 5.43% | 0.29 | 12 |
| persistent_barrier_mass_density_fade_5m | 069500 | test | -7.54% | -14.42% | 10.27% | -1.40 | 10 |
| persistent_barrier_mass_density_fade_5m | 069500 | eval | 18.15% | 35.80% | 18.51% | 1.93 | 8 |
| persistent_barrier_mass_density_fade_5m | GLD | train | 3.95% | 4.78% | 6.27% | 0.76 | 12 |
| persistent_barrier_mass_density_fade_5m | GLD | test | -1.03% | -2.03% | 5.43% | -0.37 | 8 |
| persistent_barrier_mass_density_fade_5m | GLD | eval | -7.63% | -13.56% | 14.44% | -0.94 | 11 |

## Decision

No preregistered policy passed every five-minute gate on QQQ, KODEX 200, and GLD; cross-asset transfer is rejected.

## Execution and limitations

- Signal: completed 5-minute bar; entry: next available regular-session 5-minute open.
- Exit: fixed count of tradable 5-minute bars; missing KRX halt/no-trade rows are never synthesized.
- strict MDD includes entry cost, every held high/low in favorable-then-adverse order, and exit cost.
- CAGR uses the full wall-clock split, including idle periods. Absolute return is always shown.
- Prefix invariance is recomputed at frozen split boundaries and backed by causal-builder regression tests; it is not claimed as an exhaustive proof over every row.
- Investing.com TVC is an unofficial research source. Production replication requires an entitled feed and renewed parity checks.
- Official production candidates: IBKR historical bars and KIS Open API.

Official references:
- https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/
- https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/
- https://www.interactivebrokers.com/en/trading/krx-exchange.php
- https://apiportal.koreainvestment.com/apiservice
