# Existing alpha + volume gate scan (2026-07-08)

## Scope

Volume was weak as a standalone alpha, so this scan applies volume/flow gates as allow-filters on existing fixed alphas.

Tested base alphas:

- `nonpb30_taker` from `configs/live/nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json`
- `oi_pullback` from `configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json`
- `oi_highfreq` from `results/oi_llm_selector_eval_2026-07-06.json` candidate gates

Volume gate families:

- BTC volume/quote-volume/trade-count z-score and momentum
- BTC taker flow z-score
- Upbit KRW-BTC volume and Upbit/Binance relative volume
- Altcoin quote-volume z-scores and alt/BTC quote-volume ratio

All volume thresholds are train `<2024` quantiles. Ranking below is still discovery/diagnostic, not clean OOS promotion.

## Important caveat

This script replays the candidate gates from local feature reconstruction on the 2026-07-05 extended cache. Some baseline stats differ from older exact-verification artifacts, so use this as relative gate-comparison evidence. A promotion candidate must be rechecked through the exact verifier/live path.

Raw evidence:

- `results/existing_alpha_volume_gate_scan_2026-07-08.json`

## Main findings

### 1. `oi_pullback` improves materially with volume-momentum gates

Baseline in this replay was weak in eval/2026:

| split | return | CAGR | MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| test 2024 | 50.32% | 50.19% | 12.57% | 3.99 | 80 |
| eval 2025 | -5.25% | -5.25% | 17.53% | -0.30 | 39 |
| ytd 2026 | -7.17% | -13.59% | 12.94% | -1.05 | 25 |

Best volume gate by 2026 rescue:

```text
allow if gate_vol_mom_288 >= train q0.80 = 1.2952175561045105
```

| split | return | CAGR | MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | 29.84% | 29.77% | 13.39% | 2.22 | 48 | 58.3% |
| eval 2025 | 2.97% | 2.97% | 16.23% | 0.18 | 30 | 53.3% |
| ytd 2026 | 13.76% | 28.81% | 4.00% | 7.20 | 11 | 72.7% |

Interpretation: not enough to restore 2025 strength, but it flips 2026 from bad to strongly positive while keeping reasonable trade count.

Stricter variant:

```text
allow if gate_vol_mom_288 >= train q0.90 = 2.7580129647434717
```

| split | return | CAGR | MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | 25.81% | 25.75% | 12.86% | 2.00 | 36 | 61.1% |
| eval 2025 | 0.38% | 0.38% | 13.18% | 0.03 | 21 | 52.4% |
| ytd 2026 | 11.60% | 24.04% | 1.61% | 14.96 | 7 | 71.4% |

This is a strong risk filter but likely too sparse.

### 2. `oi_pullback` + alt/BTC volume ratio gives cleaner test/eval but fewer trades

```text
allow if gate_alt_btc_qv_ratio_z_288 >= train q0.90 = 0.2041228149141364
```

| split | return | CAGR | MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | 16.14% | 16.10% | 4.48% | 3.59 | 18 | 72.2% |
| eval 2025 | 11.79% | 11.79% | 5.41% | 2.18 | 16 | 68.8% |
| ytd 2026 | 5.62% | 11.33% | 1.52% | 7.44 | 5 | 60.0% |

Interpretation: attractive as a selector/state feature, but too few trades for standalone gate promotion.

### 3. `oi_highfreq` benefits from Upbit/Binance relative-volume gates

Baseline in this replay:

| split | return | CAGR | MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| test 2024 | 16.58% | 16.54% | 13.49% | 1.23 | 214 |
| eval 2025 | -4.88% | -4.88% | 9.96% | -0.49 | 129 |
| ytd 2026 | 2.68% | 5.32% | 7.39% | 0.72 | 60 |

Most balanced candidate:

```text
allow if gate_alt_btc_qv_ratio_z_72 >= train q0.80 = 0.0
```

| split | return | CAGR | MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | 8.44% | 8.43% | 7.38% | 1.14 | 116 | 54.3% |
| eval 2025 | 10.24% | 10.25% | 7.41% | 1.38 | 61 | 62.3% |
| ytd 2026 | 3.78% | 7.57% | 5.17% | 1.46 | 31 | 48.4% |

Upbit-specific candidate:

```text
allow if gate_upbit_binance_vol_ratio_z_288 <= train q0.20 = -0.6044648574076597
```

| split | return | CAGR | MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | 2.13% | 2.13% | 11.08% | 0.19 | 84 | 46.4% |
| eval 2025 | 3.20% | 3.20% | 4.23% | 0.76 | 44 | 52.3% |
| ytd 2026 | 5.08% | 10.22% | 3.21% | 3.19 | 20 | 70.0% |

Interpretation: Upbit relative-volume helps 2026 risk, but return is too low. Alt/BTC ratio is more promising.

### 4. `nonpb30_taker` volume gates were mostly too sparse

Volume gates can flip 2026 positive, but they reduce trades to single digits in many cases. The best-looking nonpb30 variants are probably not robust enough without a separate selector.

Example:

```text
allow if gate_ADAUSDT_qv_z_72 >= train q0.90
```

| split | return | CAGR | MDD | trades |
|---|---:|---:|---:|---:|
| test 2024 | 2.41% | 2.40% | 2.69% | 14 |
| eval 2025 | 3.84% | 3.84% | 3.74% | 10 |
| ytd 2026 | 1.41% | 2.79% | 1.78% | 4 |

Too sparse for promotion.

## Decision

The useful direction is not standalone volume alpha, but a context selector for OI alphas:

1. Add `gate_vol_mom_288` and `gate_qvol_mom_288` to OI selector state.
2. Add `gate_alt_btc_qv_ratio_z_72/288` to OI selector state.
3. Add `gate_upbit_binance_vol_ratio_z_144/288` as secondary risk state.
4. Do not hard-gate nonpb30 with volume yet; too sparse.

Best next experiment: extend the existing OI LLM/symbolic selector card with these volume state tokens and fit allow/block contexts on train events, instead of using a single hard threshold gate.
