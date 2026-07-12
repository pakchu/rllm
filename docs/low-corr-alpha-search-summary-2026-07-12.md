# Low-correlation alpha search summary (2026-07-12)

Objective: find alpha candidates from features with low correlation to existing alpha component masks.

## Protocol

- Existing alpha reference: current explicit component masks (`range_bb90`, `funding10_trend70`, `premium20_mom90`, short FX/premium/kimchi components, and their unions).
- Correlation filter: train-window only (`2020-01-01` to `2024-01-01`), max absolute correlation to existing component masks generally <= `0.12` for the new scan.
- Thresholds: train-only quantiles.
- Backtest: strict path MDD, non-overlapping events, full-window CAGR including idle time.
- Caveat: ranking/seed selection is diagnostic because the final candidates were chosen after inspecting OOS columns. Do not treat as live-grade.

## Main finding

The lowest-correlation source axes are different from the existing REX/funding/premium family:

1. Alpha101-style intraday primitives: `a_ret_z_*`, `a_ret_vol_corr_*`, `a_vwap_gap_z`, `a_absret_vol_rank`, session-time tokens.
2. VPIN/orderflow primitives: `vp_ret_rank_72`, `vp_imb_z_144`, `vp_vpin_z_72`.
3. OI/price divergence: `oi_minus_px_z_288`, `px_minus_oi_z_288`.

However, the new low-correlation candidates are **not live-grade standalone alphas yet**. They are useful as weak diversifying alpha features / RLLM state tokens.

## Best weak candidate: Alpha101 early pullback long

Source: `a101q_449b220137` from `results/alpha101_random_quantile_standalone_2026-07-09.json`.

- Side: long
- Hold/stride: `144/24`
- Max absolute correlation to existing alpha components: `0.116`
- Entry terms:
  - `a_ret_z_12 <= -0.1039416971`
  - `a_early_session >= 0.5`
  - `a_ret_vol_corr_288 <= 0.0102425402`
  - `a_absret_vol_rank <= 0.4943800277`

| period | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| train 2020-2023 | 77.76% | 15.47% | 32.22% | 0.48 | 222 |
| test 2024 | 21.93% | 21.88% | 8.84% | 2.48 | 58 |
| eval 2025 | 9.42% | 9.43% | 7.98% | 1.18 | 55 |
| ytd 2026 | 6.71% | 13.62% | 4.46% | 3.06 | 26 |

TP/SL refine did not solve the ratio bottleneck. Best checked variant (`TP=10%`, no SL) improved 2025 but still failed target:

| period | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| train 2020-2023 | 77.03% | 15.35% | 32.22% | 0.48 | 222 |
| test 2024 | 21.93% | 21.88% | 8.84% | 2.48 | 58 |
| eval 2025 | 12.21% | 12.22% | 7.98% | 1.53 | 55 |
| ytd 2026 | 6.71% | 13.62% | 4.46% | 3.06 | 26 |

## Pair scan result

Restricted pair scan among top low-correlation features found several low-corr short candidates, but they are also weak:

Best pair:

- `a_ret_vol_corr_288 >= q0.90` and `vx_lowtox_momo_short <= q0.10`
- Side: short, hold/stride `144/24`
- Max abs corr to existing components: `0.101`

| period | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| train 2020-2023 | -36.62% | -10.77% | 37.70% | -0.29 | 109 |
| test 2024 | 4.33% | 4.32% | 5.72% | 0.76 | 19 |
| eval 2025 | 3.73% | 3.73% | 4.85% | 0.77 | 20 |
| ytd 2026 | 1.86% | 4.58% | 6.82% | 0.67 | 19 |

## Conclusion

- Found a genuinely low-correlation feature family, but standalone alpha strength is weak.
- The best current use is as a diversifying **alpha-feature candidate** for RLLM/portfolio meta-selection, not as an executable sleeve.
- Next productive direction: use Alpha101/orderflow/OI divergence tokens as context to gate stronger existing sleeves, rather than forcing them as standalone entries.
