# Alpha feature correlation report (2026-07-10)

Window: `2024-01-01` ~ `2026-06-02`; rows=253,909.

## Artifacts

- `continuous_pearson_csv`: `results/alpha_feature_correlation_2026-07-10/continuous_pearson.csv`
- `continuous_spearman_csv`: `results/alpha_feature_correlation_2026-07-10/continuous_spearman.csv`
- `component_phi_csv`: `results/alpha_feature_correlation_2026-07-10/component_phi.csv`
- `component_jaccard_csv`: `results/alpha_feature_correlation_2026-07-10/component_jaccard.csv`
- `continuous_spearman_heatmap_png`: `results/alpha_feature_correlation_2026-07-10/continuous_spearman_heatmap.png`
- `component_phi_heatmap_png`: `results/alpha_feature_correlation_2026-07-10/component_phi_heatmap.png`

## Continuous feature correlation

Top absolute Spearman pairs:

| rank | feature A | feature B | rho |
|---:|---|---|---:|
| 1 | `htf_1w_drawdown_4` | `weekly_drawdown_4w` | 1.0000 |
| 2 | `htf_1w_range_1` | `weekly_range_1w` | 1.0000 |
| 3 | `htf_1w_range_pos` | `weekly_range_pos` | 1.0000 |
| 4 | `htf_1w_return_1` | `weekly_return_1w` | 1.0000 |
| 5 | `htf_1w_return_4` | `weekly_return_4w` | 1.0000 |
| 6 | `rex_8640_cur_to_max_pct` | `rex_8640_max_to_cur_pct` | -1.0000 |
| 7 | `rex_144_cur_to_max_pct` | `rex_144_max_to_cur_pct` | -1.0000 |
| 8 | `rex_576_cur_to_max_pct` | `rex_576_max_to_cur_pct` | -1.0000 |
| 9 | `rex_2016_cur_to_max_pct` | `rex_2016_max_to_cur_pct` | -1.0000 |
| 10 | `rex_36_cur_to_max_pct` | `rex_36_max_to_cur_pct` | -1.0000 |
| 11 | `close_zscore_48` | `rex_36_range_pos` | 0.9156 |
| 12 | `rex_8640_cur_to_max_pct` | `rex_8640_range_pos` | 0.9051 |
| 13 | `rex_8640_max_to_cur_pct` | `rex_8640_range_pos` | -0.9051 |
| 14 | `rex_2016_max_to_cur_pct` | `rex_2016_range_pos` | -0.8715 |
| 15 | `rex_2016_cur_to_max_pct` | `rex_2016_range_pos` | 0.8715 |
| 16 | `close_zscore_48` | `trend_24` | 0.8433 |
| 17 | `rex_8640_cur_to_min_pct` | `rex_8640_range_pos` | 0.8367 |
| 18 | `rex_576_cur_to_max_pct` | `rex_576_range_pos` | 0.8292 |
| 19 | `rex_576_max_to_cur_pct` | `rex_576_range_pos` | -0.8292 |
| 20 | `bb_z` | `rex_36_range_pos` | 0.8255 |

## Component / candidate correlation

Top phi pairs:

| rank | component A | component B | phi |
|---:|---|---|---:|
| 1 | `long_range_funding_premium` | `long_minimal_funding_premium` | 0.9695 |
| 2 | `funding10_trend70` | `long_minimal_funding_premium` | 0.9321 |
| 3 | `long_funding_compression_premium` | `long_range_funding_compression` | 0.9183 |
| 4 | `funding10_trend70` | `long_range_funding_premium` | 0.9036 |
| 5 | `long_funding_compression_premium` | `long_minimal_funding_premium` | 0.9011 |
| 6 | `long_range_funding_premium` | `long_funding_compression_premium` | 0.8731 |
| 7 | `funding10_trend70` | `long_range_funding_compression` | 0.8637 |
| 8 | `funding10_trend70` | `long_funding_compression_premium` | 0.8398 |
| 9 | `long_range_funding_premium` | `long_range_funding_compression` | 0.8381 |
| 10 | `long_range_funding_compression` | `long_minimal_funding_premium` | 0.8055 |
| 11 | `short_premium_panic` | `short_premium_kimchi_union` | 0.7147 |
| 12 | `short_kimchi_unwind` | `short_premium_kimchi_union` | 0.7042 |
| 13 | `range_bb90` | `range_z70` | 0.4821 |
| 14 | `compress05_trend80` | `long_range_funding_compression` | 0.4332 |
| 15 | `compress05_trend80` | `long_funding_compression_premium` | 0.4213 |
| 16 | `premium20_mom90` | `long_minimal_funding_premium` | 0.3522 |
| 17 | `premium20_mom90` | `long_range_funding_premium` | 0.3414 |
| 18 | `premium20_mom90` | `long_funding_compression_premium` | 0.3173 |
| 19 | `short_fx_stress` | `short_kimchi_unwind` | 0.2615 |
| 20 | `range_bb90` | `long_range_funding_premium` | 0.2565 |

Top Jaccard overlaps:

| rank | component A | component B | Jaccard |
|---:|---|---|---:|
| 1 | `long_range_funding_premium` | `long_minimal_funding_premium` | 0.9420 |
| 2 | `funding10_trend70` | `long_minimal_funding_premium` | 0.8729 |
| 3 | `long_funding_compression_premium` | `long_range_funding_compression` | 0.8544 |
| 4 | `funding10_trend70` | `long_range_funding_premium` | 0.8223 |
| 5 | `long_funding_compression_premium` | `long_minimal_funding_premium` | 0.8187 |
| 6 | `long_range_funding_premium` | `long_funding_compression_premium` | 0.7794 |
| 7 | `funding10_trend70` | `long_range_funding_compression` | 0.7540 |
| 8 | `long_range_funding_premium` | `long_range_funding_compression` | 0.7298 |
| 9 | `funding10_trend70` | `long_funding_compression_premium` | 0.7146 |
| 10 | `long_range_funding_compression` | `long_minimal_funding_premium` | 0.6818 |
| 11 | `short_premium_panic` | `short_premium_kimchi_union` | 0.5194 |
| 12 | `short_kimchi_unwind` | `short_premium_kimchi_union` | 0.5046 |
| 13 | `range_bb90` | `range_z70` | 0.2687 |
| 14 | `compress05_trend80` | `long_range_funding_compression` | 0.1943 |
| 15 | `compress05_trend80` | `long_funding_compression_premium` | 0.1841 |
| 16 | `short_fx_stress` | `short_kimchi_unwind` | 0.1581 |
| 17 | `premium20_mom90` | `long_minimal_funding_premium` | 0.1281 |
| 18 | `premium20_mom90` | `long_range_funding_premium` | 0.1207 |
| 19 | `short_fx_stress` | `short_premium_kimchi_union` | 0.1086 |
| 20 | `premium20_mom90` | `long_funding_compression_premium` | 0.1049 |

## Component activity

| component | active rows | active frac |
|---|---:|---:|
| `mom85_pos50` | 18,065 | 7.11% |
| `long_funding_compression_premium` | 11,214 | 4.42% |
| `long_range_funding_compression` | 10,629 | 4.19% |
| `long_range_funding_premium` | 9,746 | 3.84% |
| `long_minimal_funding_premium` | 9,181 | 3.62% |
| `short_premium_kimchi_union` | 8,667 | 3.41% |
| `funding10_trend70` | 8,014 | 3.16% |
| `macro_usdkrw10_mom70` | 6,391 | 2.52% |
| `short_premium_panic` | 4,502 | 1.77% |
| `short_kimchi_unwind` | 4,373 | 1.72% |
| `short_fx_stress` | 3,889 | 1.53% |
| `compress05_trend80` | 2,065 | 0.81% |
| `range_z70` | 1,913 | 0.75% |
| `premium20_mom90` | 1,176 | 0.46% |
| `range_bb90` | 665 | 0.26% |

## Missing pool features

- `btc_ohlcv_taker_core`: `vwap_gap_z`, `rvol_z_24`, `rvol_z_72`, `qv_z_24`, `qv_z_72`, `spread_z_72`, `pos_288`, `range_compress_288`, `range_expand_72`, `px_ret_z_24`, `px_ret_z_72`, `px_ret_z_144`, `taker_mean_z_24`, `taker_mean_z_72`, `taker_div_72`, `taker_div_144`, `cvd_ret_z_72`, `cvd_ret_z_144`, `us_session`, `asia_session`
- `btc_oi_features`: `oi_available`, `oi_ret_z_24`, `oi_ret_z_72`, `oi_minus_px_z_72`, `px_minus_oi_z_72`, `btc_oi_unwind_long`, `btc_oi_squeeze_short`, `btc_liq_revert_long`, `btc_cvd_absorb_long`, `btc_overheat_short`, `oi_ret_4h_z`, `oi_squeeze_long_ctx`, `oi_squeeze_short_ctx`, `oi_unwind_long_ctx`, `oi_unwind_short_ctx`
- `btc_funding_premium_basis`: `funding_z`, `premium_z`, `basis_stress_z`

## Interpretation

- Long squeeze candidates are intentionally related; high phi/Jaccard confirms they should be treated as one family until marginal contribution tests prove otherwise.
- Continuous feature heatmap should be used to remove redundant raw numeric tokens before feeding an LLM/RLLM state card.
- Missing OI-derived features indicate the current cache used for this report does not expose those columns; use a DB/OI-enriched cache for a full derivatives-positioning correlation report.
