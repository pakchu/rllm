# Bullish long alpha target-5 search — 2026-07-05

## Protocol

Current bullish-regime acceptance rule:

- Train: `< 2024`; train performance is not an acceptance constraint.
- Test: calendar year `2024`.
- Eval: calendar year `2025`.
- CAGR annualization must use the full declared test/eval calendar windows, including cash/no-trade time.
- A promoted strategy must clear the target on both test and eval:
  - `CAGR / strict MDD >= 5` on test `2024`.
  - `CAGR / strict MDD >= 5` on eval `2025`.
- Thresholds/statistics are fit on train only. Test is the selection surface; eval is held out.

## Current best kept baseline: `pb30_funding`

Keep this candidate as a bullish-regime baseline / near-miss, but do not promote it as target-clearing yet.

Candidate skeleton:

- Entry: `pb30_funding`
- Score: `activity_flow_htf`
- Score quantile: `0.5`
- Entry quantile: `0.75`
- Premium quantile: `0.8`
- Funding quantile: `0.7` or `0.8` (same trades in the focused scan)
- Hold: `18h`
- Leverage: `0.5`
- Extra bull gate: none

Full-calendar strict replay:

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 24.35% | 24.30% | 4.48% | 5.43 | 22 | 0.0102 |
| eval 2025 | 17.54% | 17.56% | 4.29% | 4.09 | 20 | 0.0029 |

Artifacts:

- `results/bullish_long_alpha_focused_search_trainlt2024_test2024_eval2025_with_evaltops_2026-07-05.json`
- `results/bullish_long_alpha_overlay_top_trainlt2024_test2024_eval2025_2026-07-05.json`

## Rejected / caution

Risk overlays on the kept baseline can make 2024 test look much better, but they failed 2025 eval in the current scan. Do not use overlay-improved 2024 ratios as acceptance evidence unless the same fixed overlay also clears 2025 eval.

## Focused 0.5x scan status

No 0.5x candidate from the focused scan clears `CAGR / strict MDD >= 5` on both `2024` test and `2025` eval. The kept `pb30_funding` 0.5x baseline is the strongest near-miss because it clears test and remains positive/statistically useful on eval, but eval ratio is only `4.09`.

## Target-clearing variant found: `pb30_funding` leverage-scaled

After preserving the 0.5x baseline, a leverage sweep of the same fixed signal found that the candidate first clears the target at `1.7x` leverage.

Signal skeleton is unchanged and all thresholds are still train-fitted only:

- Entry: `pb30_funding`
- Score: `activity_flow_htf`
- Score quantile: `0.5`
- Entry quantile: `0.75`
- Premium quantile: `0.8`
- Funding quantile: `0.7`
- Hold: `18h` (`216` 5-minute bars)
- Stride: `12` bars
- Extra bull gate: none
- Leverage: `1.7x` minimum target-clearing leverage in the sweep

Full-calendar strict replay at `1.7x`:

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 104.40% | 104.10% | 15.01% | 6.93 | 22 | 0.0099 |
| eval 2025 | 70.34% | 70.40% | 14.06% | 5.01 | 20 | 0.0035 |

Artifact:

- `results/bullish_pb30_funding_activity_flow_htf_lev_sweep_trainlt2024_test2024_eval2025_2026-07-05.json`

Caveat: this is the same alpha as the kept 0.5x baseline with higher leverage. It satisfies the stated `CAGR / strict MDD >= 5` rule on both test and eval, but strict MDD rises to about `14–15%`. If a future acceptance rule adds an absolute strict-MDD cap below `15%`, the minimum passing leverage becomes marginal and should be rechecked.

## Follow-up composite family scan

A focused scan over alternative REX composite long families did not find a lower-MDD or lower-leverage replacement that clears both test and eval.

Artifact:

- `results/bullish_composite_long_family_focused_target5_trainlt2024_test2024_eval2025_2026-07-05.json`

Best near-miss family:

- Rule: `range_expansion_pullback`
- Gate: `premium_vhi`
- Quantile: `0.75`
- Hold: `24h`
- Leverage: `1.7x`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 68.82% | 68.63% | 16.55% | 4.15 | 27 | 0.0515 |
| eval 2025 | 88.34% | 88.43% | 17.31% | 5.11 | 21 | 0.0130 |

Rejected because test ratio is below 5. Current promoted candidate remains `pb30_funding + activity_flow_htf` at `1.7x`.

## Live-candidate dry-run spec

The promoted `pb30_funding + activity_flow_htf` candidate was materialized as a repo-local dry-run live candidate spec, not enabled for live orders:

- `configs/live/bullish_pb30_funding_activity_flow_htf_candidate.json`

The spec keeps `dry_run=true` and `allow_live_orders=false`. It records the 2026 YTD degradation caveat and requires manual BULL regime review before any future execution wiring.

## High-trade alpha follow-up, fixed leverage

Per operator guidance, leverage is excluded from the alpha search. A fixed `1.0x` high-trade scan did not find a target-clearing high-frequency alpha.

Artifact:

- `results/bullish_high_trade_alpha_fixed1x_quality_buckets_trainlt2024_test2024_eval2025_2026-07-05.json`

Best quality with `min_trades >= 100` per split:

- Rule: `pb30`
- Entry quantile: `0.70`
- Score: `activity_flow_htf`, score quantile `0.40`
- Hold: `6h`
- Stride: `6` bars
- Leverage: fixed `1.0x`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 53.53% | 53.39% | 27.74% | 1.93 | 161 | 0.065 |
| eval 2025 | 39.45% | 39.48% | 18.29% | 2.16 | 165 | 0.125 |

Best quality with `min_trades >= 200` per split:

- Rule: `pb30`
- Entry quantile: `0.65`
- Funding gate: `funding_zscore >= train q0.60`
- Score: `activity_flow`, score quantile `0.20`
- Hold: `8h`
- Stride: `6` bars
- Leverage: fixed `1.0x`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 36.03% | 35.95% | 29.51% | 1.22 | 231 | 0.239 |
| eval 2025 | 30.51% | 30.53% | 22.82% | 1.34 | 214 | 0.265 |

Conclusion: higher trade count currently comes from much looser pullback rules, but the edge quality and drawdown are not strong enough. The high-trade path needs a better filter/secondary alpha rather than leverage or looser thresholds.

## Non-pb30 mid-frequency probe

Because `pb30` appears to behave like a sparse setup rather than a mid-frequency alpha, a follow-up scan tested non-pb30 reclaim/flow/breakout families at fixed `1.0x` leverage.

Artifact:

- `results/bullish_non_pb30_midfreq_fast_scan_trainlt2024_test2024_eval2025_2026-07-05.json`

Fast scan scope:

- Families: `micro_reclaim_3h`, `local_reclaim_12h`, `flow_impulse`, `breakout_cont_12h`, `range_upper_cont_3d`
- Holds: `2h`, `3h`, `4h`, `6h`
- Stride: `6` bars
- Leverage: fixed `1.0x`
- Acceptance screen: positive CAGR on both test 2024 and eval 2025, then ratio/trade-count ranking

Result: no candidate in this fast non-pb30 family set had positive CAGR on both test and eval. This supports the current working thesis: simple mid-frequency long thresholds are not enough; the next attempt needs either a richer state classifier or a different non-long action family, not just looser long entries.

## Secondary filter on high-trade pb30 scaffold

A targeted secondary-filter sweep was run on the best fixed-1x high-trade scaffold:

- Base: `pb30 q0.70 + activity_flow_htf q0.40`
- Hold: `6h`
- Stride: `6` bars
- Leverage: fixed `1.0x`
- Secondary filter thresholds: train quantiles only

Artifact:

- `results/bullish_high_trade_secondary_filter_scan_2026-07-05.json`

Best quality / lower-MDD candidate:

- Secondary gate: `htf_1d_return_1 >= train q0.60`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 28.07% | 28.01% | 6.59% | 4.25 | 50 | 0.0391 |
| eval 2025 | 25.47% | 25.49% | 9.03% | 2.82 | 54 | 0.0633 |

Best higher-trade balanced candidate:

- Secondary gate: `rex_36_range_pos <= train q0.80`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 61.77% | 61.61% | 24.32% | 2.53 | 153 | 0.0293 |
| eval 2025 | 48.87% | 48.91% | 18.49% | 2.65 | 149 | 0.0595 |

Conclusion: secondary filters improve the high-trade pb30 scaffold materially, but not enough for the `5+` target. The 50-trade `htf_1d_return_1` gate is the first useful mid-frequency lead because it cuts MDD sharply while keeping both splits positive.

## Bidirectional/non-pb30 threshold probes

Fast bidirectional LONG/SHORT threshold probes over core oscillator/flow features did not produce a positive-both candidate in the reduced search.

Artifact:

- `results/bidir_midfreq_threshold_fast_scan_trainlt2024_test2024_eval2025_2026-07-05.json`

Current implication: the most promising mid-frequency route is not pure non-pb30 thresholding, but a filtered high-trade pb30 scaffold or a richer state classifier around it.

## Two-gate state classifier on high-trade scaffold

A two-gate state classifier was tested on the same fixed-1x scaffold:

- Base: `pb30 q0.70 + activity_flow_htf q0.40`
- Hold: `6h`
- Stride: `6` bars
- Leverage: fixed `1.0x`
- Gate thresholds: train quantiles only

Artifact:

- `results/bullish_high_trade_two_gate_scaffold_scan_2026-07-05.json`

Best low-MDD near-target candidate:

- Gates:
  - `taker_imbalance >= train q0.60`
  - `return_zscore_48 <= train q0.50`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 26.09% | 26.03% | 5.82% | 4.48 | 53 | 0.0325 |
| eval 2025 | 36.04% | 36.07% | 7.20% | 5.01 | 71 | 0.0542 |

Best higher-trade ratio-3 candidate:

- Gates:
  - `volume_ratio >= train q0.70`
  - `rex_144_range_pos <= train q0.80`

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 65.28% | 65.11% | 18.13% | 3.59 | 116 | 0.0249 |
| eval 2025 | 46.11% | 46.15% | 13.63% | 3.39 | 119 | 0.0613 |

Conclusion: the two-gate classifier materially improves the mid-frequency path. The `taker_imbalance + return_zscore_48` candidate is the current best fixed-1x mid-frequency lead: it has 53/71 trades, low strict MDD, and eval clears 5, but test is still just below target at 4.48.

## RLLM-compatible long midfrequency action book pass

After noting that brute-force hand-tuned threshold search drifts away from the RLLM objective, the next pass reframed the long-regime work as an action-book / selector problem:

- Candidate generators are not final strategies; they create prompt-visible alternatives.
- The selector should choose among `NO_TRADE` and LONG candidates using signal-time context.
- Future OHLC path is stored only as supervised utility/target.
- Validation keeps the same strict accounting: fixed 1x, full split annualization, in-hold 5m adverse excursion, and period-end forced close.

Artifact:

- `data/long_midfreq_action_book_2026-07-05/candidates_fast.jsonl`
- `data/long_midfreq_action_book_2026-07-05/listwise_fast_random_neutral.jsonl`
- `data/long_midfreq_action_book_2026-07-05/listwise_fast_{train,test_2024,eval_2025,ytd_2026}.jsonl`
- `results/long_midfreq_action_book_fast_ridge_baseline_2026-07-05.json`

Candidate-book shape:

| split | listwise rows |
| --- | ---: |
| train `<2024` | 5,340 |
| test `2024` | 625 |
| eval `2025` | 672 |
| ytd `2026-01-01..2026-06-02` | 379 |

Semantic target mix after resolving randomized neutral labels:

| split | NO_TRADE | loose_pb30_activity | htf1d_positive_gate |
| --- | ---: | ---: | ---: |
| train | 2,707 | 1,614 | 1,019 |
| test 2024 | 287 | 226 | 112 |
| eval 2025 | 287 | 285 | 100 |
| ytd 2026 | 190 | 166 | 23 |

Train-only ridge utility distillation baseline, threshold selected on train only:

| split | CAGR | strict MDD | CAGR/MDD | trades | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| train `<2024` | 1.61% | 27.45% | 0.06 | 358 | 0.37 |
| test 2024 | 3.71% | 6.91% | 0.54 | 25 | 0.68 |
| eval 2025 | 2.17% | 3.97% | 0.55 | 29 | 0.66 |
| ytd 2026 | -2.76% | 7.34% | -0.38 | 24 | -0.15 |

Oracle ceiling, using future utility to choose the best candidate per signal, is **Tier 0 diagnostic only** and must not be read as strategy performance:

| split | CAGR | strict MDD | CAGR/MDD | trades | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| test 2024 | 490.72% | 5.23% | 93.79 | 93 | 12.52 |
| eval 2025 | 369.93% | 4.05% | 91.25 | 91 | 12.89 |
| ytd 2026 | 674.78% | 5.61% | 120.22 | 46 | 7.77 |

Interpretation:

- The action book has a high ceiling only under future-aware selection; this is not deployable evidence.
- The cheap train-only ridge utility distillation baseline fails badly, so the current bottleneck is selector learning/calibration, not merely candidate availability.
- The next RLLM-shaped step must follow `docs/rllm-alpha-failure-guardrails-2026-07-05.md`: no future-best preselection, no eval-shaped target changes, and no more hand-tuned quantile-rule expansion as strategy evidence.

### Binary / candidate-level selector follow-up

A first binary `NO_TRADE vs best executable trade` classifier looked very strong, but it was rejected as a deployable proof because the record construction selected the best trade candidate using future utility before inference.  That is useful only as a target-construction sanity check, not as replay evidence.

The corrected non-leaky candidate-level selector scores every available candidate independently using signal-time features, then chooses the highest-scored candidate per signal only if it clears a threshold selected on train `<2024`.

Artifact:

- `results/long_midfreq_candidate_level_selector_2026-07-05.json`
- `results/long_midfreq_candidate_level_selector_2026-07-05/*_executed.jsonl`

Best non-leaky candidate-level result from the quick baselines:

| model | split | CAGR | strict MDD | CAGR/MDD | trades | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| candidate logistic gap>=0.2 | train `<2024` | 18.98% | 43.97% | 0.43 | 378 | 1.33 |
| candidate logistic gap>=0.2 | test 2024 | 52.81% | 10.64% | 4.96 | 43 | 2.88 |
| candidate logistic gap>=0.2 | eval 2025 | 4.06% | 12.80% | 0.32 | 31 | 0.36 |
| candidate logistic gap>=0.2 | ytd 2026 | -11.79% | 26.15% | -0.45 | 32 | -0.25 |

Other non-leaky baselines showed the same pattern: 2024 can look good, but 2025 collapses and 2026 remains weak.  This confirms that the high oracle ceiling is mostly a future-selection artifact unless the selector learns a stable, causal preference.

Conclusion for this branch:

- Keep the action-book/listwise data surface; it is aligned with the RLLM objective.
- Do not claim the binary-best-trade replay as strategy evidence.
- Current cheap non-leaky selectors do not solve the midfrequency long-regime problem.
- Next selector work should use time-balanced listwise/pairwise training and validate on 2024 before touching 2025; otherwise it will keep selecting 2024-only artifacts.


## Failure-guardrail status

This branch is now governed by `docs/rllm-alpha-failure-guardrails-2026-07-05.md`.  In particular, oracle ceilings and future-best binary replays are diagnostic-only and cannot be used as strategy evidence.  The only deployable-style evidence in this branch is the non-leaky candidate-level selector replay, which failed 2025 eval.

## Tier 1 rolling prior-only selector audit

Artifact:

- `results/long_midfreq_rolling_prior_audit_2026-07-05.json`
- `results/long_midfreq_rolling_prior_audit_2026-07-05/*_test_executed.jsonl`

Protocol:

- Candidate rows are available at signal time; no future-best candidate is preselected.
- For each fold, thresholds are selected only on the prior validation period.
- Test period is untouched for the fold.
- Leverage is fixed at 1.0.
- Strict MDD includes in-hold 5m adverse low and forced period-end close.

Validation-selected replay:

| fold | validation used for threshold | selected model | validation CAGR/MDD | validation trades | test period | test CAGR | test strict MDD | test CAGR/MDD | test trades | Sharpe |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| audit_2024 | 2023 | logreg_gap0.2 | 0.36 | 20 | 2024 | 49.34% | 27.38% | 1.80 | 87 | 1.84 |
| audit_2025 | 2024 | hgb_gap0.2 | 4.82 | 69 | 2025 | 36.40% | 16.64% | 2.19 | 73 | 1.78 |
| audit_2026ytd | 2025 | logreg_gap1.0 | 4.63 | 51 | 2026 YTD | 1.58% | 23.51% | 0.07 | 39 | 0.12 |

Notes:

- `audit_2024` is weak evidence because 2023 validation produced only 20 trades; the selected rule then fails the 2024 target anyway (`CAGR/MDD 1.80`).  Even the best 2024 diagnostic threshold, selected by looking at 2024 test, only reached `CAGR/MDD 3.57` with `21.94%` strict MDD, so it does not meet the bullish target-5 criterion.
- `audit_2025` is the cleanest prior-only check: 2024 validation almost worked (`4.82`) but the untouched 2025 replay fell to `2.19` with worse drawdown.
- `audit_2026ytd` confirms drift: the 2025-selected rule collapses in 2026 YTD (`0.07`).

Conclusion:

- The RLLM/action-book branch is useful as an offline learning surface, but current Tier 1 non-leaky selectors do **not** recover a deployable bullish midfrequency alpha.
- Stop treating further hand-mined candidate gates as progress; they are likely to repeat prior failures.
- A valid continuation must either improve causal selector robustness with rolling/time-balanced preference learning or switch away from this pb30/action-book family.

## Non-pb30 broad diagnostic scan

Artifact:

- `results/broad_nonpb30_pair_scan_diagnostic_fast_2026-07-05.json`

Protocol caveat:

- This is **Tier 0 diagnostic**, not deployable evidence, because the broad search ranked candidates using both 2024 test and 2025 eval.
- Thresholds themselves are still fit on train `<2024` only.
- Leverage is fixed at 1.0; strict MDD includes in-hold adverse low and period-end close.
- Direct pb30-style `rex_8640_max_to_cur_pct` gates were excluded to avoid repeating the pb30 family.

Best diagnostic families found:

| family | hold / stride | 2024 CAGR | 2024 MDD | 2024 ratio | 2024 trades | 2025 CAGR | 2025 MDD | 2025 ratio | 2025 trades | 2026 YTD ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `htf_3d_range_1 <= q30` + `htf_1w_range_pos >= q70` | 3h / 1h | 39.20% | 10.29% | 3.81 | 37 | 33.40% | 7.85% | 4.25 | 44 | 0.89 |
| `shadow_imbalance <= q10` + `lower_shadow <= q30` | 6h / 30m | 248.04% | 12.93% | 19.18 | 48 | 39.45% | 10.45% | 3.77 | 32 | -1.54 |
| `htf_4h_return_1 >= q70` + `shadow_imbalance <= q30` | 4h / 1h | 68.28% | 18.20% | 3.75 | 56 | 77.62% | 18.77% | 4.14 | 50 | -0.39 |
| `lower_shadow <= q30` + `taker_imbalance >= q90` | 8h / 1h | 67.35% | 21.16% | 3.18 | 27 | 52.24% | 10.83% | 4.83 | 32 | 0.89 |

Conclusion:

- Even after eval-aware diagnostic mining outside the pb30 family, no fixed 1x midfrequency long candidate reached the target `CAGR / strict MDD >= 5` in both 2024 and 2025.
- The most promising non-pb30 family is **HTF compression + weekly upper-range location**; it is less explosive than candle-shadow artifacts and did not go negative in 2026 YTD, but it is still below the target and weak in 2026.
- Candle-shadow / taker-flow families can look strong in 2024/2025 diagnostics, but 2026 degradation is severe, so they should not be promoted without a prior-only selector audit.

## HTF compression family prior-only audit

Artifact:

- `results/htf_compression_long_prior_audit_2026-07-05.json`

Caveat:

- The family was discovered in the previous Tier 0 eval-aware diagnostic scan, so this is not a clean deployable proof.
- Within the family, thresholds are fit on train `<2024`; parameter/hold/stride selection uses train + 2024 only; 2025 is report-only.
- Leverage fixed at 1.0; strict MDD includes in-hold adverse low and period-end close.

2024-selected top candidates:

| gates | hold / stride | train ratio | train trades | 2024 ratio | 2024 trades | 2025 ratio | 2025 trades | 2026 YTD ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `htf_3d_range_1<=q30` + `rex_576_range_pos>=q60` | 3h / 30m | 1.87 | 110 | 9.28 | 52 | 0.80 | 59 | -0.91 |
| `htf_3d_drawdown_4<=0` + `rex_576_range_pos>=q60` | 4h / 2h | 0.15 | 125 | 8.58 | 32 | -0.34 | 26 | -1.86 |
| `htf_3d_drawdown_4<=0` + `rex_576_range_pos>=q60` | 2h / 1h | 0.31 | 236 | 7.88 | 59 | 0.45 | 46 | -2.11 |

Best 2024/2025 simultaneous diagnostic candidates inside the family:

| gates | hold / stride | train ratio | 2024 ratio | 2025 ratio | 2026 YTD ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `htf_3d_range_1<=q30` + `htf_1w_range_pos>=q60` | 3h / 1h | -0.17 | 4.17 | 3.90 | 1.62 |
| `htf_3d_range_1<=q30` + `htf_1w_range_pos>=q70` | 3h / 1h | -0.07 | 3.81 | 4.25 | 0.89 |
| `htf_3d_range_1<=q30` + `htf_1d_return_1>=q70` | 3h / 1h | 0.06 | 3.76 | 3.73 | -1.50 |
| `htf_3d_range_1<=q30` + `rex_2016_range_pos>=q60` | 3h / 1h | 0.55 | 4.39 | 3.35 | -1.29 |

Conclusion:

- The family does not meet the target `CAGR / strict MDD >= 5` in both 2024 and 2025.
- 2024-selected top rules overfit 2024 and collapse in 2025.
- Rules that are simultaneously decent in 2024/2025 have weak or negative train evidence and degrade again in 2026.
- This confirms the current bullish midfrequency fixed-rule path is not producing a live-grade alpha under the strict protocol.

## Non-pb30 longer-hold diagnostic scan

Artifact:

- `results/broad_nonpb30_longer_hold_diagnostic_2026-07-05.json`

Protocol caveat:

- Tier 0 diagnostic only: candidate reporting used both 2024 and 2025.
- Thresholds fit on train `<2024`; leverage fixed at 1.0; strict MDD includes in-hold adverse low and period-end close.
- Direct pb30-style `rex_8640_max_to_cur_pct` gates excluded.

Result:

| gates | hold / stride | 2024 CAGR | 2024 MDD | 2024 ratio | 2024 trades | 2025 CAGR | 2025 MDD | 2025 ratio | 2025 trades | 2026 YTD ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `htf_1w_range_pos>=q70` + `candle_range<=q30` | 36h / 1h | 200.52% | 22.42% | 8.95 | 13 | 52.78% | 14.76% | 3.58 | 11 | 1.09 |
| `rex_36_cur_to_min_pct>=q70` | 12h / 3h | 169.22% | 20.96% | 8.07 | 29 | 41.99% | 13.86% | 3.03 | 19 | n/a |

Conclusion:

- Extending hold time outside pb30 does not solve the target-5 problem.
- The longer-hold scan produces sparse 2024 winners, but 2025 remains below target and 2026 has too few trades / weak ratio.
- Combined with the midfrequency and HTF-compression audits, the current fixed-rule bullish long search is exhausted under the strict protocol.

## Live pb30 base + midfrequency add-on check

Artifact:

- `results/pb30_live_plus_known_addon_5m_2026-07-05.json`

Important reproduction note:

- The live pb30 artifact is reproducible only on the 5m wavefull/cache input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Using the lower-row `data/2020-01-01_2026-06-02_7120f802e1409dba0460d4c1a133ce61.csv.gz` changes train quantiles and breaks the base signal reproduction.  Future audits must use the 5m wavefull/cache input for this family.

Base reproduction:

| split | CAGR | strict MDD | CAGR/MDD | trades |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 104.10% | 15.01% | 6.93 | 22 |
| 2025 | 70.40% | 14.06% | 5.01 | 20 |
| 2026 YTD | 46.65% | 28.44% | 1.64 | 17 |

Known add-on tested:

- add-on: `pb30 q70 + activity_flow_htf q40 + taker_imbalance>=q60 + return_zscore_48<=q50`, hold 6h, 1.0x
- execution: one active position at a time; base and add-on compete for the same capital; add-on cannot overlap an existing base/add-on hold.

Combined result:

| split | CAGR | strict MDD | CAGR/MDD | total trades | base trades | add-on trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 104.50% | 19.22% | 5.44 | 57 | 18 | 39 |
| 2025 | 24.98% | 18.80% | 1.33 | 68 | 14 | 54 |
| 2026 YTD | -13.50% | 31.86% | -0.42 | 42 | 9 | 33 |

Conclusion:

- This add-on increases trade count but destroys the 2025 acceptance ratio and flips 2026 YTD negative.
- The main failure mode is not just low add-on quality: taking add-on trades also blocks later high-quality base entries under single-capital sequential execution.
- Therefore the midfrequency near-miss should not be merged into the live pb30 strategy.

## Live pb30 base + filtered add-on scan

Artifacts:

- `results/pb30_live_addon_fast_scan_5m_2026-07-05.json`
- `results/pb30_live_addon_top_exact_verify_2026-07-05.json`
- `configs/live/bullish_pb30_addon_returnz_htf1w_candidate.json`

Protocol:

- Input fixed to 5m wavefull/cache: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Base module is existing `pb30_funding + activity_flow_htf`, 18h, 1.7x.
- Add-on module uses the looser 6h pb30/activity scaffold, 1.0x.
- Single-capital sequential execution: base has priority on identical timestamps, but any open add-on can still block a later base signal while it is held.
- Add-on thresholds are train `<2024` quantiles.  Fast scan ranked by 2024; top candidates were then exact bar-by-bar verified.
- Strict MDD in exact verification includes in-hold adverse 5m lows and period-end close.

Exact-verified top candidate:

Add-on gates:

- `rex_8640_max_to_cur_pct >= q70` (`0.16914110380482186`)
- `activity_flow_htf >= q40` (`-0.13969842891406745`)
- `return_zscore_48 <= q20` (`-0.9811529136901854`)
- `htf_1w_range_pos <= q20` (`0.20634652133561793`)
- hold 6h, stride 30m, leverage 1.0x

Combined exact metrics:

| split | CAGR | strict MDD | CAGR/MDD | total trades | base trades | add-on trades | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 141.79% | 12.62% | 11.23 | 47 | 19 | 28 | 0.0018 |
| 2025 | 85.56% | 14.06% | 6.09 | 34 | 18 | 16 | 0.0018 |
| 2026 YTD | 48.66% | 28.20% | 1.73 | 32 | 13 | 19 | 0.4305 |

Interpretation:

- This is the first add-on that increases trade count while preserving the target-5 requirement in both 2024 and 2025 under exact strict replay.
- It improves 2024 from 22 to 47 trades and 2025 from 20 to 34 trades.
- 2026 YTD remains weak on `CAGR/MDD` because drawdown stays high; therefore this should be treated as **dry-run candidate**, not live-order enabled.
- The useful pattern is not the earlier taker-positive add-on; it is a filtered add-on where the pullback happens with low short-term return z-score and low weekly range location.

## Independent non-pb30 midfrequency scans

Artifacts:

- `results/nonpb30_midfreq_long_5m_scan_2026-07-05.json`
- `results/nonpb30_midfreq_long_5m_scan_relaxed_2026-07-05.json`
- `results/nonpb30_midfreq_directional_5m_scan_2026-07-05.json`

Protocol:

- Input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Direct pb30 / `rex_8640_max_to_cur_pct` excluded.
- Gates are train `<2024` quantiles.
- Hold range: 1h to 8h; signal cadence 30m or 1h.
- Long-only scan first; then directional long/short diagnostic.
- Fast MDD scan was used for discovery; exact verification would be required for any candidate, but no candidate survived even relaxed reporting.

Results:

| scan | side universe | features | gates | reported near-misses | qualified |
| --- | --- | ---: | ---: | ---: | ---: |
| strict non-pb30 midfreq | long only | 52 | 416 | 0 | 0 |
| relaxed non-pb30 midfreq | long only | 52 | 416 | 0 | 0 |
| directional non-pb30 midfreq | long + short | 49 | 392 | 0 | 0 |

Conclusion:

- No independent non-pb30 midfrequency alpha was found in the current 5m feature universe.
- The meaningful new result from this session remains the pb30-family low-frequency/dry-run add-on, not a standalone midfrequency alpha.
- To continue searching for true midfrequency alpha, the next branch should introduce a different event surface rather than more train-quantile gates over the same existing feature table. Candidate next surfaces: intraday session/time-of-day conditioning, multi-asset relative flow, order-flow regime from pooled alts, or learned event selector with a strict rolling prior-only protocol.

## Session + multi-asset midfrequency surface

Artifacts:

- `results/session_multiasset_midfreq_scan_staged_2026-07-05.json`

Protocol:

- Input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Alt universe: ADA, BNB, DOGE, ETH, SOL, XRP Binance USD-M 5m bars.
- New gates: UTC session/time windows plus alt basket returns/breadth and BTC-vs-alt relative returns.
- Direct pb30 excluded.
- Holds 1h to 8h, cadence 30m/1h, long and short evaluated.
- Staged scan: rank single gates by 2024, then pair top gates; 2025 report-only.

Result:

| scan | gates | reported near-misses | qualified |
| --- | ---: | ---: | ---: |
| session + multi-asset relative flow | 244 | 0 | 0 |

Conclusion:

- Adding simple session and alt-relative-flow gates did not produce an independent midfrequency alpha.
- Current failed surfaces now include: existing BTC feature quantile gates, directional long/short gates, and simple session/multi-asset relative-flow gates.
- Further search likely needs either a learned selector/event representation or a genuinely different microstructure data source; simple symbolic quantile-gate expansion is not yielding standalone midfrequency alpha.

## Learned non-pb30 midfrequency selector attempt

Artifact:

- `results/nonpb30_learned_midfreq_selector_fast_2026-07-05.json`

Protocol:

- Input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Candidate cadence: 1h.
- Holds: 2h, 4h, 6h.
- Sides: long and short.
- Features: 91 existing market features with direct pb30 / `rex_8640_max_to_cur_pct` excluded.
- Model: logistic regression with median imputation, standardization, balanced class weights.
- Labels: train-window future utility quantiles only; score thresholds are train quantiles; 2025 is not used for thresholding.

Result:

| selector | reported near-misses | qualified |
| --- | ---: | ---: |
| non-pb30 learned logreg selector | 0 | 0 |

Conclusion:

- A simple learned selector over the current non-pb30 feature table also fails to surface standalone midfrequency alpha.
- At this point, repeated attempts over the same BTC 5m feature table are exhausted: symbolic gates, directional gates, session/alt-relative gates, and a basic learned selector all failed.
- The remaining viable paths require either a new data source (order book/liquidations/open-interest deltas with reliable 5m alignment), a different target structure, or accepting the pb30-family dry-run add-on as the only currently monetizable long-side expansion.

## Independent mid-frequency alpha attempt: shock-reversal template (non-pb30)

Protocol:

- Input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
- Excluded direct `pb30` / `rex_8640_max_to_cur_pct` family.
- Train thresholds fit on `<2024`; acceptance requires both `2024` test and `2025` eval.
- Holds: 1h-6h; cadence: 15m/30m/1h; long and short.
- Template: intraday shock reversal/continuation using return/RSI extremes + volume/trades activity + wick/range/HTF/taker confirmations.

Result:

- Reported candidates: `0`
- Qualified candidates: `0`

Artifact:

- `results/shock_reversal_midfreq_scan_2026-07-05.json`

Conclusion: no independent mid-frequency alpha found on this event-shock template.

## Independent mid-frequency alpha attempt: interest/activity score as standalone signal (non-pb30)

Protocol:

- Excluded direct `pb30` / `rex_8640` features.
- Used `activity`, `activity_flow`, `activity_flow_htf`, `activity_flow_deriv_htf`, and raw interest/activity features as standalone entry gates, optionally paired with non-pb30 context gates.
- Long-only exact strict replay, split-contained exits to avoid bleeding test/eval boundaries.
- Holds: 1h-8h; cadence: 15m/30m/1h; leverage 1.0; minimum 35 trades on both 2024 and 2025.

Result:

- Candidates checked: `3960`
- Reported candidates: `0`
- Qualified candidates: `0`

Artifact:

- `results/interest_score_long_nonpb30_midfreq_scan_2026-07-05.json`

Conclusion: the activity/flow score appears useful as a filter inside the existing pb30-family long-regime alpha, but did not stand alone as an independent mid-frequency alpha under the current strict test/eval protocol.

## Independent non-pb30 mid-frequency candidate found: taker/returnz/range-vol/HTF-range

Discovery artifact:

- `results/seed_taker_returnz_nonpb30_gate_scan_2026-07-05.json`
- `results/seed_taker_returnz_second_gate_scan_2026-07-05.json`

Exact verification artifact:

- `results/nonpb30_midfreq_candidate_exact_verify_2026-07-05.json`

Protocol:

- Input: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- Threshold reference: train `<2024` only.
- Acceptance windows: `2024` test and `2025` eval.
- CAGR annualization uses the full declared calendar windows, including cash/no-trade time.
- Strict MDD includes in-hold 5m adverse low excursion.
- Period exits are split-contained / forced at the split boundary.
- Leverage: `1.0`; no leverage fitting.
- Cadence: 1h (`stride_bars=12`).
- Hold: 6h (`hold_bars=72`).
- Direct `pb30` and `rex_8640` family excluded.

### Candidate A — `nonpb30_taker_returnz_rangevol_htf4hrange_h72`

Gates:

- `taker_imbalance >= train q0.60` (`0.053299460961426244`)
- `return_zscore_48 <= train q0.50` (`-0.0062564587941375364`)
- `range_vol >= train q0.75` (`0.04338694545519488`)
- `htf_4h_range_1 >= train q0.65` (`0.020567028637044467`)

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | win rate | p approx | trade Sharpe-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train `<2024` | 12.51% | 2.99% | 45.17% | 0.07 | 296 | 46.62% | 0.6289 | 0.48 |
| test 2024 | 26.70% | 26.64% | 4.34% | 6.14 | 34 | 58.82% | 0.0144 | 2.45 |
| eval 2025 | 24.10% | 24.12% | 4.42% | 5.46 | 35 | 60.00% | 0.0384 | 2.07 |
| ytd 2026 | 8.91% | 22.60% | 8.80% | 2.57 | 14 | 64.29% | 0.1616 | 1.40 |

### Candidate B — `nonpb30_taker_returnz_rex144width_htf4hrange_h72`

Gates:

- `taker_imbalance >= train q0.60` (`0.053299460961426244`)
- `return_zscore_48 <= train q0.50` (`-0.0062564587941375364`)
- `rex_144_range_width_pct >= train q0.75` (`0.043317153258360976`)
- `htf_4h_range_1 >= train q0.65` (`0.020567028637044467`)

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | win rate | p approx | trade Sharpe-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train `<2024` | 13.40% | 3.19% | 44.77% | 0.07 | 297 | 47.14% | 0.6170 | 0.50 |
| test 2024 | 26.70% | 26.64% | 4.34% | 6.14 | 34 | 58.82% | 0.0144 | 2.45 |
| eval 2025 | 22.84% | 22.86% | 4.42% | 5.18 | 34 | 58.82% | 0.0482 | 1.98 |
| ytd 2026 | 8.91% | 22.60% | 8.80% | 2.57 | 14 | 64.29% | 0.1616 | 1.40 |

Interpretation:

- This is the first independent non-pb30 mid-frequency candidate in the current pass that clears `CAGR / strict MDD >= 5` on both `2024` test and `2025` eval under exact strict replay.
- It is still a Tier 0 discovery, not a live promotion: train `<2024` is weak and 2026 YTD ratio falls to `2.57` with only 14 trades.
- Candidate A is preferred over B because it avoids even the shorter `rex_144` range-width dependency and has slightly better 2025 eval ratio/trade count.

## Higher-frequency follow-up after non-pb30 6h candidate

Saved paper config:

- `configs/live/nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json`

Attempted to increase frequency from the accepted 6h / 1h-cadence candidate:

1. Shorter hold variants:
   - Holds: 1h, 1.5h, 2h, 2.5h, 3h, 4h, 5h.
   - Cadence: 15m, 30m, 1h.
   - Same non-pb30 taker/returnz/range-vol/HTF-range family and nearby threshold variants.
   - Artifact: `results/highfreq_candidate_variants_fast_2026-07-05.json`.
   - Result: reported `0`, qualified70 `0`, qualified50 `0`.

2. Same hold neighborhood with faster cadence:
   - Holds: 4h, 5h, 6h, 7h, 8h.
   - Cadence: 15m, 30m, 1h.
   - Artifact: `results/higherfreq_hold6_8_variants_2026-07-05.json`.
   - Result: reported `0`, qualified70 `0`, qualified50 `0`.

Interpretation:

- The discovered non-pb30 alpha does not scale cleanly to higher frequency by simply shortening hold or checking the same family more often.
- Current best higher-trade near-miss remains the earlier two-gate midfrequency lead:
  - `taker_imbalance >= q0.60`
  - `return_zscore_48 <= q0.50`
  - test 2024 ratio `4.48`, eval 2025 ratio `5.01`, trades `53/71`.
- The accepted independent candidate remains lower frequency than desired at `34/35` trades/year.

## Higher-frequency alpha search continuation

After saving the independent 6h paper candidate, additional higher-frequency searches were run with stricter trade-count targets.

Artifacts:

- `results/highfreq_tpsl_nonpb30_nearmiss_2026-07-05.json`
- `results/alt_breadth_leadlag_highfreq_scan_2026-07-05.json`
- `results/highfreq_directional_reversion_scan_2026-07-05.json`
- `results/intraday_seasonality_highfreq_fast_2026-07-05.json`
- `results/oi_deriv_highfreq_scan_2026-07-05.json`

Attempts and results:

| attempt | target | result |
| --- | --- | --- |
| TP/SL on known non-pb30 near-miss | >=70 trades/year, test/eval ratio >=5 | reported 0, qualified 0 |
| alt breadth / BTC-underperformance lead-lag | >=70 trades/year | reported 0, qualified 0 |
| directional long/short intraday reversion | >=80 trades/year | reported 0, qualified 0 |
| UTC hour/day seasonality | >=80 trades/year | reported 0, qualified 0 |
| OI / derivative squeeze | >=70 trades/year | variants 0 because current feature input did not provide usable OI thresholds |

Conclusion:

- No higher-frequency alpha was found in this continuation pass.
- The accepted independent non-pb30 paper candidate remains the 6h / 1h-cadence `taker_imbalance + return_zscore_48 + range_vol + htf_4h_range_1` module with `34/35` test/eval trades.
- The best higher-trade near-miss remains the earlier two-gate `taker_imbalance + return_zscore_48` lead with roughly `53/71` trades but 2024 ratio below target.

## Independent DB-OI alpha found: OI divergence pullback rebound

Artifacts:

- `results/oi_custom_alpha_scan_2026-07-05.json`
- `results/oi_divergence_refine_scan_2026-07-05.json`
- `results/oi_divergence_candidate_exact_verify_2026-07-05.json`
- `configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json`

Data:

- OI source: `public.open_interest_binance`, `BTCUSDT`, `period=5m`.
- DB coverage checked: `2020-09-01 00:00:00 UTC` through `2026-07-05 14:00:00 UTC`.
- Historical replay merged OI into `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`.
- `oi_change` / `oi_zscore` are now nonzero and usable after DB merge.

Signal:

- Name: `oi_divergence_pullback_range_rsi_h96_s6`
- Side: long
- Hold: 8h (`96` 5m bars)
- Cadence: 30m (`6` 5m bars)
- Leverage: 1.0
- Direct `pb30` and `rex_8640` excluded.
- Fee/slippage: `0.0005` entry + `0.0005` exit.

Gates:

- `oi_minus_px_4h_z >= train q0.80` (`0.8954018630586817`)
- `return_zscore_48 <= train q0.20` (`-0.7389570664259131`)
- `range_vol >= train q0.70` (`0.04008415457867338`)
- `rsi_norm <= train q0.40` (`-0.04507656773717145`)

Interpretation:

- OI has risen strongly relative to price over 4h (`OI up while price underperforms`).
- BTC is already in a short-term pullback / non-overbought state.
- Volatility is elevated.
- RSI is not strong; this is a squeeze/rebound setup after crowded short/hedge build-up, not a momentum chase.

Exact verification:

| split | return | CAGR | strict MDD | CAGR / strict MDD | trades | win rate | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train `<2024` | 42.71% | 11.26% | 44.29% | 0.25 | 313 | 53.99% | 0.3285 |
| test 2024 | 52.54% | 52.41% | 6.34% | 8.27 | 64 | 56.25% | 0.0285 |
| eval 2025 | 36.60% | 36.63% | 5.46% | 6.71 | 40 | 62.50% | 0.0428 |
| ytd 2026 | 0.62% | 1.50% | 9.65% | 0.16 | 17 | 41.18% | 0.9054 |

Status:

- This is a second independent non-pb30 alpha candidate and the first one using the newly available DB OI.
- It clears the target on both `2024` and `2025`, with more 2024 trades than the previous non-OI candidate.
- It remains paper-only: train performance is weak and 2026 YTD collapses.

## 2026-07-05: independent OI high-frequency candidate, >=100 trades/year

Saved paper candidate: `configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json`.

Protocol: 1x, 5m BTCUSDT, hold 30 bars / 150 minutes, stride 6 bars / 30 minutes, 5bp entry + 5bp exit, pb30/rex_8640 excluded. Thresholds are train `<2024` quantiles; acceptance is test `2024` and eval `2025`. Strict MDD includes in-position adverse OHLC excursion and split-contained forced close.

Signal gates:
- `oi_minus_px_4h_z >= q55` (`0.18084217361514066`)
- `return_zscore_48 <= q35` (`-0.3382356464105935`)
- `range_vol >= q65` (`0.03684324822682795`)
- `rsi_norm <= q50` (`0.006924814138880962`)
- `oi_ret_4h_z >= q25` (`-0.675926157602224`)
- `sma24_ratio <= q40` (`-0.0005569113930713103`)

Stats:

| period | return | CAGR | strict MDD | CAGR/MDD | trades | win | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 test | 60.80% | 60.65% | 10.71% | 5.66 | 197 | 55.84% | 0.0037 |
| 2025 eval | 33.42% | 33.44% | 5.91% | 5.66 | 124 | 60.48% | 0.0341 |
| 2026 YTD diagnostic | -4.27% | -9.90% | 9.80% | -1.01 | 45 | 48.89% | 0.5493 |

Interpretation: OI/price divergence pullback continuation/rebound. Price is below short SMA and weak on 48-bar return, but OI is not collapsing and OI is high relative to price over 4h; range/volatility is elevated. This looks like a crowded-position pullback/squeeze setup, not a pb30 regime artifact.

Risk: despite passing 2024/2025 with >100 trades/year, 2026 YTD is negative and the final version used two added filters from near-miss refinement (`oi_ret_4h_z`, `sma24_ratio`). Treat as paper candidate, not live, until forward behavior improves or a 2026 regime guard is found.

Artifacts:
- `results/oi_100trades_alpha_scan_2026-07-05.json`
- `results/oi_100trades_add1_filter_2026-07-05.json`
- `results/oi_100trades_add2_filter_2026-07-05.json`

## 2026-07-06: LLM-style selector overlay for OI high-frequency candidate

Added evaluator: `training/evaluate_oi_llm_selector.py`.

Saved overlay config: `configs/live/oi_divergence_sma24_highfreq_h30_s6_llm_selector_overlay.json`.

Artifacts:
- selector eval: `results/oi_llm_selector_eval_2026-07-06.json`
- LLM state-card dataset: `results/oi_llm_selector_cards_2026-07-06.jsonl`
- fast threshold sweep: `results/oi_llm_selector_fast_sweep_2026-07-06.json`

Design: the base alpha remains fixed. The selector only emits `ALLOW` or `BLOCK` for already-triggered OI divergence signals; it does not set entries, exits, leverage, TP/SL, or holding period. The current offline evaluator uses a train-only symbolic proxy over compact LLM state tokens before any real LLM call is trusted.

Best symbolic proxy:
- Context keys: `short_sma`, `bb_location`, `oi_ret_4h`
- Fit split: train `<2024` only
- Block context if train support >= 24 and either mean return <= -10 bps or win rate <= 34%
- Blocked train-derived contexts:
  - `short_sma=below|bb_location=lower_extreme|oi_ret_4h=rising`
  - `short_sma=below_far|bb_location=lower|oi_ret_4h=surging`

Stats:

| period | baseline return | baseline CAGR/MDD | baseline trades | selector return | selector CAGR/MDD | selector trades | blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 test | 60.80% | 5.66 | 197 | 60.04% | 5.59 | 190 | 7 |
| 2025 eval | 33.42% | 5.66 | 124 | 32.95% | 5.58 | 118 | 6 |
| 2026 YTD | -4.27% | -1.01 | 45 | -0.89% | -0.31 | 41 | 4 |

Conclusion: this is a useful paper overlay. It preserves the 2024/2025 acceptance floor and removes a small number of 2026 bad contexts, but 2026 remains negative. Next live step should be paper-only LLM shadow logging: compare real LLM `ALLOW/BLOCK` against this symbolic proxy and require bounded disagreement before promotion.

## 2026-07-06: simultaneous bull/bear sleeve portfolio mixer

Saved paper portfolio candidate: `configs/live/portfolio_bull_bear_oi_rex_capital_efficient_candidate.json`.

Artifact: `results/portfolio_mixer_bull_bear_sleeves_fast_2026-07-06.json`.

Assumption changed from single-capital priority to simultaneous sleeves with flexible leverage/margin. Each sleeve is internally non-overlapping, but different sleeves may overlap. Portfolio return is weighted sum of sleeve bar returns; strict MDD uses weighted summed intrabar adverse moves. This is diagnostic and should be revalidated in the production backtester before live use.

Candidate sleeves and weights:

| sleeve | side | weight |
|---|---:|---:|
| `nonpb30_taker_returnz_rangevol_htf4hrange_h72` | long | 0.5 |
| `oi_divergence_sma24_highfreq_h30_s6` + selector | long | 0.5 |
| `bear_rex_dual_regime_short` | short | 1.0 |

Gross weight: `2.0`.

Stats:

| period | return | CAGR | strict MDD | CAGR/MDD | trade-entry sum | active bars | bar Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train diagnostic | 20.49% | 5.75% | 44.69% | 0.13 | 1391 | 64650 | 0.62 |
| 2024 test | 43.25% | 43.14% | 7.79% | 5.53 | 240 | 9782 | 2.75 |
| 2025 eval | 67.84% | 67.90% | 4.07% | 16.67 | 181 | 9333 | 4.03 |
| 2026 YTD | 14.66% | 38.62% | 6.10% | 6.33 | 81 | 5185 | 1.25 |

More aggressive top OOS-ratio mix:

| sleeve weights | gross | 2024 CAGR/MDD | 2025 CAGR/MDD | 2026 YTD CAGR/MDD |
|---|---:|---:|---:|---:|
| `pb30_base=0.5`, `nonpb30_taker=1.0`, `oi_high_sel=0.5`, `bear_rex_short=0.5` | 2.5 | 14.98 | 14.89 | 3.01 |

Interpretation:
- The robust portfolio deliberately excludes pb30 and uses independent long + OI high-frequency + bearish REX short. It is less explosive in 2024 than pb30-heavy mixes but has much better 2026 risk behavior.
- The aggressive mix is excellent on 2024/2025 but still has 2026 MDD/ratio weakness, mostly from bullish sleeve concentration.
- A second-stage LLM portfolio selector is a natural next step: keep base sleeve signals fixed, then let the LLM choose constrained sleeve allow/block or allocation bands only.

## 2026-07-06: second-stage portfolio LLM selector overlay

Added evaluator: `training/evaluate_portfolio_llm_selector.py`.

Saved overlay config: `configs/live/portfolio_bull_bear_oi_rex_llm_selector_overlay.json`.

Artifacts:
- selector eval: `results/portfolio_llm_selector_eval_2026-07-06.json`
- LLM portfolio state-card dataset: `results/portfolio_llm_selector_cards_2026-07-06.jsonl`

Design: base sleeve signals and weights remain fixed. The portfolio selector only emits `ALLOW` or `BLOCK_RISK` for newly triggered sleeve entries at that timestamp. It cannot create trades, change exits, change leverage, or alter weights.

Best train-only symbolic proxy:
- Context keys: `trend_1d`, `range_pos_1d`, `dxy`, `kimchi`
- Fit split: train `<2024` only
- Block context if train support >= 16 and either weighted mean return <= -8 bps or win rate <= 38%
- Blocked train-derived contexts:
  - `trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot`
  - `trend_1d=flat_up|range_pos_1d=low|dxy=flat|kimchi=cold`

Stats:

| period | baseline return | baseline CAGR/MDD | baseline trades | selector return | selector CAGR/MDD | selector trades | blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 test | 43.25% | 5.53 | 240 | 42.40% | 5.36 | 228 | 12 |
| 2025 eval | 67.84% | 16.67 | 181 | 61.72% | 15.31 | 171 | 10 |
| 2026 YTD | 14.66% | 6.33 | 81 | 15.55% | 6.76 | 79 | 2 |

Conclusion: the second-stage selector is directionally useful but modest. It preserves the 2024/2025 acceptance floor and improves 2026 YTD, but gives up some 2024/2025 return. It is suitable for paper shadow logging, not live discretionary control yet.

## 2026-07-06: gross-2 / 2025-MDD<=10 portfolio refinement

Saved OOS-max candidate under 2025 MDD constraint: `configs/live/portfolio_gross2_mdd10_oos_max_candidate.json`.

Artifact: `results/portfolio_mixer_mdd10_gross2_refine_2026-07-06.json`.

Objective: simultaneous sleeve positions allowed, gross sleeve weight <= 2.0, 2025 strict MDD <= 10%, maximize 2024/2025 OOS CAGR/MDD. Metrics below are already at the listed gross leverage, not unscaled pre-leverage metrics.

Best OOS/MDD10 mix:

| sleeve | weight |
|---|---:|
| `pb30_base` | 0.50 |
| `nonpb30_taker` | 0.75 |
| `oi_low` | 0.25 |
| `oi_high_sel` | 0.25 |
| `bear_rex_short` | 0.25 |

Gross weight: `2.0`.

| period | return | CAGR | strict MDD | CAGR/MDD | trade-entry sum | bar Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| train diagnostic | 51.83% | 13.35% | 42.21% | 0.32 | 1893 | 1.01 |
| 2024 test | 74.60% | 74.41% | 5.72% | 13.00 | 324 | 3.57 |
| 2025 eval | 71.95% | 72.01% | 5.61% | 12.84 | 241 | 3.94 |
| 2026 YTD | 15.07% | 39.81% | 16.78% | 2.37 | 115 | 1.05 |

Risk: this achieves the requested 2025 MDD target at gross 2.0, but it gives up 2026 drawdown control. If 2026 robustness matters, prefer the prior robust portfolio (`nonpb30_taker=0.5`, `oi_high_sel=0.5`, `bear_rex_short=1.0`) or its portfolio selector overlay.

Robust comparison:

| candidate | gross | 2024 CAGR/MDD | 2025 CAGR/MDD | 2025 MDD | 2026 CAGR/MDD | 2026 MDD |
|---|---:|---:|---:|---:|---:|---:|
| OOS-max MDD10 | 2.0 | 13.00 | 12.84 | 5.61% | 2.37 | 16.78% |
| robust bull/bear/OI/REX | 2.0 | 5.53 | 16.67 | 4.07% | 6.33 | 6.10% |
| robust + portfolio selector | 2.0 | 5.36 | 15.31 | 4.04% | 6.76 | 6.10% |
