# Alpha edge investigation — 2026-06-17

## Protocol
- Data: `data/2023-01-01_2026-02-28_d2a88c0700504d6a5e15bc3839ad84b6.csv.gz` plus leak-safe backward-asof external joins from `/home/pakchu/workspace/wave_trading`.
- Strict execution: entry delay 1 bar, costs, non-overlapping holds, bar-by-bar intrabar adverse excursion included in strict MDD.
- Split discipline for combo scan:
  - train: 2023-01-01 .. 2024-06-30
  - test/ranking: 2024-07-01 .. 2025-08-31
  - eval/holdout: 2025-09-01 .. 2026-02-28
- No eval tuning: model weights, score thresholds, and long/short direction are fit from train only; test ranks candidates; eval is final audit.

## Univariate Kimchi premium strict backtests
Best univariate result was `kimchi_premium_change h288 q0.20`:
- eval CAGR 14.42%, strict MDD 6.88%, ratio 2.10
- 66 trades, mean trade +0.107%, p≈0.454, CI includes 0

Other h72/h144 variants were negative. Conclusion: Kimchi premium has visible IC, but as a standalone strict trading rule it is not statistically meaningful.

## Linear feature-combination scan
Candidate groups: external, kimchi-only, trend, range/reversion, candle/flow, funding/OI, and combinations. Ridge L2 values tested: 10, 100, 1000.

Most stable-but-weak candidates:
- `kimchi_plus_trend h288 q0.15 L2=100`: test 15.78/16.14=0.98, eval 24.05/13.11=1.83, 370/147 trades, p≈0.408/0.379.
- `range_reversion h288 q0.20 L2=1000`: test 14.00/22.08=0.63, eval 14.09/14.78=0.95, 406/171 trades, p≈0.447/0.601.

Important rejection:
- `trend h288 q0.10 L2=1000` had eval ratio 3.16, but test ratio only 0.23. This is not a valid success because the test split does not support selecting it.

## Current conclusion
The currently available feature families do not yet contain a robust, statistically meaningful alpha satisfying CAGR/strict-MDD ≥ 3 under train/test/eval discipline. The useful signal is weak and concentrated around 2-day horizon trend/reversion + Kimchi context, but it is insufficient as a direct policy.

## Next direction
Move from linear/global rules to regime-aware interaction discovery:
1. Detect regimes from past-only volatility/range/trend/Kimchi/DXY states.
2. Fit simple rules inside regimes, not globally.
3. Require regime candidates to pass train and test before eval is inspected.
4. Feed only robust regime descriptors into Gemma-based LLM policy; do not ask the LLM to infer raw numeric edge from weak raw features.

## Follow-up: regime-conditioned candidate audit

Candidate discovered by sensitivity scan:
- Regime: `kimchi_premium_change` in train-window low bucket.
- Signal: `trades_ratio` quantile rule.
- Horizon: 288 bars.
- Fit from 2023-01-01..2024-06-30 with rq=0.25/sq=0.25:
  - test 2024-07..2025-08: CAGR 60.70%, strict MDD 8.12%, ratio 7.47, 280 trades, p≈0.004.
  - eval 2025-09..2025-12-01 effective: CAGR 40.60%, strict MDD 11.55%, ratio 3.52, 61 trades, p≈0.062.

External data caveat:
- wave_trading Kimchi/DXY caches end in early/mid December 2025 while the market file extends to 2026-02-27.
- The apparent 2026 eval interval produced no 2026 trades for this candidate; effective OOS trading ended on 2025-12-02.

Longer split audit:
- Fit 2020..2022, test 2023..2024, eval 2025:
  - test failed: CAGR -3.22%, strict MDD 38.04%, 478 trades, p≈0.936.
  - eval 2025 strong: CAGR 52.56%, strict MDD 11.55%, 217 trades, p≈0.013.
- Fit 2020..2023, test 2024, eval 2025:
  - test weak: CAGR 23.29%, strict MDD 18.41%, ratio 1.27, 243 trades, p≈0.267.
  - eval 2025 strong: CAGR 50.03%, strict MDD 11.81%, ratio 4.24, 217 trades, p≈0.017.

Interpretation:
- The candidate is not a timeless alpha. It appears to be a strong 2025 regime-specific alpha.
- It should not be deployed as an always-on rule.
- Next LLM/RL direction: train Gemma to identify when the 2025-like Kimchi-flow regime is active and abstain otherwise, rather than directly predicting every trade from raw numeric bars.

## LLM regime descriptor update: edge_state_v5

Monthly state descriptors were weak (largest effect around d≈0.34), so the useful signal is not a broad monthly regime label alone. Entry-context audit on 2025 eval trades was more informative:
- Overall winners had higher `bb_z`, `rsi_norm`, `range_pos`, `sma48_ratio`, `close_zscore_48`, and lower bearish shadow imbalance than losers.
- LONG winners were more associated with positive taker imbalance / taker buy ratio, higher `bb_z`, higher `rsi_norm`, and stronger upper-shadow context.
- SHORT winners were more associated with higher `close_zscore_48`, `sma48_ratio`, lower `window_drawdown`, stronger volume participation, and lower taker imbalance.

Implemented `edge_state_v5` in `training/vlm_trading_data.py`:
- Builds on `edge_state_v4`.
- Adds Kimchi-flow activation descriptors from the audited rule:
  - `Kimchi Flow Regime`
  - `Long Entry Context`
  - `Short Entry Context`
  - `Regime Failure Cue`
- Adds numeric context scores:
  - `Kimchi Flow Change`, `Kimchi Z`, `Trades Participation`, `Taker Imbalance`
  - `LLM Long Context Score`, `LLM Short Context Score`, `LLM Failure Cue Score`

Interpretation:
- V5 does not encode a deployable always-on rule.
- It gives Gemma explicit language for when the 2025-like Kimchi/liquidity opportunity is present and when to abstain.
- The next training run should compare `edge_state_v4` vs `edge_state_v5` under identical train/test/eval splits before any live integration.

## edge_state_v4 vs edge_state_v5 prompt smoke

A direct prompt-mode comparison on 2025 samples showed why `prompt_style=hybrid` is required:
- `prompt_style=numeric` includes V5 numeric scores but omits symbolic descriptors such as `Kimchi Flow Regime`.
- `prompt_style=hybrid` includes both numeric evidence and symbolic regime/context descriptors.

Smoke result with 128 uniform 2025 samples:
- edge_state_v4: labels LONG 63 / SHORT 65, mean prompt length ≈1637 chars.
- edge_state_v5: labels LONG 63 / SHORT 65, mean prompt length ≈2072 chars.
- V5 prompt includes `Kimchi Flow Regime`, `Long Entry Context`, `Short Entry Context`, and `Regime Failure Cue`.

Operational directive:
- Future Gemma V5 runs should use `--prompt-style hybrid`, not `numeric`, otherwise the language descriptors that make V5 useful are not visible to the model.

## Cheap descriptor-signal audit

Using exported V5 hybrid prompt samples:
- 512 `trade_side` samples: LONG 259 / SHORT 253.
- 512 `trade_gate` samples: TRADE 321 / NO_TRADE 191.

Descriptor mutual information against generic targets was very weak:
- For LONG/SHORT, strongest field was `Regime Memory` at ≈0.016 bits; Kimchi-flow and entry-context fields were lower.
- For TRADE/NO_TRADE, strongest field was `Kimchi Flow Regime` at ≈0.007 bits.

Interpretation:
- V5 descriptors are not useful as generic direction/gate predictors by themselves.
- This supports the prior conclusion: V5 should not be bolted onto the old generic path-outcome target and expected to fix it.
- Next target should be aligned to the discovered edge: Kimchi-flow activation, side-context quality, and abstain decisions around the audited regime-conditional rule.

## Kimchi-flow activation SFT target

Implemented `training/kimchi_flow_activation_sft_data.py` to align the LLM target with the discovered edge instead of generic path-outcome direction labels.

Generated 2025 activation rows from the fixed Kimchi-flow rule:
- Total rows: 217 fixed-rule signal dates.
- Target counts: ACTIVATE 109 / ABSTAIN 108.
- Side counts: LONG 61 / SHORT 48 / NONE 108.
- Split:
  - train: 2025-01..2025-07, 137 rows.
  - val: 2025-08..2025-09, 38 rows.
  - test: 2025-10..2025-12, 42 rows.

Implemented `training/eval_kimchi_flow_activation.py` for target-echo and simple baselines.
Test split proxy:
- target_echo oracle: +25.32 pct-points over 20 activations.
- all_abstain: 0 pct-points over 0 activations.
- all_activate_long: +10.75 pct-points over 42 activations.

Interpretation:
- Unlike generic LONG/SHORT or TRADE/NO_TRADE labels, this target is aligned with the actual discovered edge.
- Next step is a small Gemma SFT run on train, validation on val, then test activation predictions mapped back to fixed-rule returns.

## Gemma-4 Kimchi-flow activation SFT smoke result

Trained `google/gemma-4-E4B-it` LoRA on the 2025 Kimchi-flow activation target:
- Train rows: 137, split 2025-01..2025-07.
- Config: LoRA r=8/alpha=16/dropout=0.05, max_seq_length=3072, max_steps=40, lr=2e-5.
- Runtime: 323.5s, train_loss 1.286, epoch 1.146.
- Checkpoint: `checkpoints/gemma4_kimchi_flow_activation_v5_r8_step40` (~404MB with checkpoint-40).

Evaluation modes added to `training/eval_kimchi_flow_activation.py`:
- `model`: free JSON generation then strict parser.
- `candidate_score`: fixed JSON candidate logprob selection among ACTIVATE_LONG, ACTIVATE_SHORT, ABSTAIN_BAD, ABSTAIN_MARGINAL.

Leak-safe holdout results:

| split | mode | pred sum ret pct | oracle sum ret pct | pred activations | exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| val 2025-08..09 | all_abstain | 0.000 | 10.286 | 0 | 0.026 |
| val 2025-08..09 | all_activate_long | 0.615 | 10.286 | 38 | 0.289 |
| val 2025-08..09 | model/free generation | 1.508 | 10.286 | 12 | 0.079 |
| val 2025-08..09 | candidate_score | 0.615 | 10.286 | 38 | 0.447 |
| test 2025-10..12 | all_abstain | 0.000 | 25.315 | 0 | 0.119 |
| test 2025-10..12 | all_activate_long | 10.749 | 25.315 | 42 | 0.167 |
| test 2025-10..12 | model/free generation | -0.319 | 25.315 | 11 | 0.095 |
| test 2025-10..12 | candidate_score | 10.653 | 25.315 | 39 | 0.405 |

Interpretation:
- The target/oracle is profitable, but the current prompt+SFT does not learn a profitable activation boundary.
- Free generation is not reliable: it emits unseen regime strings such as `UPTREND`/`RANGE`, which the parser must coerce back into valid labels.
- Candidate scoring removes JSON-format noise and improves exact accuracy, but mostly collapses to near all-activate behavior; it does not add selection alpha.
- This is not a deployable result. The next improvement should not be “more steps” first; it should diagnose which past-only features distinguish the missed large winners from the false activations, then expose those features in a simpler activation target/prompt.

## Activation feature separability diagnosis

Added `training/diagnose_activation_feature_separability.py` to check whether the prompt features contain stable, past-only separability for the Kimchi-flow activation target.

Result summary from `results/activation_feature_separability_v5_2025.json`:
- Train 2025-01..07: target activation is moderately separable by `llm_long_context_score` (AUC 0.67), `side_pressure_score` (0.625), `past_return_1h` (0.622), and `tradeability_score` (0.620).
- Val 2025-08..09: strongest features shift to `past_return_2h` (0.769), `side_pressure_score` (0.726), `llm_short_context_score` (0.720), `range_position` (0.679).
- Test 2025-10..12: strongest apparent split shifts again to `dxy_z` (0.670), while several flow/context features weaken or reverse (`order_flow_imbalance`/`taker_imbalance` AUC 0.390, `llm_failure_cue_score` AUC 0.333).

Interpretation:
- The prompt does contain some local signal, but not a stable activation boundary across 2025 subperiods.
- This explains the Gemma SFT failure: the target/oracle is profitable, but the observable prompt features do not expose a consistent rule that survives val/test drift.
- Next step should be a feature/target redesign, not merely longer SFT. Candidate directions:
  1. restrict prompts to features with cross-split stable sign;
  2. add explicit month/regime shift descriptors and ask Gemma for confidence-calibrated abstention;
  3. create pairwise preference/ranking examples inside the same local regime window rather than absolute GOOD/BAD labels;
  4. validate feature separability before spending GPU on another SFT run.

## Stable compact prompt v1 SFT

Built `training/build_stable_activation_sft_data.py` to remove unstable prompt noise before another Gemma run.
Selection protocol:
- Feature selection uses train 2025-01..07 and val 2025-08..09 only.
- Test 2025-10..12 is not used to choose features.
- Numeric prompt fields are selected only when train and val activation AUC have the same direction and both exceed the minimum edge threshold.

Selected stable-v1 fields:
`llm_long_context_score`, `side_pressure_score`, `past_return_2h`, `llm_short_context_score`, `tradeability_score`, `long_evidence_votes`, `range_position`, `past_return_1h`, `past_path_return_6h`, `kimchi_flow_change`.

Prompt compression:
- Original V5 hybrid prompt mean length ≈2546 chars.
- Stable-v1 compact prompt mean length ≈692 chars.

Gemma-4 LoRA stable-v1 train-only run:
- Train rows: 137, steps: 40, max_seq_length: 1024.
- Test free generation improved from the prior negative result to +3.356 pct-points over 10 activations.
- Test candidate scoring remained effectively all-activate-like: +10.749 pct-points over 42 activations.

Gemma-4 LoRA stable-v1 train+val run:
- Train rows: 175, steps: 80, max_seq_length: 1024.
- Test free generation: +2.601 pct-points over 5 activations.
- Test candidate scoring: +2.524 pct-points over 7 activations.

Interpretation:
- Compact stable features improved free-generation from negative to positive, so prompt noise was part of the failure.
- However, the LLM still underperforms the simple all-activate-long baseline on the final test.
- Adding val to training made the model too conservative and missed too many large winners.
- Current best Gemma SFT result is not deployable. Next check: whether a transparent stable-feature score/threshold can beat all-activate on test. If not, SFT has no stable boundary to learn.

## Transparent stable-score threshold check

Added `training/evaluate_stable_activation_score.py` to verify whether the selected stable features contain a simple threshold boundary that beats all-activate before asking Gemma to learn it.

Protocol:
- Fit scaler on train only.
- Use stable-v1 selected features and train/val AUC-edge weights.
- Select threshold on val only with minimum validation trade count.
- Evaluate final result on untouched test.

Result from `results/eval_stable_activation_score_v1.json`:
- Train: score threshold +28.505 pct-points vs all-activate +27.050.
- Val: score threshold +6.877 pct-points vs all-activate +0.615.
- Test: score threshold -3.156 pct-points vs all-activate +10.749.

Monthly test decomposition:
- 2025-10: all-activate -0.8 pct-points, stable-v1 train-only model +2.88. A selective filter helps.
- 2025-11: all-activate +11.6 pct-points, oracle +17.3, but stable-v1 model captures only +0.48 or 0.0 depending on checkpoint. Selective filtering hurts by missing broad winners.
- 2025-12: only one sample, not meaningful.

Interpretation:
- There are at least two sub-regimes inside the final holdout: October needs filtering; November rewards broad activation.
- The next model should not learn a single global activation threshold. It needs a regime switch between selective mode and broad-on mode, or a target that explicitly learns whether filtering is useful in the current macro/micro context.

## Completed-weekly regime features: edge_state_v6

Added leak-safe completed-weekly features to `preprocessing/market_features.py` and exposed them through `edge_state_v6`:
- `weekly_return_1w`
- `weekly_return_4w`
- `weekly_range_1w`
- `weekly_range_pos`
- `weekly_drawdown_4w`
- `weekly_filter_score`
- symbolic `Weekly Regime` and `Weekly Location`

Leakage guard:
- Weekly bars are resampled as week-ending Sunday candles.
- The current/incomplete weekly candle is excluded by shifting weekly features by one completed week before backward as-of alignment to each 1m row.
- Therefore row `t` only sees weekly information from weeks completed before `t`.

Generated `data/kimchi_flow_activation_edge_state_v6_2025*.jsonl` from the same fixed Kimchi-flow trade report used for v5. Prompt mean length increased from ~2546 chars to ~2811 chars.

Feature separability:
- Weekly fields are not strong train/val individual GOOD/BAD classifiers.
- On final test, `weekly_range_1w` becomes a meaningful feature (AUC ≈0.605, return corr ≈0.208), but this was not visible enough in train/val to justify using it as a direct activation threshold.

Weekly bucket audit:
- Test `WEEKLY_DEFENSIVE_FILTER`: 16 rows, all-activate +13.24 pct-points, oracle +15.93.
- Test `WEEKLY_MIXED`: 12 rows, all-activate -4.91 pct-points, oracle +2.17.
- Test filter-score=1: 28 rows, all-activate +11.23 pct-points, oracle +18.80.
- Test filter-score=0: 8 rows, all-activate -1.89 pct-points, oracle +3.74.

Interpretation:
- Weekly context is useful, but not as a simple “defensive means filter” rule.
- In the 2025-11 holdout, high weekly drawdown/defensive context coincided with broad Kimchi-flow opportunity, which explains why the prior selective activation model missed many winners.
- Next target should explicitly learn a higher-level switch such as `BROAD_ON_AFTER_WEEKLY_STRESS` vs `SELECTIVE_IN_MIXED_WEEKLY_REGIME`, then apply lower-timeframe activation only inside the selective branch.

## Multi-timeframe regime context: edge_state_v7

Generalized completed-weekly features into completed higher-timeframe features for 4h/1d/3d/1w:
- `htf_4h_*`, `htf_1d_*`, `htf_3d_*`, `htf_1w_*`
- each set includes completed-bar return, 4-bar return, range, range position, drawdown, and stress score.

Leakage guard:
- Every higher-timeframe candle is resampled, shifted by one completed candle, then backward as-of joined to 1m rows.
- The current incomplete 4h/1d/3d/1w candle is never used.

Implemented `edge_state_v7`, exposing numeric and symbolic 4H/1D/3D/1W regimes plus an aggregate `MTF Activation Mode`.
Generated `data/kimchi_flow_activation_edge_state_v7_2025*.jsonl`.

Important implementation fix:
- The feature-diagnosis regex previously ignored numeric feature names starting with digits, so `4H`, `1D`, `3D`, `1W` numeric fields were missed.
- Updated `diagnose_activation_feature_separability.py`, `build_stable_activation_sft_data.py`, and `evaluate_stable_activation_score.py` to parse digit-leading feature labels.

Findings:
- Individual activation GOOD/BAD separability still does not select higher-timeframe features under train+val stability. Stable-v7 selected only short-horizon/Kimchi context fields and still failed test (`pred -1.61` vs all-activate `+10.75`).
- Therefore higher-timeframe context should not be treated as a direct per-trade activation threshold.
- Bucket audit shows it is useful as a regime-switch layer:
  - Test `1W_STRESS`: 16 rows, all-activate +13.24, oracle +15.93 → broad-on is mostly fine.
  - Test `1W_MIXED`: 12 rows, all-activate -4.91, oracle +2.17 → selective filter needed.
  - Test `1W_CHOP`: 10 rows, all-activate -0.16, oracle +3.75 → selective filter needed.
  - Test `3D_STRESS`: 16 rows, all-activate +8.05, oracle +11.11 → stress can be opportunity, not just risk.

Interpretation:
- The useful long-timeframe information is mainly 3D/1W, not 4H/1D in this sample.
- The next LLM target should be hierarchical:
  1. determine higher-timeframe mode (`BROAD_ON_STRESS`, `SELECTIVE_MIXED_OR_CHOP`, `AVOID`);
  2. only in selective mode, apply lower-timeframe activation/side judgement.
- This preserves the user goal of using LLM/RL while giving the LLM a task aligned with its strength: regime explanation and mode selection rather than noisy numeric micro-thresholding.

## Hierarchical MTF mode and pairwise selective ranking

Implemented `training/mtf_mode_policy_dataset.py` to test a hierarchical policy:
1. choose higher-timeframe mode (`BROAD_ON`, `SELECTIVE`, `AVOID`) from 3D/1W/MTF buckets;
2. in `SELECTIVE`, apply lower-timeframe confirmation.

Findings:
- With a simple lower-timeframe rule, train+val-fit MTF mode policies did not beat all-activate on test.
- With oracle selective decisions, the hierarchy recovers the upper bound, confirming the bottleneck is the selective lower-timeframe chooser, not the concept of mode switching.

Implemented `training/kimchi_flow_pairwise_dataset.py` and `training/eval_pairwise_choice.py` to convert selective lower-timeframe judgement into pairwise ranking:
- Train: 600 A/B pairs from 2025-01..07.
- Val: 259 pairs from 2025-08..09.
- Test: 79 pairs from 2025-10..12.
- Orientation is balanced; always-A/always-B baselines are ~50%.

Gemma-4 pairwise LoRA (`checkpoints/gemma4_pairwise_v7_r8_step80`):
- Val accuracy: 69.1% (179/259), clear improvement over ~50% baseline.
- Test accuracy: 50.6% (40/79), no improvement over baseline.

Interpretation:
- Pairwise/ranking is learnable within the validation regime, so it is a better target shape than flat activation.
- It still fails across the October/November/December regime shift, reinforcing that the next missing piece is broader regime coverage, not just target format.
- Next data step: build the same MTF/pairwise dataset over a longer historical period with external features, not only 2025, so pairwise ranking sees stress/broad-on transitions before final holdout.

## Longer timeframe pairwise expansion: 2024 train → 2025 holdout

Expanded the MTF pairwise-ranker training set from the 2024 fixed-report `test` trades and used 2025 only as forward evaluation:
- Train source: `data/kimchi_flow_pairwise_v7_2024.jsonl`, 1,773 balanced A/B pairs.
- Model: Gemma-4 E4B instruction LoRA, r=8, alpha=16, dropout=0.05, max length 2048, 120 steps.
- Evaluation mode: candidate JSON log-probability comparison, not sampled generation, via `training/eval_pairwise_choice.py --mode model_logprob`.

Results:
- 2025 train-period pairs: 53.3% (320/600), weak.
- 2025 validation-period pairs: 63.3% (164/259), good but regime-specific.
- 2025 final test-period pairs: 43.0% (34/79), worse than balanced baseline.

Added `training/audit_pairwise_regime_coverage.py` to verify whether the final holdout regimes were covered by the training set.
Coverage audit (`results/audit_pairwise_v7_2024_train_to_2025_test_gemma4_2024_step120.json`):
- Final 2025 test contains 35 rows of `3D_STRESS|1W_STRESS`.
- 2024 training contains 0 rows of `3D_STRESS|1W_STRESS`.
- Model accuracy in that unseen bucket: 25.7% (9/35), with a strong wrong-side bias toward B.
- On known training buckets only, final-test accuracy is 56.8% (25/44), still modest but no longer catastrophic.

Interpretation:
- The latest failure is not just model size or prompt shape. It is regime coverage.
- Longer-timeframe context exposed the missing condition: the final holdout is dominated by a 3D+1W stress-stress regime absent from 2024 training.
- A deployable LLM/RL policy must include either:
  1. enough historical stress-stress examples before the final holdout, or
  2. an explicit out-of-distribution/unknown-regime abstention layer, then only trade known/regime-supported buckets.
- This supports adding longer historical data and regime-coverage gating before any live trading integration.

## Long-history mirrored pairwise POC

Built a correct 5m long-history cache from wave_trading 1m BTCUSDT:
- Raw source: `/home/pakchu/workspace/wave_trading/data/2020-01-01_2025-12-15_52bcaa88960cc9b2e902e496475d0fec.csv.gz`.
- 5m aggregate: `data/btcusdt_5m_2020-01-01_2025-12-15.csv.gz`, 626,399 rows.
- External cache: `data/cache_market_ext_5m_2020-01-01_2025-12-02.csv.gz`, 622,765 rows with DXY proxy, Kimchi premium, USDKRW.

Important correction:
- The first long-cache attempt used 1m bars, so `horizon=288` became 4.8h instead of the intended 24h. That result is invalid for comparison.
- The corrected 5m fixed-rule report reproduces the intended bar semantics:
  - train 2020-2023 trades: 940
  - 2024 test: CAGR 23.3 / strict MDD 18.4 / 243 trades
  - 2025 eval: CAGR 50.1 / strict MDD 11.8 / 216 trades / p≈0.017

Regime coverage repair:
- 2020-2023 pairwise data contains 85 `3D_STRESS|1W_STRESS` pairs.
- Combined 2020-2024 train contains no unseen bucket for 2025 final test.
- Training on non-mirrored 2020-2024 data collapsed to almost always-A:
  - 2025 train 48.8%, val 51.4%, test 49.4%.
  - This confirms pairwise SFT has strong position-bias risk even when target counts are globally balanced.

Implemented `--mirror-pairs` in `training/kimchi_flow_pairwise_dataset.py`:
- For each good-vs-bad pair, add both original and swapped A/B prompts with opposite target.
- Mirrored 2020-2024 train: 50,764 pairs, exactly A=25,382 / B=25,382.
- Stress-stress coverage after mirroring: 4,216 pairs.

Gemma-4 mirrored pairwise LoRA POC:
- Train: random 8,000 mirrored pairs from 2020-2024, LoRA r=8, LR=1e-5, 100 steps.
- 2025 train: 61.2% (367/600), A/B predictions 259/341.
- 2025 val: 62.5% (162/259), A/B predictions 125/134.
- 2025 final test: 51.9% (41/79), A/B predictions 18/61.

Margin audit:
- Logprob margin is calibrated on train/val (higher margin → higher accuracy).
- On 2025 final test it is anti-calibrated (top-margin rows are worse), so margin gating would be another overfit.

Interpretation:
- Mirroring is a necessary fix: it removes A/B position collapse and should be kept for all pairwise LLM/RL stages.
- However, the long-history mirrored LLM still does not produce a deployable final-test edge. It is only slightly above random on 79 pairs.
- The next substantive step should map pairwise predictions back into actual trade selection/returns and then search for target formulations that include explicit regime outcome labels, not only pair ordering.

## Pairwise economic-return mapping check

Although mirrored pairwise final-test accuracy is only 51.9%, mapping each pair prediction to the realized return of the selected candidate shows a stronger economic signal:

Mirrored Gemma-4 pairwise model (`checkpoints/gemma4_pairwise_v7_2020_2024_mirror_r8_step100_lr1e5`):
- 2025 train: mean selected return +0.4345 pct per pair, n=600, one-sided p≈6.6e-19.
- 2025 val: mean selected return +0.1719 pct per pair, n=259, one-sided p≈8.5e-06.
- 2025 final test: mean selected return +0.3561 pct per pair, n=79, one-sided p≈0.0046.

Bucket notes on 2025 final test:
- `3D_STRESS|1W_STRESS`: +0.4925 pct mean, n=35.
- `3D_MIXED|1W_CHOP`: +0.6146 pct mean, n=9.
- `3D_MIXED|1W_MIXED`: near flat, n=27.

Caution:
- Pair rows are not independent executable trades; the same trade candidate can appear in multiple pairs.
- Therefore this is not yet a valid live-trading backtest or CAGR/MDD proof.
- The next required step is to aggregate pairwise wins into one score per candidate signal date, remove duplicate comparisons, and backtest the resulting selected/abstained fixed-rule trades with strict MDD.

## Candidate-level aggregation and future-pool leakage correction

Converted pairwise predictions into candidate-level scores and strict OHLC backtests.

First attempt: split-internal candidate aggregation
- Rebuilt pairwise rows with candidate A/B metadata and aggregated pairwise logprob margins into one score per candidate.
- Natural threshold `score_mean > 0` on split-internal pairs:
  - 2025 train: CAGR 67.7 / strict MDD 4.75 / 58 trades / p≈0.0013.
  - 2025 val: CAGR 41.3 / strict MDD 3.32 / 18 trades / p≈0.020.
  - 2025 test: CAGR 23.8 / strict MDD 4.10 / 13 trades / p≈0.33.
- Train/val-selected threshold `score_mean > -0.25` looked stronger:
  - 2025 test: CAGR 87.6 / strict MDD 4.35 / 24 trades / p≈0.12.

Critical correction:
- Split-internal candidate aggregation is not live-causal because a candidate is scored by comparing it to other candidates from the same future evaluation window.
- This is not feature leakage, but it is ranking-pool leakage: at live decision time, future candidates do not exist yet.
- Therefore the split-internal candidate backtest must not be treated as a deployable result.

Implemented causal historical-reference scoring:
- `training/build_pairwise_reference_score_dataset.py` builds pairs where each eval candidate is compared only against 2020-2024 historical reference candidates.
- `training/eval_pairwise_candidate_backtest.py --candidate-role eval_candidate` aggregates only live eval candidates, excluding historical references from execution.

Causal historical-reference results with Gemma-4 mirrored pairwise model:
- refs_per_side=4:
  - 2025 val: CAGR 15.9 / strict MDD 2.91 / 14 trades / p≈0.33.
  - 2025 test: CAGR -15.7 / strict MDD 10.19 / 9 trades / p≈0.53.
- refs_per_side=16:
  - 2025 val: CAGR 19.0 / strict MDD 3.32 / 14 trades / p≈0.24.
  - 2025 test: CAGR -27.1 / strict MDD 10.19 / 7 trades / p≈0.15.

Interpretation:
- The apparent candidate-level edge was largely dependent on future-pool comparison.
- The honest live-causal reference-scoring version fails.
- Keep the useful engineering pieces (mirrored pairs, candidate metadata, causal reference scoring), but do not promote this Gemma pairwise policy to live trading.
- Next target should avoid cross-sectional future-pool ranking and instead train a single-candidate value/regime outcome predictor calibrated against historical references known at decision time.
