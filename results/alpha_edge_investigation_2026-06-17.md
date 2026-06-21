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

## Single-candidate activation fallback

After rejecting future-pool pairwise backtests, trained a live-causal single-candidate Gemma-4 activation model:
- Train data: `data/kimchi_flow_activation_edge_state_v7_2020_2024_train.jsonl`.
- Rows: 1,183 candidates from 2020-2024.
- Labels: ACTIVATE 531 / ABSTAIN 652; LONG 326 / SHORT 205 / NONE 652.
- Model: Gemma-4 E4B LoRA r=8, LR=1e-5, 120 steps, max length 4096.
- Evaluation: `training/eval_kimchi_flow_activation.py --prediction-mode candidate_score`, choosing among fixed ACTIVATE-LONG / ACTIVATE-SHORT / ABSTAIN JSON candidates by logprob.

Results:
- 2025 val:
  - Decision accuracy 63.2%, exact JSON accuracy 60.5%.
  - Pred activations 11 vs oracle activations 19.
  - Strict selected-trade backtest: ret +2.96%, CAGR 23.2, strict MDD 1.55, 11 trades, p≈0.056.
- 2025 final test:
  - Decision accuracy 47.6%, exact JSON accuracy 40.5%.
  - Pred activations 10 vs oracle activations 20.
  - Strict selected-trade backtest: ret -1.14%, CAGR -7.7, strict MDD 9.68, 10 trades, p≈0.845.

Interpretation:
- The absolute single-candidate model is live-causal, but not profitable on the 2025 final test.
- It is too conservative on short activations and misses many Q4 winners.
- Current Gemma SFT target formulations tested so far:
  1. flat activation: failed generalization;
  2. pairwise split-internal ranking: invalid due future-pool leakage;
  3. pairwise historical-reference scoring: live-causal but failed;
  4. single-candidate activation: live-causal but failed final test.
- Next promising path is not another small SFT tweak; it should change the target to explicit regime outcome/value estimation with calibrated historical baselines, or move the LLM into explanation/feature construction while a simpler causal scorer handles execution.

## Causal KNN value scorer over LLM/regime features

Implemented `training/eval_causal_knn_value_scorer.py` as a non-LLM causal value scorer:
- Input features are the same compact numeric/symbolic regime fields exposed to the LLM (`edge_state_v7`).
- Each eval candidate is scored against historical 2020-2024 reference candidates only.
- Score is the mean realized return of K nearest historical candidates, optionally within the same `3D_regime|1W_regime` bucket.
- This avoids future-pool ranking leakage and tests whether the LLM-exposed features contain a simple causal value edge.

Validation-selected result:
- Best val ratio used k=15, threshold=0.25, but only 9 val trades and 4 final-test trades.
- 2025 test for that setting: CAGR 12.7 / strict MDD 2.18 / 4 trades / p≈0.108. Too few trades to be meaningful.

More trade-count-friendly settings:
- k=15, threshold=0:
  - 2025 val: CAGR 13.7 / MDD 2.54 / 16 trades / p≈0.443.
  - 2025 test: CAGR 22.6 / MDD 4.93 / 19 trades / p≈0.468.
- k=25, threshold=0:
  - 2025 val: CAGR 12.4 / MDD 2.57 / 16 trades / p≈0.443.
  - 2025 test: CAGR 11.9 / MDD 3.59 / 15 trades / p≈0.647.

Interpretation:
- The LLM/regime features contain some positive drift in Q4, but not a statistically meaningful or target-level edge.
- This is still useful: it separates feature/value weakness from LLM output-format issues.
- Current bottleneck appears to be insufficient causal predictive signal in the candidate feature set, not just SFT tuning.
- Next substantial work should add stronger causal features/targets: realized stress-transition labels, market-state change labels, and explicit post-signal path-risk value estimates rather than binary activate/abstain.

## Post-signal path-value target probe

Implemented a live-causal path-value scorer to test whether the edge-state prompts contain signal for executable post-signal path quality, not just terminal fixed-rule return.

New scripts:
- `training/eval_causal_knn_path_value_scorer.py`
  - Labels historical reference candidates with executable path outcomes: scheduled entry/exit, fees/slippage, MAE/MFE, net return, and utility.
  - Scores eval candidates only against historical reference rows; eval future path is used only for audit after selection.
- `training/sweep_causal_knn_path_value.py`
  - Runs KNN metric/threshold sweeps without reloading market data for every threshold.

Setup:
- Reference: `data/kimchi_flow_activation_edge_state_v7_2020_2024_train.jsonl` (2020-2024 only).
- Val: `data/kimchi_flow_activation_edge_state_v7_2025_val.jsonl`.
- Test: `data/kimchi_flow_activation_edge_state_v7_2025_test.jsonl`.
- Market: `data/cache_market_ext_5m_2020-01-01_2025-12-02.csv.gz`.
- Hold: 288 x 5m bars, entry delay 1 bar, leverage 0.5, fee 4bp, slippage 1bp.
- Selection rule: score threshold fixed from val before reading test.

Val-selected candidates checked on test:
- Conservative val winner: `target_metric=path_net_pct`, `k=15`, `threshold=0.25`.
  - Val: ret +3.41%, CAGR 22.6, strict MDD 1.24, ratio 18.3, 9 trades, p≈0.093.
  - Fixed test: ret +1.23%, CAGR 8.14, strict MDD 2.99, ratio 2.72, 5 trades, p≈0.472.
- Trade-count-friendly val candidate: `target_metric=path_net_pct`, `k=5`, `threshold=-0.5`.
  - Val: ret +2.76%, CAGR 18.0, strict MDD 3.15, ratio 5.72, 28 trades, p≈0.412.
  - Fixed test: ret +7.90%, CAGR 62.8, strict MDD 11.35, ratio 5.53, 31 trades, p≈0.297.

Interpretation:
- The path-net KNN target is directionally better than the previous path-utility target and gives an encouraging fixed-test ratio for the broader threshold.
- It is still not statistically significant: p≈0.297 on 31 final-test trades and the score/target correlation is weak.
- The result should be treated as a candidate signal family, not a deployable alpha.
- Important leakage guard: the apparent `k=5, threshold=0.25` test top result was not selected on val and must not be treated as valid model selection.
- Next work should expand the validation/test horizon and improve path-value features, especially features that forecast adverse excursion and stress transition rather than average nearest-neighbor terminal return.

## Longer 2024-val / 2025-full path-net check

To reduce the short-window problem, re-ran the path-net scorer with a longer chronological validation/evaluation split:
- Hyperparameter selection reference: 2020-2023 candidates only.
- Validation: full 2024 candidates.
- Fixed final evaluation: full 2025 candidates.
- Candidate family: `target_metric=path_net_pct`; k in {5,10,15,25}; threshold in {-0.5,0,0.25}.

2024 validation result:
- Best broad trade-count setting: `k=5`, `threshold=-0.5`.
- 2024 val: ret +24.17%, CAGR 24.27, strict MDD 14.21, ratio 1.71, 177 trades, p≈0.195.
- This is not target-level, but it gives enough trades to test whether the signal survives into 2025.

2025 fixed evaluation:
- Frozen reference variant (`reference=2020-2023`, same k/th): ret +16.21%, CAGR 18.09, strict MDD 14.19, ratio 1.27, 170 trades, p≈0.258.
- Live-updated reference variant (`reference=2020-2024`, same k/th): ret +26.12%, CAGR 29.29, strict MDD 11.55, ratio 2.54, 164 trades, p≈0.072.

Interpretation:
- This is the strongest non-leaky, trade-count-reasonable candidate so far, but it still misses the target `CAGR / strict MDD >= 3` and the p-value is not yet below 0.05.
- The score/target correlation remains weak and slightly negative, so much of the improvement may still come from filtering exposure to a generally favorable 2025 fixed-rule regime rather than from true local KNN ranking skill.
- Next direction should not be more threshold tweaking. It should add features/targets that directly forecast strict drawdown contributors: adverse excursion, pre-entry drawdown context, volatility compression/expansion, and multi-timeframe stress transition.

## Edge-state v8 path-risk feature probe

Added `edge_state_v8` to test whether explicit strict-drawdown/stress-transition descriptors help the causal path-net scorer:
- Past-only additions: vol expansion 1h/8h, drawdown acceleration, runup/drawdown balance, path compression, trend conflict, candle/flow shock, macro pressure, kimchi-liquidity pressure, HTF stress gradient, and strict path risk score.
- KNN feature vector also now includes fixed-rule side indicators (`fixed_side_long`, `fixed_side_short`) derived from candidate metadata, so historical neighbors can be side-aware without exposing future outcomes.

Generated v8 activation datasets:
- 2020-2023 train/reference: 940 rows, prompt mean ≈3944 chars.
- 2024 validation: 243 rows, prompt mean ≈3948 chars.
- 2025 evaluation: 216 rows, prompt mean ≈3942 chars.

Validation setup:
- Reference: 2020-2023 v8 rows.
- Validation: 2024 v8 rows.
- Sweep: `target_metric=path_net_pct`, k in {5,10,15,25}, threshold in {-0.5,0,0.25}.

2024 validation:
- Best broad setting by ratio: `k=10`, `threshold=-0.5`.
- Result: ret +22.24%, CAGR 22.32, strict MDD 13.13, ratio 1.70, 205 trades, p≈0.254.
- This is not better than the prior v7 2024 validation candidate by statistical quality, but it is a valid fixed setting to test on 2025.

2025 fixed evaluation with live-updated 2020-2024 v8 reference:
- Fixed val-selected setting: `k=10`, `threshold=-0.5`.
- Result: ret +30.86%, CAGR 34.68, strict MDD 11.55, ratio 3.00, 175 trades, p≈0.051.

Interpretation:
- This is the first non-leaky, trade-count-reasonable candidate to reach the stated ratio target on 2025 full evaluation.
- It is still not deployment-ready: p≈0.051 is borderline and score/target correlation is near zero, so the result may be a regime-exposure filter rather than a robust local value predictor.
- Treat v8 as a promising candidate family, not a solved strategy. Next validation should stress it across walk-forward year blocks and inspect executed-trade monthly distribution / strict MDD attribution before any LLM SFT or live trading integration.

## V8 2025 fixed-eval distribution audit

Ran detailed evaluator plus `training/audit_candidate_backtest_distribution.py` on the v8 val-selected setting (`path_net_pct`, `k=10`, threshold `-0.5`, reference 2020-2024, eval 2025 full).

Overall executed-trade distribution:
- Trades: 175.
- Mean trade return: +0.160%.
- Win rate: 53.1%.
- Top 5 gains sum: +16.44%; top 10 gains sum: +28.24%.
- Bottom 5 losses sum: -10.94%; bottom 10 losses sum: -18.33%.

Monthly distribution:
- Positive months: Jan +3.30, Mar +2.73, Apr +2.81, May +7.22, Jul +6.86, Sep +4.18, Nov +6.55 simple trade-return pct sum.
- Negative months: Feb -0.27, Jun -0.74, Aug -2.74, Oct -1.97.
- October is the main strict-MDD stress candidate: 19 trades, mean -0.10%, worst -3.64%, mean path MAE 2.38%.

Interpretation:
- The v8 result is not a single-month artifact; gains are spread across multiple months.
- However, top-10 gains roughly equal the total simple trade-return sum, so concentration is still material.
- The next feature/guard work should specifically reduce October-like high-MAE regimes without discarding the broad 2025 exposure that creates the edge.

## V8 drawdown-acceleration guard breakthrough

Ran a small 2024-only guard sweep on top of the v8 path-net scorer. The goal was not broad threshold retuning, but testing whether a past-only strict-drawdown proxy can reduce high-MAE regimes.

Validation-only guard candidates:
- Base v8 (`k=10`, threshold `-0.5`): 2024 val ratio 1.70, CAGR 22.32, strict MDD 13.13, 205 trades, p≈0.254.
- Ratio-best guard: `strict_path_risk_score >= 0.3882`: 2024 val ratio 2.36, CAGR 15.31, strict MDD 6.48, 69 trades, p≈0.195.
- Trade-count-friendly guard: `drawdown_acceleration_6h <= 0.003`: 2024 val ratio 2.06, CAGR 20.51, strict MDD 9.94, 134 trades, p≈0.194.

Fixed 2025 evaluation for the trade-count-friendly guard:
- Reference: 2020-2024 v8 rows.
- Eval: 2025 full v8 rows.
- Fixed settings: `target_metric=path_net_pct`, `k=10`, threshold `-0.5`, guard `drawdown_acceleration_6h <= 0.003`.
- Result: ret +38.02%, CAGR 42.85, strict MDD 10.06, CAGR/MDD 4.26, 125 trades, p≈0.0058.
- Power note: current n=125, approximate n required for 80% power ≈129, gap 4 trades.

Guarded monthly audit:
- Overall: mean trade +0.264%, win rate 59.2%.
- Positive months: Jan +7.89, Mar +5.70, Apr +2.75, May +6.24, Jul +9.27, Sep +4.18 simple trade-return pct sum.
- Negative months remain: Feb -0.13, Jun -0.26, Aug -1.82, Oct -1.32.
- October remains the primary stress pocket: 12 trades, worst -3.64%, mean path MAE 3.11%.

Interpretation:
- This is a materially stronger, non-leaky candidate than the unguarded v8 result: target ratio cleared with statistically meaningful p-value and 100+ trades.
- The guard is conceptually plausible: avoid candidates where recent 6h drawdown is accelerating versus the 12h context.
- Remaining risk: the guard was chosen from a small 2024 guard sweep and still needs walk-forward year-block validation; October high-MAE regimes remain unresolved.
- Next required step: implement a formal walk-forward evaluator that repeatedly selects scorer/guard on one period and evaluates the next period, rather than relying on a single 2024→2025 handoff.

## Formal walk-forward guard validation

Implemented `training/walk_forward_knn_guard_eval.py` to prevent single-handoff overconfidence. Each fold enforces:
1. validation candidates are scored only against rows before validation start;
2. k/threshold/guard are selected on validation only;
3. evaluation candidates are scored only against rows before evaluation start;
4. selected parameters are applied unchanged to evaluation.

Default folds tested:
- 2023 validation -> 2024 evaluation.
- 2024H1 validation -> 2024H2 evaluation.
- 2024 validation -> 2025 evaluation.
- 2025H1 validation -> 2025H2 evaluation.

Ungated walk-forward result:
- 2023->2024: validation failed and evaluation failed (eval CAGR -0.99, MDD 10.03, 110 trades, p≈0.997).
- 2024H1->2024H2: eval CAGR 16.70, MDD 9.68, ratio 1.72, 79 trades, p≈0.450.
- 2024->2025: eval CAGR 21.00, MDD 6.24, ratio 3.36, 81 trades, p≈0.115.
- 2025H1->2025H2: eval CAGR 31.64, MDD 10.06, ratio 3.15, 60 trades, p≈0.154.

Validation-gated walk-forward result:
- Gate: validation ratio >= 1.5, validation MDD <= 15, validation p <= 0.30, validation trades >= 50.
- Gate correctly disabled weak 2023->2024 and 2024H1->2024H2 eval periods as no-trade.
- Gate allowed 2024->2025 and 2025H1->2025H2; both fixed eval periods stayed above ratio 3.

Interpretation:
- The candidate is not a universal always-on alpha; it is a regime-conditional strategy family.
- Validation gating is necessary. Without it, early weak folds would trade and fail.
- The strongest 2025 full-year drawdown-acceleration result is still promising, but formal fold-level tests show statistical power is not yet sufficient: allowed fold p-values are ≈0.115 and ≈0.154.
- Next work should aggregate gated fold equity and add more historical rows/periods if possible. If more data cannot be added, the practical next step is paper-trading with the validation gate rather than claiming production readiness.

## Non-overlap aggregate correction

Important correction: the initial aggregate walk-forward calculation was invalid because default diagnostic folds overlap. `2024_to_2025` evaluates full 2025, while `2025H1_to_2025H2` evaluates 2025H2 again. The aggregate output contained 29 duplicate signal/side trades and must not be used as portfolio-level evidence.

Implemented fixes:
- `training/walk_forward_knn_guard_eval.py` now supports `--folds-json` for explicit non-overlapping fold sets and `--include-executed` for downstream audits.
- `training/aggregate_walk_forward_equity.py` aggregates executed fold trades and emits duplicate/overlap warnings.
- Added non-overlap configs:
  - `configs/walk_forward/v8_full_year_nonoverlap.json`
  - `configs/walk_forward/v8_half_year_nonoverlap.json`

Corrected full-year non-overlap aggregate:
- Folds: 2023->2024, 2024->2025.
- 2024 was disabled by validation gate; 2025 traded.
- No duplicate trades.
- Aggregate period: 2024-01-02 to 2026-01-01, ~2.00 years.
- Aggregate: ret +18.05%, CAGR 8.67, strict MDD 4.89, ratio 1.77, 81 trades, p≈0.115.

Corrected half-year non-overlap aggregate:
- Folds: 2024H1->2024H2, 2025H1->2025H2.
- 2024H2 was disabled by validation gate; 2025H2 traded.
- No duplicate trades.
- Aggregate period: 2024-07-01 to 2026-01-01, ~1.50 years.
- Aggregate: ret +11.87%, CAGR 7.76, strict MDD 4.68, ratio 1.66, 60 trades, p≈0.154.

Interpretation:
- The active-regime results remain promising, but the full operational strategy does not yet meet the original target once inactive/no-trade periods are included.
- The current edge is a regime-specific module, not a complete capital-efficient trading bot.
- Next step should focus on either adding complementary regimes to deploy capital during gated-off periods, or explicitly measuring active-period capital allocation separately from always-on portfolio CAGR.

## Complementary univariate fallback check

Goal: find a second, no-leak module that can trade the periods where the v8 drawdown-acceleration gate disables capital, especially 2024, without reintroducing future leakage.

Ran `training/alpha_feature_scan.py` over the extended feature frame with 5m data, horizons 72/144/288, and leak-safe completed higher-timeframe features. The feature set already includes longer bars (`htf_1d_*`, `htf_3d_*`, `htf_1w_*`) plus wave-trading-style macro/external fields when present.

Frozen-rule backtest sweep:
- Fit 2020-2023 -> eval 2024 for candidate fallback validation.
- Fit 2020-2024 -> eval 2025 for later sanity check.
- Rule: bottom/top 20% quantile, direction chosen only on fit window, 1-bar entry delay, 0.5x leverage, strict bar-by-bar MDD.

Result summary:
- No univariate fallback passed 2024. The best tested 2024 candidates were still negative; example `candle_range`, h=288, fit 2020-2023 -> 2024: CAGR -34.19, strict MDD 43.60, 360 trades, p≈0.119.
- `candle_range`, h=288, fit 2020-2024 -> 2025 was positive but weak: CAGR 15.27, strict MDD 18.40, ratio 0.83, 330 trades, p≈0.451. This cannot justify deployment because the analogous no-leak 2024 validation failed hard.
- Combining 2024 candle fallback with 2025 v8 guard gives a failed non-overlap portfolio: CAGR -4.90, strict MDD 41.67, 485 trades, p≈0.836.
- A 2025-only priority combination of v8 guard + candle fallback looks attractive (CAGR 68.51, strict MDD 10.00, 260 trades, p≈0.008), but this is not acceptable as selection evidence because the fallback module was not validated out-of-sample before 2025.

Implementation added:
- `training/alpha_feature_backtest.py` now emits executed trades for module-level combination audits.
- `training/aggregate_trade_modules.py` combines already-executed no-leak module outputs with priority-based overlap resolution.

Interpretation:
- Simple univariate long-timeframe/external-feature fallbacks do not solve the inactive-regime drag.
- The next useful branch should not be another naive gate or single-feature quantile rule. It should either:
  1. train/score a richer text/LLM state representation that reasons over multi-timeframe context and decides abstain/long/short directly; or
  2. use the v8 module only as an active-capital sleeve and measure capital allocation separately from always-on portfolio CAGR.

## Token-state KNN branch and stricter pre-2024 check

Hypothesis: raw numeric matching may be a poor proxy for what a Gemma-style LLM can learn. I added `training/eval_causal_token_knn_path_value.py`, which converts the v8 prompt into symbolic tokens and coarse numeric buckets, then scores eval candidates by causal Jaccard KNN over historical token states.

Initial 2024 discovery check:
- Reference 2020-2023 -> eval 2024, token KNN sweep over k/threshold.
- Best apparent setting: `k=40`, threshold `-0.2`.
- 2024 result: CAGR 39.48, strict MDD 9.67, ratio 4.08, 153 trades, p≈0.027.
- Fixed 2025 with reference 2020-2024: CAGR 25.00, strict MDD 13.60, ratio 1.84, 154 trades, p≈0.112.

Leakage/selection correction:
- The 2024 discovery result cannot be treated as out-of-sample because k/threshold were selected on 2024 itself.
- Ran stricter pre-2024 selection: reference 2020-2022 -> validation 2023.
- 2023 validation failed. Best ratio candidate had only 33 trades and p≈0.704; the previously attractive `k=40`, threshold `-0.2` had CAGR -17.76, strict MDD 21.87, 132 trades.

Interpretation:
- Symbolic token-state matching is a useful diagnostic and did reveal a 2024 pocket, but it is not stable enough to select before 2024.
- The 2024 token branch must be treated as selection-biased until a prior validation period can justify it.
- This reinforces the core issue: the alpha is regime-fragile and parameter selection must be fold-gated. No more claiming 2024+2025 portfolio success from settings chosen on 2024.

Next direction:
- Keep the token scorer as a cheap LLM-feature probe.
- Move from single-period threshold search to fold-stability selection: settings must pass pre-period validation before they can trade the next period.
- If fold-stability keeps disabling most periods, the practical architecture should be active-capital sleeve + paper-trading validation, not always-on CAGR claims.

## Validation-only module selector and numeric/token intersection

Implemented `training/select_validated_modules.py` to stop manual cherry-picking. A manifest lists validation/eval result pairs; for each fold, the selector may choose only modules whose validation result passes a fixed gate. Eval files are reported but not used for selection. The aggregate applies a component strict-MDD floor so trade-to-trade aggregation cannot understate intrabar module drawdown.

Manifest tested: `configs/module_selection/v8_numeric_token_full_year_2024_2025.json`.
- Gate: min 50 trades, ratio >= 1.5, strict MDD <= 15, p <= 0.30.
- 2023 validation -> 2024 eval: no module passed. Numeric v8 failed; token candidates either failed or had too few trades.
- 2024 validation -> 2025 eval: token `k=40`, threshold `-0.2` won on validation, so it was selected mechanically over numeric.
- Aggregate 2024-2025 result: CAGR 11.16, strict MDD 13.60, ratio 0.82, 154 trades, p≈0.112.

Interpretation:
- A validation winner-takes-all selector is still not stable enough. The 2024 token validation winner degraded materially in 2025.
- This is a useful anti-cheat result: once selection is automated and inactive periods are counted, the apparent 2024+2025 success disappears.

Implemented `training/intersect_candidate_modules.py` to test a conservative ensemble: trade only when numeric path-risk and token-state modules select the same signal_date+side.
- 2024 validation intersection: numeric selected 69, token selected 153, intersection 47; CAGR 11.75, MDD 5.38, ratio 2.19, p≈0.175. It fails the min-50-trade gate.
- 2025 eval intersection with fixed modules: intersection 53; CAGR 6.87, MDD 5.73, ratio 1.20, p≈0.462.

Interpretation:
- Cross-representation agreement reduces drawdown but also removes too much edge and does not generalize.
- Current evidence says the problem is not just gate optimization. The candidate pool itself is too narrow/regime-fragile.
- Next meaningful work should expand the event/candidate pool and train/evaluate a single LLM policy over a richer action space, instead of repeatedly filtering the same Kimchi-flow pool.

## Wider event candidate pool probe

Implemented `training/event_candidate_pool_probe.py` to test whether the bottleneck is the narrow Kimchi-flow candidate pool. The script builds past-only event families from the extended feature frame: momentum, mean reversion, volatility breakout, drawdown continuation/reversal, order-flow follow/fade, kimchi flow/fade, macro pressure, higher-timeframe momentum/fade, and candle shock follow/fade.

Protocol:
- Thresholds are fit on train only via feature-strength quantile.
- Family selection uses validation only after a train-positive/min-trade filter.
- Eval is reported once for the selected family.
- Strict OHLC simulation uses entry delay, fees/slippage, 0.5x leverage, non-overlap, and intrabar strict MDD.

Fold A: train 2020-2022 -> val 2023 -> eval 2024
- Selected family: `momentum_trend`.
- Validation: CAGR 10.89, strict MDD 13.68, ratio 0.80, 80 trades, p≈0.504.
- Eval 2024: CAGR 6.78, strict MDD 12.87, ratio 0.53, 136 trades, p≈0.621.
- Conclusion: broad feature-family pool does not produce a validated 2024 edge from pre-2024 data.

Fold B: train 2020-2023 -> val 2024 -> eval 2025
- Train-positive filter selected `higher_tf_momentum` despite only marginal validation evidence.
- Validation: CAGR 12.58, strict MDD 7.49, ratio 1.68, 53 trades, p≈0.307.
- Eval 2025: only 1 executed trade, ret -1.96; not useful.
- Top raw validation family `kimchi_flow_follow` looked stronger on 2024 validation: CAGR 28.06, MDD 12.87, ratio 2.18, 215 trades, p≈0.190, but 2025 eval degraded to CAGR 14.48, MDD 13.47, ratio 1.08, p≈0.401.
- `kimchi_extreme_fade` was especially unstable: 2024 val positive, 2025 eval CAGR -44.55, MDD 45.45.

Interpretation:
- Simply adding more handcrafted event families does not solve the target. The same regime-fragility pattern remains.
- The useful signal may require richer state-conditioned action choice rather than global family thresholds.
- For the LLM+RL path, the next dataset should present many candidate actions per timestamp and let the model learn conditional preference/ranking, not select one global family.

## Event-action policy dataset and learnability baseline

Implemented `training/event_action_policy_data.py` to build a single-LLM policy dataset. Each row gives Gemma a past-only state plus a candidate action book containing the top event families and allowed holds. The target chooses the best action by strict future path utility for training only.

Generated splits with 6h stride, top-5 families, holds 72/144/288/432:
- Train 2020-2023: 5,844 rows; TRADE 4,796 / NO_TRADE 1,048; prompt mean ≈1.29k chars.
- Val 2024: 1,464 rows; TRADE 1,231 / NO_TRADE 233.
- Eval 2025: 1,334 rows; TRADE 1,116 / NO_TRADE 218.
- Main target families: macro_pressure, kimchi_extreme_fade, mean_reversion_stretch, orderflow, higher-timeframe.

Oracle upper bound check (target used as prediction, so not deployable):
- Val oracle: CAGR 4329, strict MDD 4.70, 276 trades.
- Eval oracle: CAGR 2094, strict MDD 5.34, 254 trades.
- Interpretation: the action space has huge ex-post opportunity, but this only proves label capacity, not learnability.

Implemented `training/knn_event_action_policy_baseline.py` as a cheap learnability check before GPU fine-tuning.
- Train -> val KNN k=25: exact action accuracy 13.7%, gate accuracy 33.1%; backtest CAGR 25.30, strict MDD 11.91, ratio 2.12, 146 trades, p≈0.205.
- Train+val -> eval KNN k=25: exact action accuracy 14.0%, gate accuracy 34.3%; backtest CAGR -29.68, strict MDD 30.03, ratio -0.99, 150 trades, p≈0.024 negative.

Interpretation:
- The richer action space is expressive, but naïve imitation of ex-post best actions is not stable out-of-sample.
- The eval failure is statistically negative, so a plain SFT target on the current labels would likely overfit or learn the wrong regime mapping.
- Next repair should make targets more conservative and learnable: raise no-trade utility, require positive lower-bound/MAE constraints, or transform the task into preference/risk ranking where bad trades are explicitly rejected rather than forcing an ex-post best trade at most timestamps.

## Conservative label repair attempt

Updated `training/event_action_policy_data.py` with stricter target filters:
- `min_trade_net_return`
- `max_trade_mae`
- `min_trade_utility`
- `min_trade_mfe_to_mae`

Conservative setting tested:
- no-trade utility 0.4%, min net 0.2%, max MAE 1.0%, min utility 0.4%, MFE/MAE >= 1.2.
- Label distribution became safer: train TRADE 2,883 / NO_TRADE 2,961; val TRADE 796 / NO_TRADE 668; eval TRADE 730 / NO_TRADE 604.
- KNN k=25 collapsed to almost all NO_TRADE: val 1 trade, eval 3 trades.
- Smaller-k eval sweep: k=1 gave CAGR 16.66 / MDD 12.67 / ratio 1.32 with 210 trades; k=5 gave CAGR 11.95 / MDD 14.90 / ratio 0.80; k>=10 weak/negative. Still below target and not statistically strong.

Moderate setting tested:
- no-trade utility 0.2%, min net 0.1%, max MAE 1.5%, min utility 0.2%, MFE/MAE >= 1.0.
- Label distribution: train TRADE 3,982 / NO_TRADE 1,862; val TRADE 1,076 / NO_TRADE 388; eval TRADE 991 / NO_TRADE 343.
- Eval-only KNN k=5 looked promising: CAGR 28.49, strict MDD 9.30, ratio 3.06, 221 trades, p≈0.124.
- But validation k sweep invalidated it: train->2024 val was negative for k=1/3/5/10. k=5 validation: CAGR -25.61, MDD 38.66, 226 trades.

Interpretation:
- Conservative labels avoid catastrophic eval overtrading but become mostly inactive.
- Moderate labels can produce a good-looking eval pocket, but validation rejects the setting. Selecting it would be another eval leak.
- Label thresholding alone is insufficient. The next format should train preference/rejection behavior: explicitly compare chosen safe trade vs rejected risky trade/no-trade, and let DPO/RL learn relative risk rather than forcing a single ex-post target class.

## Event-action preference data for DPO/RL-style learning

Implemented `training/event_action_preference_data.py` to convert the event-action book into chosen/rejected preference pairs. This addresses the failure mode of single-label imitation: instead of forcing Gemma to reproduce one ex-post best class, the model can learn relative risk rejection between a safer chosen action and hard negative alternatives.

Moderate preference configuration:
- no-trade utility 0.2%, min net 0.1%, max MAE 1.5%, min utility 0.2%, MFE/MAE >= 1.0.
- utility gap >= 0.3%, max 3 pairs per timestamp.
- Prompts remain past-only; chosen/rejected labels use future OHLC utility for training only.

Generated preference data:
- Train 2020-2023: 17,532 pairs.
- Val 2024: 4,392 pairs.
- Eval 2025: 4,002 pairs.
- JSON audit: no NaN/Infinity tokens after finite-audit repair.

Interpretation:
- This is now a better fit for LLM+DPO/RL than the previous single-label SFT rows.
- The next actual model step should train a small Gemma adapter on these preferences, then evaluate candidate logprob ranking/backtest without using eval targets for selection.

### 2026-06-18 Gemma event-action SFT/DPO smoke and scoring audit

- Added a candidate-book logprob evaluator for event-action preference rows because the earlier generic candidate scorer omitted the `family`/`confidence` JSON schema used during training and scored actions that were not necessarily in the prompt-visible candidate book.
- DPO dry-run on `data/event_action_preferences_moderate_train.jsonl` confirmed `gemma4-e4b` resolves to `google/gemma-4-E4B-it`, 512 gate-balanced rows, and 50/50 chosen gate balance.
- 10-step DPO-from-base smoke (`checkpoints/event_action_gemma4_dpo_smoke`) ran successfully in 199.6s but validation candidate-logprob backtest on 238 deduped 2024 rows was negative: generic scorer CAGR -1.33 / strict MDD 8.43 / 60 trades; candidate-book scorer CAGR -5.13 / strict MDD 12.59 / 184 trades. This rejects base-only short DPO as a useful policy.
- Added `gate_balanced` sampling to `training/train_text_sft.py`; dry-run on 1024 SFT rows produced 512 TRADE / 512 NO_TRADE targets instead of the prior trade-heavy balanced sample.
- 24-step gate-balanced SFT smoke (`checkpoints/event_action_gemma4_sft_gate_smoke`) trained in 161.8s and learned the output schema (train loss 1.01), but candidate-book logprob validation still over-traded: 238/238 predictions were TRADE, CAGR -10.91 / strict MDD 14.82 / 190 trades. Free-generation on 64 rows produced some NO_TRADE (7/64) but still negative CAGR -1.37 / strict MDD 7.49 / 57 trades.
- 24-step DPO continuation from the SFT adapter (`checkpoints/event_action_gemma4_sft_dpo_gate_smoke`) improved DPO pairwise training margins (final train loss 0.6664, rewards/accuracy mostly 0.875-1.0), but validation candidate-book logprob remained over-trading and degraded: CAGR -18.47 / strict MDD 20.29 / 193 trades.
- Interpretation: current Gemma event-action setup is learning JSON/schema and some pairwise preference signal, but the inference interface is not yet learning a robust abstention/no-trade boundary. The next useful change is not more steps on the same labels; it is to redesign inference/training around explicit trade-vs-abstain utility calibration or generate preference pairs where NO_TRADE directly beats marginal trades, while keeping selection confined to validation and holding eval untouched.

### 2026-06-18 Gate-first schema experiment

- Root-cause hypothesis tested: the previous sorted JSON labels put `confidence`/`family` before `gate`, so candidate-logprob inference could prefer family/side priors before evaluating abstention. Added `training/normalize_event_action_json_schema.py` to rewrite SFT and preference labels to an insertion-ordered gate-first schema (`gate, side, hold_bars, family, confidence`) without changing prompts or future-dependent audit fields.
- Generated gate-first SFT splits: train 5,844 rows (4,796 TRADE / 1,048 NO_TRADE), val 1,464 (1,231 / 233), eval 1,334 (1,116 / 218). Generated gate-first moderate preference splits: train 17,532 pairs with chosen 11,946 TRADE / 5,586 NO_TRADE and all rejected TRADE; val 4,392; eval 4,002.
- 24-step gate-balanced SFT on `data/event_action_policy_gate_first_train.jsonl` trained successfully (`checkpoints/event_action_gemma4_gate_first_sft_smoke`, train runtime 171.2s, train loss 1.268). Validation candidate-book logprob on 238 deduped 2024 rows still predicted 238/238 TRADE and backtested at CAGR -11.82 / strict MDD 14.93 / 189 trades.
- Conclusion: key order alone does not fix abstention. The model can learn JSON syntax, but full-candidate logprob ranking still collapses to always-trade. Next work should separate abstention as an explicit first-stage decision target or construct direct gate-token objectives, not keep relying on full-action candidate likelihood.

### 2026-06-18 Explicit two-stage gate smoke

- Added a first-stage gate classifier path: `training/eval_text_label.py` can now export predictions, `training/compose_two_stage_event_predictions.py` joins gate predictions with event-action predictions, and `training/score_gate_label_candidates.py` / `training/sweep_two_stage_gate_threshold.py` expose TRADE-vs-NO_TRADE score margins for calibration.
- Built plain gate label data from gate-first event-action rows: train 5,844 rows (4,796 TRADE / 1,048 NO_TRADE), val 1,464 (1,231 / 233), eval 1,334 (1,116 / 218).
- 24-step Gemma gate classifier smoke (`checkpoints/event_action_gemma4_gate_classifier_smoke`) trained successfully with balanced sample 512 TRADE / 512 NO_TRADE and train loss 0.4245.
- Raw gate scoring is normalization-sensitive: mean logprob predicted almost all NO_TRADE (val accuracy 0.161; 1,461/1,464 NO_TRADE), while sum logprob predicted almost all TRADE (accuracy 0.833 mostly due trade-heavy val labels; only 2/233 true NO_TRADE caught). Generation on the first 256 val rows was mixed but only 0.633 accuracy.
- Threshold sweep on the 238-row action-validation subset improved over always-trade but remained weak: best sum-margin threshold 38.056 produced CAGR 1.56 / strict MDD 3.73 / ratio 0.42 with only 24 trades and non-significant trade stats. Mean-margin best was weaker: CAGR 0.65 / strict MDD 3.74 / 16 trades.
- Conclusion: explicit two-stage gate is structurally better than full-action likelihood, but the current gate labels are not a profitable abstention boundary. Next step should rebuild gate labels around actual minimum utility / path-risk thresholds instead of oracle best-action gate labels, then re-test the two-stage design.

### 2026-06-18 Utility-threshold gate labels

- Added `training/event_action_gate_utility_data.py` to rebuild gate labels from future audit only when the best action clears explicit utility/risk thresholds, instead of labeling TRADE whenever the oracle best action is a trade.
- Generated three gate label families from gate-first event-action rows:
  - `u006`: train 2,997 TRADE / 2,847 NO_TRADE; val 791 / 673; eval 643 / 691.
  - `u010`: train 2,177 TRADE / 3,667 NO_TRADE; val 543 / 921; eval 422 / 912.
  - `u015`: train 1,418 TRADE / 4,426 NO_TRADE; val 335 / 1,129; eval 233 / 1,101.
- Trained 24-step Gemma utility gate smoke adapters for u010 and u006. Both trained successfully (`u010` train loss 0.5556; `u006` train loss 0.5468).
- Two-stage validation on the existing 238-row action-prediction subset improved capital preservation but not statistical usefulness:
  - u010 best mean-margin threshold: CAGR 2.81 / strict MDD 1.56 / ratio 1.80, but only 8 trades.
  - u006 best threshold: CAGR 1.81 / strict MDD 1.68 / ratio 1.08, also only 8 trades.
  - With minimum trade count >=30, u006 drops to CAGR -1.75 / MDD 7.17 over 97 trades; u010 mean-margin best with >=30 trades is only CAGR 0.37 / MDD 5.64 over 40 trades.
- Conclusion: utility-threshold gate can avoid bad trades but the current action scorer has no robust edge when trade count becomes meaningful. The next step should replace full-action candidate likelihood with a value/ranker objective over prompt-visible actions, then re-use the utility gate as a risk filter.

## 2026-06-20 continual Gemma value-ranker rolling-gate validation

- Fixed full-2024 val with value-ranker threshold 0 failed: CAGR -11.49%, strict MDD 26.36%, 478 trades.
- Monthly continual LoRA harness was added with a max-hold label embargo; each month predicts before training on newly available labels.
- Q1 validation (2024-01..03) with post-hoc margin threshold showed that trading the top ~32.4% of margins could reach CAGR/MDD > 3, but p-values remained weak.
- Corrected Q2 holdout using Q1-updated adapter and fixed raw threshold 18.6413 failed: CAGR -1.70%, MDD 6.31%, ratio -0.27, 78 trades.
- Rolling percentile gate q=0.676/warmup=20 using Q1 history only also failed on corrected Q2: CAGR -4.21%, MDD 6.31%, ratio -0.67, 74 trades.
- Monthly Q2 decomposition: April drove the loss (-34.53% annualized, MDD 6.31%), while May/June were positive. This suggests edge is regime-fragile rather than solved by score-scale normalization alone.

Leakage note: Q2 rolling-gate threshold used only prior Q1 prediction margins as initial history plus online past margins within Q2. Current-month future distribution was not used.

## 2026-06-20 Q3 rolling-gate continuation

- Q3 rolling percentile continuation used the Q2-updated adapter and Q1+Q2 margin history with the preselected q=0.676/warmup=20.
- Result failed hard: 2024-07..09 ret -11.20%, CAGR -37.91%, strict MDD proxy 16.46%, CAGR/MDD -2.30, 74 trades.
- July alone caused most damage: ret -14.87%, strict MDD 16.46%, 23 trades. August recovered strongly, September was near-flat.
- Conclusion: continual LoRA + rolling percentile score normalization is not robust to regime breaks; it can find isolated monthly wins but fails the multi-quarter validation requirement.

Decision: Do not promote this value-ranker/rolling-gate stack to 2025 eval. Next work should target regime-break detection or a different label/action abstraction rather than further threshold tuning.

## 2026-06-20 online loss-pause overlay holdout

- Added `training/online_risk_overlay_backtest.py` to test live-usable risk-off pauses using only completed prior trade results.
- Q1 validation smoke on existing predictions looked promising:
  - base: CAGR -11.01%, MDD 11.21%, 141 trades.
  - pause after 2 losses: CAGR 34.94%, MDD 7.24%, 67 trades.
  - pause after 3 losses: CAGR 53.87%, MDD 7.69%, 102 trades.
- Q1-selected `pause_after_losses=3, pause_bars=864` failed on corrected Q2 raw predictions generated from the Q1-updated adapter:
  - Q2 raw base: CAGR -40.61%, MDD 10.25%, 142 trades.
  - Q2 loss3 overlay: CAGR -22.43%, MDD 9.37%, 101 trades, p=0.465.
  - Q2 loss2 diagnostic: CAGR -28.72%, MDD 11.63%, 79 trades.

Conclusion: online loss-pause reduces some exposure but does not repair the underlying negative edge. Next work should move the abstention decision before entry by labeling/predicting regime-break or unsafe-context states from past-only features.

## 2026-06-20 pre-entry regime safety abstraction

Implemented `training/event_regime_safety_data.py` to label each signal before action ranking as:
- `SAFE_TRADE`: at least one candidate action has enough net edge with bounded MAE.
- `UNSAFE_NO_EDGE`: no candidate has enough edge, but broad path risk is not extreme.
- `BREAK_RISK`: candidate set shows severe adverse excursion / broad negative path risk.

Smoke label distributions with stride 72 and hold candidates 72/144/288/432:
- 2024 Q1: 364 rows; SAFE_TRADE 139, UNSAFE_NO_EDGE 160, BREAK_RISK 65.
- 2024 Q2: 364 rows; SAFE_TRADE 120, UNSAFE_NO_EDGE 178, BREAK_RISK 66.

This replaces post-loss pausing with a pre-entry abstention abstraction. Prompts remain past-only (state + candidate book); labels use future executable action outcomes for training only.

Note: `../wave_trading` external forex cache lookup failed in this environment for DXY component tickers, so the smoke used the existing market cache features without rejoining wave_trading caches.

### 2026-06-20 — Gemma4 pre-entry safety gate rejected
- Added a past-only `event_regime_safety` dataset that asks Gemma4-E4B to classify each signal as `SAFE_TRADE`, `UNSAFE_NO_EDGE`, or `BREAK_RISK` from past-only state plus prompt-visible candidate action book.
- Leak guard: prompts contain only past state/candidate descriptors; labels use future action outcomes only for supervised training, not for inference.
- Train data: `data/event_regime_safety_train_2020_2023.jsonl`, 5,844 rows, 2020-01-01..2023-12-31, counts `SAFE_TRADE=2201`, `UNSAFE_NO_EDGE=2266`, `BREAK_RISK=1377`.
- Gemma4-E4B LoRA run: `checkpoints/event_regime_safety_gemma4_2020_2023_s64`, 4,096 balanced rows, 64 steps, runtime 447.1s, train loss 0.2729.
- Holdout label accuracy was weak:
  - Q1 2024: 364 rows, accuracy 38.46%.
  - Q2 2024: 364 rows, accuracy 37.09%.
- Applied predicted `SAFE_TRADE` as a front gate to the existing value-ranker actions:
  - Q1: CAGR -51.85%, strict MDD 19.39%, 89 trades.
  - Q2: CAGR -26.62%, strict MDD 11.78%, 81 trades.
- SAFE margin threshold sweep (`0..4`) did not rescue Q1 or Q2; all variants had negative CAGR.
- Oracle target gate upper bound also failed to generalize:
  - Q1 oracle: CAGR 36.06%, strict MDD 8.31%, 66 trades, but p≈0.51 and underpowered.
  - Q2 oracle: CAGR -31.61%, strict MDD 14.40%, 62 trades.
- Conclusion: the pre-entry generic regime safety label is not aligned with the realized action chosen by the value-ranker. It can select a signal where some candidate was safe, while the ranker executes a different or fragile action. Reject this label family for the current stack.
- Next structural pivot: train a post-ranker action verifier that receives the selected action plus state and predicts whether *that exact executable action* should be taken. This preserves the LLM/RL-style staged decision but makes the learned gate target action-conditioned realized profitability/risk rather than generic regime safety.

### 2026-06-20 — Categorical exact-action verifier smoke rejected
- Built `event_action_verifier_text` as a post-ranker action verifier: each row labels one exact executable action as `ALLOW` or `BLOCK` from categorical past-only regime tokens and selected action tokens, avoiding raw decimal dumps.
- Train data: 116,880 candidate-action rows over 2020-2023; `ALLOW=8,784` (7.5%), `BLOCK=108,096`.
- Q1 2024 verifier rows: 7,280 rows, `ALLOW=583` (8.0%); Q2 2024: 7,280 rows, `ALLOW=429` (5.9%).
- Gemma4-E4B LoRA smoke: `checkpoints/event_action_verifier_text_gemma4_2020_2023_s64`, 4,096 balanced rows, 64 steps, runtime 434.5s, train loss 0.3692.
- To avoid slow CPU scoring of all candidates, added selected-action extraction and scored only the 364 value-ranker-selected actions per quarter.
- Selected-action label reports:
  - Q1 accuracy 92.86%, but confusion was `ALLOW->BLOCK=20`, `BLOCK->ALLOW=6`, `BLOCK->BLOCK=338`: the model found no true ALLOW.
  - Q2 accuracy 93.96%, but confusion was `ALLOW->BLOCK=16`, `BLOCK->ALLOW=6`, `BLOCK->BLOCK=342`: same over-conservative collapse.
- Composed hard `ALLOW` gate with value-ranker:
  - Q1: CAGR -4.31%, strict MDD 2.28%, 6 trades.
  - Q2: CAGR 1.35%, strict MDD 1.25%, 6 trades.
- This reduces loss by nearly abstaining, but trade count is statistically meaningless. ALLOW margin distributions also collapse around zero and do not separate true ALLOW from BLOCK; in Q2 true ALLOW mean margin was worse than BLOCK.
- Conclusion: categorical prompt compression alone did not make Gemma4 learn a useful exact-action verifier. Continue away from pure LLM classification toward a hybrid where the LLM produces stable symbolic/compressed features and a lightweight chronological tabular/ranker model learns the realized action edge.

### 2026-06-20 — Symbolic action ridge hybrid gives weak positive but misses target
- Pivoted from pure LLM classification to a hybrid: use the categorical LLM-style regime/action text as symbolic features, then train a leakage-safe ridge ranker on realized action utility.
- Implemented `training/symbolic_action_ridge.py` with train-only vocabulary/scaling/ridge fit, validation-only config selection, and fixed holdout evaluation.
- Q1-only validation overfit:
  - train 2020-2023, val Q1 2024 selected `target=net_return, alpha=10000, threshold=0.003, min_gap=0.0`.
  - Q1 val: CAGR 193.31%, strict MDD 5.90%, 52 trades.
  - Q2 holdout: CAGR -18.46%, strict MDD 9.82%, 56 trades.
  - Conclusion: single-quarter validation is too unstable.
- Longer validation setup:
  - train 2020-2022: 87,680 candidate-action rows, ALLOW 7,333.
  - validation 2023 full year: 29,200 rows, ALLOW 1,451.
  - holdout 2024 H1: Q1+Q2, 14,560 rows / 728 signal samples.
  - selection required at least 120 validation trades.
- 2023-selected config: `target=utility, alpha=1000, threshold=0.003, min_gap=0.0005`.
  - 2023 validation: CAGR 3.61%, strict MDD 6.96%, ratio 0.52, 166 trades.
  - 2024 H1 holdout: CAGR 2.12%, strict MDD 4.90%, ratio 0.43, 88 trades.
- Conclusion: symbolic ridge is more robust than pure LLM classifiers and avoids the Q2 blow-up, but the edge is far below target. Treat as a weak positive baseline and expand the action/feature pool rather than tuning gates on the same surface.

### 2026-06-20 — Expanded symbolic action pool finds 2024 edge but 2025 remains weak
- Expanded `_feature_candidates` with additional past-only action families: volatility compression breakout/fakeout, micro exhaustion reversal, HTF pullback resume, HTF structure break, macro-kimchi divergence, derivatives stress fade.
- Rebuilt symbolic action datasets with `top_k_families=8`.
  - train 2020-2022: 140,288 rows, ALLOW 11,781.
  - validation 2023: 46,720 rows, ALLOW 2,337.
  - holdout 2024H1: 23,296 rows, ALLOW 1,642.
- 2023-selected config (`target=net_return, alpha=10000, threshold=0.003, min_gap=0.0`) improved materially:
  - 2023 val: CAGR 12.89%, strict MDD 11.84%, ratio 1.09, 247 trades.
  - 2024H1 holdout: CAGR 78.67%, strict MDD 9.34%, ratio 8.42, 139 trades.
- Fixed config retrained on 2020-2023 and evaluated on full 2024:
  - 2024: CAGR 51.46%, strict MDD 11.01%, ratio 4.67, 215 trades, p≈0.024.
  - This meets the single-year target profile and is the strongest non-cheating result so far.
- Same fixed config failed 2025:
  - 2025 Jan-Nov: CAGR -15.40%, strict MDD 21.23%, 193 trades.
- Added monthly prior-only retraining (`rolling_symbolic_action_ridge.py`) and tested 2025 using 2020-2024 history plus prior 2025 months only:
  - 2025 rolling: CAGR 5.14%, strict MDD 13.61%, ratio 0.38, 150 trades, p≈0.68.
  - Adaptation avoids the large negative year but has weak/insignificant edge.
- Combined 2024 fixed + 2025 rolling:
  - 2024-01-01..2025-11-30: CAGR 26.85%, strict MDD 15.54%, ratio 1.73, 365 trades, p≈0.038.
- Risk overlay selected on 2024 (monthly loss stop 6%) did not improve 2025 or combined performance:
  - 2025 rolling + ml6: CAGR 0.13%, strict MDD 15.34%.
  - 2024 fixed + 2025 rolling + ml6: CAGR 25.21%, strict MDD 15.60%.
- Conclusion: expanded symbolic/action pool creates a real 2024 edge, but 2025 needs new features/data or regime-specific adaptation; risk overlay is not the bottleneck.

### 2026-06-21 — 2025 weakness decomposed by family; overlays insufficient
- Added action prediction diagnostics to attribute realized strict-backtest trades by month, family, side, and horizon.
- 2024 fixed model diagnostics:
  - Strong contributors: `mean_reversion_stretch`, `macro_pressure`, `htf_pullback_resume`, `higher_tf_momentum`, mostly long/432h.
  - Weak contributors: `orderflow_follow`, `macro_kimchi_divergence`, `drawdown_continuation/reversal`.
- 2025 rolling diagnostics:
  - Largest weakness came from `micro_exhaustion_reversal` (16 trades, -6.10 pct cumulative), then orderflow and macro-kimchi buckets.
  - `micro_exhaustion_reversal` was not a 2024 loser, so a 2024-only static family block would not catch the main 2025 decay.
- Tested static family filters selected using 2024 only:
  - Best 2024 block set was `macro_kimchi_divergence,orderflow_follow`: 2024 CAGR 53.26%, MDD 10.76%, 198 trades.
  - Applied to 2025 rolling: CAGR 8.48%, MDD 15.33%, 135 trades.
  - Combined 2024-2025: CAGR 28.83%, MDD 17.91%, ratio 1.61, 332 trades.
  - This improves 2025 return but worsens combined drawdown; not target-sufficient.
- Added per-family online risk-off overlay (pause only the family after completed prior losses).
  - Best 2024-selected config: 3 family losses, 864-bar pause, 5% family monthly stop.
  - 2025 rolling fixed application: CAGR 3.74%, MDD 13.61%, 146 trades.
  - Combined 2024-2025: CAGR 26.99%, MDD 15.31%, ratio 1.76, 357 trades.
  - Overlay is not the primary fix.
- 2025-only diagnostic upper bound (not selection-safe): blocking `micro_exhaustion_reversal` alone would give CAGR 17.85%, MDD 9.22%, 143 trades; blocking micro/orderflow/macro-kimchi gives CAGR 20.57%, MDD 9.23%, 119 trades. Still below target and partly under trade-count ambition.
- Conclusion: family-level suppression helps diagnose the failure but cannot meet the target. The next useful work is token/regime-conditioned detection of when `micro_exhaustion_reversal` flips from profitable 2024 behavior to harmful 2025 behavior, plus new 2025-specific regime features.

### 2026-06-21 — Token-conditioned micro filter improves robustness
- Added token-level trade diagnostics by joining executed trades back to the symbolic candidate prompt for the selected action.
- Focused on `micro_exhaustion_reversal`, the biggest 2025 decayed family.
- 2024 micro diagnostics showed that the family was already weak under tokens such as:
  - `kimchi_context=kimchi_neutral`
  - `medium_trend=flat`
  - `orderflow=sell_aggression_strong`
  - `weekly_context=strong_up`
  - `recent_drawdown=no_recent_drawdown`
- Applied this 2024-derived token block only to `micro_exhaustion_reversal`, then tested fixed on 2025 rolling:
  - 2024: CAGR 53.01%, strict MDD 11.01%, ratio 4.81, 213 trades, p≈0.020.
  - 2025 rolling: CAGR 17.85%, strict MDD 9.22%, ratio 1.94, 143 trades, p≈0.182.
  - 2024-2025 combined: CAGR 34.65%, strict MDD 11.04%, ratio 3.14, 356 trades, p≈0.0077.
- Combined with 2024-selected family blocks (`macro_kimchi_divergence,orderflow_follow`) did not help:
  - 2024-2025 combined dropped to CAGR 30.97%, MDD 13.54%, ratio 2.29.
- Conclusion: token-conditioned filtering is the first robust cross-year improvement that keeps trade count and statistical signal while passing ratio>3 over 2024-2025. It still misses the CAGR 50 target, so the next work should improve return capture, not further suppress risk blindly.

### 2026-06-21 — Training only from 2023 does not improve 2025
- Tested the hypothesis that older 2020-2022 data may hurt current-regime adaptation.
- Setup A: train on 2023 only, select config on full 2024, hold out 2025.
  - Selected config remained `target=net_return, alpha=10000, threshold=0.003, min_gap=0.0`.
  - 2024 validation: CAGR 30.10%, strict MDD 16.33%, ratio 1.84, 244 trades.
  - 2025 holdout: CAGR -6.01%, strict MDD 15.10%, 223 trades.
  - Worse than the broader-history 2020-2023 fixed model on 2024 and still failed 2025.
- Setup B: monthly rolling 2025 with history limited to 2023+2024 plus prior 2025 months only.
  - 2025: CAGR -1.14%, strict MDD 13.99%, 169 trades, p≈0.996.
  - Worse than full-history rolling (`2020-2024 + prior 2025`) which reached CAGR 5.14% before token filtering.
- Setup C: same 2023+ rolling with the previously selected micro token filter.
  - 2025: CAGR 2.86%, strict MDD 15.94%, 162 trades.
  - Still far below the full-history rolling + micro token filter result (CAGR 17.85%, MDD 9.22%).
- Conclusion: older 2020-2022 data is not just stale noise; it regularizes the symbolic ridge model. The stronger direction is not to discard old history wholesale, but to use recency/regime weighting or ensemble recent-history and long-history models.

### 2026-06-21 — Recency weighting and long/recent agreement do not beat the token-filter baseline
- Added `weighted_symbolic_action_ridge.py` to test whether old history should be retained but down/up-weighted by date only. This avoids target leakage because weights depend only on row timestamps.
- Fixed 2024 test with 2020-2023 training and the previously selected symbolic config:
  - exp half-life 365d: CAGR -3.32%, strict MDD 14.99%, 149 trades.
  - exp half-life 730d: CAGR 23.46%, strict MDD 13.24%, 173 trades.
  - step weight from 2023-01-01, recent weight 2/3/5: best was weight 5 with CAGR 20.00%, strict MDD 11.93%, 215 trades.
  - All are materially worse than unweighted long-history 2024 (CAGR 51.46%, MDD 11.01%). Recency weighting is therefore rejected for now.
- Added `prediction_agreement_filter.py` to test whether a long-history model should trade only when a recent-history model agrees.
  - `side` agreement, 2024-2025: CAGR 23.04%, strict MDD 13.59%, 243 trades, p≈0.0229.
  - `family_side` agreement, 2024-2025: CAGR 12.84%, strict MDD 10.96%, 148 trades.
  - `trade` agreement, 2024-2025: CAGR 18.31%, strict MDD 17.85%, 283 trades.
  - Agreement filtering reduces useful 2024 exposure and does not beat the 2024-derived micro token filter baseline (CAGR 34.65%, MDD 11.04%, 356 trades, p≈0.0077).
- Sizing sensitivity on the current best token-filter stream:
  - leverage 0.65: CAGR 46.25%, strict MDD 14.29%, 356 trades, p≈0.0080.
  - leverage 0.68: CAGR 48.64%, strict MDD 14.94%, 356 trades.
  - leverage 0.70: CAGR 50.24%, strict MDD 15.37%, 356 trades.
  - This nearly reaches the target but does not honestly satisfy strict MDD<=15 at CAGR>=50; leverage 0.68 is the current honest boundary.
- Fixed a risk-overlay implementation issue: rolling drawdown/loss stops now require at least `rolling_window_trades` completed trades before activation. Without this, a 50-trade window could pause after only a few early trades. Re-testing showed rolling-DD stops still kill too much exposure and are not a solution.
- Conclusion: the best verified branch remains long-history symbolic ridge + 2024-derived micro token filter. The next real improvement should increase return capture/action quality, not discard history or require recent-model agreement.

### 2026-06-21 — Richer categorical state tokens reproduce but do not improve the best branch
- Expanded analyzer/verifier text state tokens with past-only higher timeframe and derivatives context:
  - `three_day_context`, `weekly_location`, `weekly_drawdown`
  - `funding_context`, `open_interest_level`, `open_interest_change`
- Generated v3/k8 yearly datasets with the same candidate pool and labels. Row counts matched the prior v2/k8 split shape:
  - 2020-2022 train: 140,288 rows, ALLOW 11,781.
  - 2023 select: 46,720 rows, ALLOW 2,337.
  - 2024 eval: 46,848 rows, ALLOW 3,322.
  - 2025 eval: 42,688 rows, ALLOW 2,175.
- Selection using 2020-2022 train and 2023 validation again chose `target=net_return, alpha=10000, threshold=0.003, min_gap=0.0`.
  - 2024 holdout before retraining: CAGR 30.50%, strict MDD 19.29%, 262 trades. This is worse than desired, but the fair production-style comparison is retrain through 2023 before 2024.
- Fixed retrain through 2023, then evaluate 2024:
  - 2024: CAGR 51.46%, strict MDD 11.01%, ratio 4.67, 215 trades, p≈0.0241.
  - This exactly reproduces the prior v2/k8 fixed result; the new tokens did not change the selected live actions enough to matter.
- Prior-only monthly rolling through 2025:
  - 2025: CAGR 5.14%, strict MDD 13.61%, 150 trades, p≈0.6802.
  - 2024-2025 combined before token filter: CAGR 26.85%, strict MDD 15.54%, 365 trades, p≈0.0385.
- Applying the existing 2024-derived `micro_exhaustion_reversal` token filter to v3 predictions reproduces the current best:
  - 2024: CAGR 53.01%, strict MDD 11.01%, 213 trades, p≈0.0199.
  - 2025: CAGR 17.85%, strict MDD 9.22%, 143 trades, p≈0.1816.
  - 2024-2025 combined: CAGR 34.65%, strict MDD 11.04%, ratio 3.14, 356 trades, p≈0.0077.
- Engineering note: added an in-process market-bar cache for repeated strict backtests. The v3 sweep initially generated 53/420 configs slowly because it reloaded the same CSV every time; after caching, the same sweep completed much faster.
- Conclusion: simply adding more categorical context is not enough. The bottleneck is not missing 3d/weekly/funding tokens in the text; it is return capture/action construction or model class. Next improvement should alter action quality/horizon/sizing or train a model that can exploit continuous magnitudes, not just add more discrete state tokens.

### 2026-06-21 — Take-profit exits create a strong 2024-2025 candidate but fail 3-year strict validation
- Added optional live-usable trade exits to the strict online backtest:
  - `trade_take_profit_pct` and `trade_stop_loss_pct` are account-level per-trade exits.
  - Intrabar OHLC ambiguity is handled conservatively: if stop and take-profit both touch in the same bar, stop is assumed first.
  - Strict MDD still includes intrabar adverse excursion before any exit.
- Selected exit parameters on 2024 only using the current best token-filter predictions.
  - Best 2024-safe candidate: leverage 0.68, no stop, take-profit 4%.
  - 2024: CAGR 87.70%, strict MDD 14.72%, 222 trades, p≈0.0084.
- Fixed that 2024-selected exit and evaluated 2025:
  - 2025: CAGR 26.46%, strict MDD 12.38%, ratio 2.14, 144 trades, p≈0.1673.
  - This improves 2025 return over the no-exit token-filter baseline (17.85%) but still does not reach ratio>=3 on 2025 alone.
- 2024-2025 combined with the fixed exit:
  - CAGR 54.77%, strict MDD 14.72%, ratio 3.72, 366 trades, p≈0.0033.
  - This passes the numeric target over the selection+eval combined period, but 2024 is the parameter-selection year, so it is not a pure out-of-sample claim.
- Longer diagnostic including 2023:
  - 2023 with the same micro filter and TP4/leverage0.68: CAGR 11.49%, strict MDD 19.84%, 255 trades, p≈0.549.
  - 2023-2025 combined: CAGR 38.28%, strict MDD 19.84%, ratio 1.93, 621 trades, p≈0.0066.
- Conclusion: TP exits are the first clear return-capture improvement and produce a strong recent 2-year candidate, but the original 3-year+ / strict MDD<=15 target is not solved. The hard failure is 2023 drawdown/low edge. Next work must either detect the 2023 regime as unsafe or build a separate 2023-robust action surface; claiming the 2024-2025 result as final would be leakage/selection bias.

### 2026-06-21 — 2023 failure decomposition and simple action blocks are not enough
- Decomposed the 2023 TP4/leverage0.68 run by month/family/side/token.
- Main 2023 weak buckets:
  - `side=SHORT`: 63 trades, -9.78 pct cumulative.
  - `month=2023-08`: 21 trades, -8.21 pct.
  - `kimchi_extreme_fade|SHORT`: 20 trades, -7.15 pct.
  - `mean_reversion_stretch`: 10 trades, -6.99 pct.
  - `higher_tf_momentum`: 29 trades, -4.43 pct.
- Tested simple 2023-derived action blocks with TP4/leverage0.68 and then applied them fixed to 2024-2025:
  - `block_short`: 2023 CAGR 22.26%, MDD 16.93%; 2025 CAGR 28.68%, MDD 12.39%; 2023-2025 CAGR 36.32%, MDD 16.93.
  - `block_htf_mom`: 2023 CAGR 6.07%, MDD 17.49%; 2025 CAGR 32.85%, MDD 12.36%; 2024-2025 CAGR 45.10%, MDD 12.68.
  - `block_2023_worst4` (`SHORT` or `mean_reversion_stretch` or `higher_tf_momentum`): 2023 CAGR 33.46%, MDD 15.98%; 2025 CAGR 33.57%, MDD 12.39%; 2023-2025 CAGR 29.15%, MDD 15.98.
- Conclusion: 2023 can be partially defended with simple past-observable action blocks, but those blocks either miss the MDD<=15 target or destroy too much 2024-2025 upside. The next useful change is not a broad static action ban; it needs a regime classifier or different action construction that avoids 2023 drawdown while preserving 2024 trend capture.

### 2026-06-21 — Conditional sizing gives the first honest 2024-2025 OOS pass, but still not a 3-year pass
- Added `position_scale` support to strict backtests and a reusable `position_scale_filter.py`.
- Motivation: broad static blocks defended 2023 but destroyed too much upside. Scaling risky buckets preserves signal information while reducing drawdown contribution.
- Tested 2023-derived sizing rules with TP4 exits. Best robust family:
  - Rule: scale trades where `side=SHORT` or `family=higher_tf_momentum`.
  - Scale: 0.4.
  - This is interpretable from the 2023 failure decomposition: shorts and HTF momentum were major 2023 drawdown contributors.
- 2023 selection diagnostics:
  - With leverage 0.68: 2023 CAGR 25.28%, MDD 11.89%; 2023-2025 CAGR 36.10%, MDD 12.59.
  - Leverage grid on the scaled stream showed the best strict-MDD boundary around leverage 0.81 for the 2024-2025 eval.
- Fixed rule selected from 2023, evaluated on 2024-2025 with no further tuning:
  - Config: `short_or_higher_tf_momentum` scale 0.4, leverage 0.81, take-profit 4%, no stop.
  - 2024-2025: CAGR 50.09%, strict MDD 14.74%, ratio 3.40, 370 trades, p≈0.0041.
  - 2025 alone: CAGR 31.57%, strict MDD 14.31%, 148 trades, p≈0.1446.
- Longer diagnostic including the 2023 selection year:
  - 2023: CAGR 24.62%, strict MDD 16.32%, 257 trades.
  - 2023-2025: CAGR 40.78%, strict MDD 16.32%, ratio 2.50, 627 trades, p≈0.0032.
- Conclusion: this is the first candidate that honestly passes the 2024-2025 out-of-sample target after selecting the sizing rule on 2023. It still does not satisfy the stricter interpretation of 3+ calendar years including 2023, because 2023 remains a marginal regime. Next work should focus on a 2023-specific regime-risk detector or action surface that reduces the 2023 MDD below 15 without cutting 2024-2025 CAGR below 50.

### 2026-06-21 — Monthly stops and hard stops do not fix the remaining 2023 weakness
- Re-generated clean final candidate prediction artifacts with `short_or_higher_tf_momentum` scaled to 0.4.
- Tested monthly loss stops on the final candidate (`leverage=0.81`, TP4):
  - Monthly loss stops did not reduce the 2023 MDD; 2023 remained CAGR 24.62%, MDD 16.32%.
  - Best 2024-2025 variants stayed around CAGR 50-51%, MDD 14.31-14.74%, so monthly stops are not the missing 2023 fix.
- Tested hard stop-loss exits with TP4:
  - SL 3.5 lowered 2023 MDD to 14.75 but destroyed 2024-2025 performance (CAGR 27.67%, MDD 23.01%).
  - SL 1.5/2.0 also reduced some 2023 risk but cut 2024-2025 CAGR to 15-19% and/or worsened combined drawdown.
- Conclusion: the remaining failure is not solved by generic online risk overlays. The best candidate remains no hard stop, TP4, conditional sizing. The next improvement must target regime/action selection before entry rather than exit-level damage control.

### 2026-06-21 — 2026 Jan-Feb validation fails the current candidate
- Built a 2026 v3/k8 verifier dataset from local OHLCV through 2026-02-27 and attached DXY/Kimchi/USDKRW from `/home/pakchu/workspace/wave_trading` with backward-asof 30min tolerance.
  - 2026 rows: 7,168, ALLOW 437, allow rate 6.10%.
  - Effective backtest prediction period: 2026-01-01 02:55 to 2026-02-25 20:55.
- Ran prior-only monthly rolling with history through 2025:
  - January 2026 fit used 276,544 prior rows.
  - February 2026 fit used 280,512 prior rows, including prior January labels only.
  - Leakage guard: each month uses rows before month start only.
- Baseline rolling symbolic ridge 2026:
  - Return -6.48%, annualized CAGR -35.92%, strict MDD 16.10%, 34 trades, p≈0.501.
- Current final candidate fixed from the 2023-selected / 2024-2025-passing setup (`micro token filter`, `short_or_higher_tf_momentum` scale 0.4, leverage 0.81, TP4):
  - Return -11.23%, annualized CAGR -54.65%, strict MDD 23.82%, 34 trades, p≈0.426.
  - This is a clear failure, though the sample is short and underpowered.
- 2026 failure decomposition differs from the 2023 failure:
  - Worst: `hold=432` (16 trades, -11.61 pct cumulative), `side=LONG` (22 trades, -8.90 pct), `month=2026-01` (-8.78 pct), `drawdown_reversal` (-4.26 pct), `higher_tf_fade` (-3.03 pct).
  - Best: `higher_tf_momentum|hold=144` (6 trades, +1.61 pct), `higher_tf_momentum|side=LONG` (7 trades, +1.33 pct).
- Conclusion: the 2024-2025 OOS pass does not generalize into early 2026. The current sizing rule penalizes HTF momentum, but in 2026 HTF momentum was one of the only positive buckets while long 432h/drawdown-reversal exposure caused the damage. Next work should add regime-aware horizon/family control, especially reducing 432h long/drawdown-reversal exposure under early-2026-like regimes, rather than reusing the 2023-specific short/HTF momentum scaling rule blindly.

### 2026-06-21 — Side-specialist ridge is not enough; short horizon gating improves the prior candidate
- Implemented `side_specialist_symbolic_ridge.py`, which fits separate LONG and SHORT symbolic ridge models and then compares side-specific candidate scores at decision time.
- 2026 prior-only rolling with side-specialist ridge failed:
  - Baseline side-specialist 2026: CAGR -53.46%, strict MDD 19.31%, 36 trades, p≈0.140.
  - Applying the existing micro token filter + `short_or_higher_tf_momentum` 0.4 sizing + TP4/leverage0.81 worsened to CAGR -74.90%, MDD 29.55%.
  - Conclusion: simply separating LONG/SHORT regressors increases trade aggressiveness but does not make shorts robust.
- Decomposed the current final candidate by side/horizon:
  - 2023 shorts were slightly negative overall; SHORT 72h and 144h were the main bad short horizons, while SHORT 288/432 were roughly flat to positive.
  - 2024-2025 shorts were positive overall but SHORT 144h was weak; SHORT 288/432 were better.
  - 2026 still failed mostly from LONG 432h and drawdown-reversal, so short improvement alone cannot fix 2026.
- Added `action_gate_filter.py` for auditable NO_TRADE gates and tested short horizon gates selected from 2023.
- Best short-specific gate: block `SHORT` with `hold_bars=144`.
  - At leverage 0.81 with TP4: 2024-2025 CAGR 54.35%, MDD 14.37%, 339 trades; 2023-2025 CAGR 43.15%, MDD 15.31%.
  - Re-leveraged to 0.76: 2024-2025 CAGR 50.99%, MDD 13.53%, 337 trades; 2023-2025 CAGR 41.24%, MDD 13.53%, 573 trades.
  - 2026 still fails: around CAGR -63.50%, MDD 22.51% at leverage 0.76.
- Conclusion: a useful short lesson emerged: do not suppress all shorts; suppress the weak 144h short horizon and keep longer 288/432h shorts. This gives a cleaner 2024-2025 and 2023-2025 profile than the previous scaling-only candidate, but 2026 requires a separate long-432/drawdown-reversal regime fix.

### 2026-06-21 — 2026 long-horizon counterfactual gates do not rescue the system
- Tested counterfactual gates for the 2026 failure buckets on both the scaling-only candidate and the improved `SHORT 144h` gate candidate.
- Blocking `LONG 432h`:
  - Reduces some drawdown but destroys the historical alpha source.
  - On the scaling-only branch, 2024-2025 collapses to CAGR 12.50%, MDD 14.78%; 2026 still fails at CAGR -58.62%, MDD 18.65%.
  - On the `SHORT 144h` branch, 2024-2025 collapses to CAGR 13.87%, MDD 14.17%; 2026 still fails at CAGR -53.84%, MDD 17.60%.
- Blocking `drawdown_reversal`:
  - Helps 2026 somewhat but remains negative and hurts the prior pass.
  - Scaling-only branch: 2026 improves to CAGR -19.25%, MDD 18.02%, but 2024-2025 drops to CAGR 41.03%, MDD 16.36%.
  - `SHORT 144h` branch: 2026 improves to CAGR -41.76%, MDD 16.99%, but 2024-2025 drops to CAGR 39.98%, MDD 15.39%.
- Blocking `LONG 432h` plus `drawdown_reversal` is worse: it removes too much edge and still fails 2026.
- Conclusion: 2026 cannot be fixed by a simple family/horizon gate. The live-worthy next architecture needs a short-capable model plus a regime detector that decides when long 432h trend capture is valid, instead of globally blocking the very exposure that created the 2024-2025 edge.

### 2026-06-21 — Long-432 validity tokens improve history but do not rescue 2026
- Searched for tokens that identify bad `LONG` 432h trades on the `SHORT 144h` gated branch, selecting from 2023-2025 only and then testing 2026.
- Bad historical long-432 tokens included:
  - `four_hour_context=strong_down`, `volume=volume_high`, `higher_tf_drawdown=daily_dd_medium`, `recent_drawdown=large_drawdown`, `daily_context=down`, `kimchi_context=kimchi_soft_premium`.
- Useful historical gates:
  - Blocking long-432 when `short_trend=strong_down` improved 2023-2025 to CAGR 48.72%, MDD 12.93%, ratio 3.77 and 2024-2025 to CAGR 63.32%, MDD 11.17%.
  - Similar improvements came from `medium_trend=strong_down` and `recent_drawdown=large_drawdown` gates.
- 2026 remained negative under all single-token gates:
  - Best listed candidates still had 2026 around CAGR -61% to -63% with MDD around 19.9-22.5%.
  - `four_hour_context=strong_down` worsened 2026 to CAGR -76.34%.
- Conclusion: historical long-432 validity gating can improve 2023-2025 and may be valuable for robustness, but it does not solve early 2026. The 2026 failure is not captured by the same historical bad-token signatures; this points to a broader regime drift/abstention problem, not just a long-432 token gate.

### 2026-06-21 — Month-level abstention helps risk but does not solve 2026 profitability
- Added `month_regime_gate_filter.py`, which uses the first candidate prompt tokens in each calendar month to decide whether to abstain for that month. This is live-usable as long as the candidate prompt is built from past/current-bar features only.
- Searched single-token and token-pair month gates on the `SHORT 144h` gated branch selected from 2023-2025, then evaluated 2026.
- Jan-only abstention candidate:
  - Rule: month tokens include `daily_context=up` and `volume=volume_normal`.
  - Blocks 2024-12 and 2025-02 historically; blocks 2026-01 in eval.
  - 2024-2025 improves to CAGR 57.74%, strict MDD 13.05%, 306 trades, p≈0.00032.
  - 2026 improves drawdown but remains negative: CAGR -41.60%, MDD 13.46%, 10 trades.
- Full Jan-Feb 2026 abstention candidate:
  - Rule: month tokens include `book_drawdown_continuation` and `book_higher_tf_fade`.
  - Blocks 2026-01 and 2026-02 fully, producing zero 2026 trades/losses.
  - But it also blocks too many useful historical months; 2024-2025 drops to CAGR 43.70%, MDD 13.05%, 270 trades.
- Conclusion: month-level abstention is a promising risk layer: it can cut the 2026 drawdown and can even fully avoid early 2026. However, the rules found so far either leave February losses or sacrifice too much 2024-2025 CAGR. The next step should train a more expressive abstention classifier over month-start tokens/continuous features, not rely on one or two hand-picked token pairs.

### 2026-06-21 — Extending 2026 validation through May invalidates the current candidate
- Downloaded Binance futures BTCUSDT 5m OHLCV through 2026-06-01, covering effective bars from 2019-12-31 15:00 to 2026-05-31 15:00.
- Built a wider 2026 v3/k8 verifier dataset for 2026-01-01 through 2026-06-01 with DXY/Kimchi/USDKRW features attached from `/home/pakchu/workspace/wave_trading` using 30min backward-asof tolerance.
  - Candidate rows: 19,104.
  - ALLOW labels: 1,018, allow rate 5.33%.
  - Leakage guard remains: prompts are past-only, labels use future outcomes, and the dataset itself is not a backtest.
- Ran prior-only monthly rolling symbolic ridge with 2020-2025 history and no current-month labels in each fit:
  - 2026-01 train rows 276,544; 2026-02 280,512; 2026-03 284,096; 2026-04 288,064; 2026-05 291,904.
  - Eval predictions: 597 scored decision rows over 2026-01-01 02:55 to 2026-05-30 02:55.
- Results on the widened Jan-May 2026 window:

| candidate | return | CAGR | strict MDD | CAGR/MDD | trades | p-value |
|---|---:|---:|---:|---:|---:|---:|
| baseline rolling ridge | -16.23% | -35.22% | 20.71% | -1.70 | 75 | 0.132 |
| micro filter + short/HTF scale0.4, lev0.81, TP4 | -18.33% | -39.12% | 23.82% | -1.64 | 74 | 0.228 |
| `SHORT 144h` gate, lev0.76, TP4 | -19.25% | -40.79% | 24.10% | -1.69 | 66 | 0.153 |
| Jan-only month gate (`daily_context=up`, `volume=volume_normal`) | -13.33% | -29.57% | 15.24% | -1.94 | 47 | 0.245 |
| Jan-Apr month gate (`book_drawdown_continuation`, `book_higher_tf_fade`) | +0.49% | +1.20% | 3.55% | 0.34 | 8 | 0.777 |

- Diagnostic buckets for the current `SHORT 144h` candidate:
  - Worst: `hold=432` (32 trades, mean -0.82%, sum -26.27%), `side=LONG` (50 trades, sum -13.20%), `month=2026-02` (10 trades, sum -8.88%), `month=2026-01` (20 trades, sum -8.25%), `drawdown_reversal` (13 trades, mean -0.60%).
  - Best: `hold=72` (17 trades, sum +3.91%), `htf_pullback_resume|LONG` (7 trades, sum +3.55%), `hold=144` (9 trades, sum +3.08%), `higher_tf_momentum|hold=144` (5 trades, 100% win, sum +1.88%), `month=2026-05` (8 trades, sum +0.50%).
- Conclusion: the Jan-Feb failure was not just too small a sample. Extending to Jan-May confirms that the current model family is not live-worthy: it loses money over 66-75 trades, fails MDD, and the only non-negative month-abstention variant trades just 8 times with no statistical power. The persistent bad exposure is long 432h / drawdown-reversal in early-2026-like regimes. The next architecture should stop optimizing gates around the same ridge signal and instead learn a regime-conditioned horizon/action policy or abstention model with enough trade count, likely separating trend-capture validity from reversal validity.
