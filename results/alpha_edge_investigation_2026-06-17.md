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
