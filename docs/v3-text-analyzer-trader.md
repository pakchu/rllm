# V3 Text-Only Analyzer -> Trader Redesign

## Why V2 still wasn't enough

V2 improved the problem formulation by splitting gate (`TRADE/NO_TRADE`) and side (`LONG/SHORT`), and directional models generalized much better than the original 3-way image policy. However, the final objective is still not being met.

Observed facts:
- Side models (`trade_side`) generalize well on 6m val and 6m OOS.
- Gate models (`trade_gate`) generalize in a weak sense but still over-trade and fail to produce profitable validation candidates.
- Hierarchical composition plus multiple direct-search variants still yields zero or non-robust trading candidates.
- Image-based chart rendering adds failure surface (cache corruption, GPU load, multimodal overhead) without proving that vision is the key differentiator for the actual trading objective.

## V3 hypothesis

The highest-value next move is to abandon image conditioning and move to a fully text-first architecture:

1. **Analyzer model**
   - input: engineered text prompt only
   - output: structured market interpretation (regime, setup quality, volatility, directional bias, confidence, trade/no-trade recommendation)

2. **Trader model**
   - input: analyzer output + compact current-state text
   - output: final action / execution decision

This better matches the actual trading workflow:
- one model interprets the market
- another model decides what to do with that interpretation

## Core design goals

### Goal A: Remove image dependency
No chart rasterization, no image cache, no vision model load path.
This removes a major source of runtime instability and reduces memory pressure.

### Goal B: Make the analyzer output explicit and inspectable
Instead of forcing hidden reasoning into one token, require a compact structured output from the analyzer.

### Goal C: Make the trader operate on analyzer output only
The trader should learn execution policy conditioned on analyzer judgments, not raw price tensors/images.

### Goal D: Keep leak-safe evaluation protocol
The same 6m validation -> 6m OOS sequential split remains mandatory.

## Planned V3 phases

### Phase 1: Text-only modality support
- allow train/eval dataset building without images
- support text-only conversational records
- keep existing schemas working

### Phase 2: Analyzer task schema
- add structured analyzer target format
- first target should be a compact JSON-like schema or stable tagged line format

### Phase 3: Trader task schema
- add trader input builder that consumes analyzer output
- train trader on analyzer-conditioned decisions

### Phase 4: Two-stage evaluation
- analyzer quality metrics
- trader execution metrics
- final trading metrics on the composed chain

## First implementation unit

The first V3 implementation unit will:
- add text-only mode to the VLM/text pipeline
- preserve existing image-capable paths
- add tests for text-only dataset records and eval/training CLI surfaces

Only after that will analyzer/trader split-specific tasks be added.

## 2026-06-02 research + experiment reflection: what must change next

Recent long-horizon experiments invalidate the idea that a single symbolic rule book trained on recent BTC regimes is enough.  The 2025-focused hybrid candidate looked good on the recent holdout, but failed on 2020-2022 with large strict drawdown; the broader 2020-2024 retrain improved robustness but only reached about `63` eval trades, `~15%` CAGR, `~8.7%` strict MDD, and `~1.72` CAGR/MDD on 2025-2026.  Skip gates raised precision only by collapsing to too few trades.

External research points to the same structural fix: use the LLM as a **reasoning/router/risk signal generator**, not as a raw numeric predictor.  Relevant current directions:

- Trading-R1 frames trading LLMs around structured thesis generation, facts-grounded analysis, volatility-adjusted decision making, and RL-trained reasoning rather than one-shot numeric labels: https://arxiv.org/abs/2509.11420
- FinRL-DeepSeek combines LLM-generated risk assessment/recommendation signals with risk-sensitive RL (CVaR/PPO style), which matches our strict-MDD bottleneck better than pure return maximization: https://arxiv.org/abs/2502.07393
- Language-model-guided RL work treats LLM-generated strategies/signals as guidance for an RL execution agent rather than directly executing model text: https://huggingface.co/papers/2508.02366
- MM-DREX-style routing uses a dynamic router plus specialist experts for trend/reversal/breakout/positioning, aligning with our finding that 2021/2022 and 2025 require different specialists: https://huggingface.co/papers/2509.05080
- Feature-enriched imitative RL emphasizes enriched context and imitation+RL under partial observability, aligning with the need to pull DXY/kimchi/wave-trading macro signals into analyzer text: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5375707

### New V3 target structure

1. **Analyzer = state and risk router**
   - Produces discrete text/state labels: market cycle, volatility stress, trend/reversal/breakout suitability, expected adverse excursion bucket, trade horizon bucket, and confidence.
   - Input must include price/volume plus wave-trading macro context such as DXY and kimchi premium when available.
   - It must be trained/evaluated on past-only labels and never on eval-period outcomes.

2. **Specialist trader heads = action experts**
   - Separate candidates for trend, reversal, breakout, and flat/skip.
   - Each specialist should be selected only by train/test evidence, then frozen for final eval.

3. **RL/risk overlay = strict-MDD controller**
   - Optimize position decision, skip, cooldown, and horizon with a risk-sensitive objective (`return - drawdown/CVaR/MAE penalty`).
   - CAGR/MDD must be reported on untouched eval; test-selected parameters cannot be re-picked from eval.

4. **Validation protocol**
   - Train: long period, e.g. 2020-2022/2023 depending on experiment.
   - Test: at least 6 months, used for parameter/model selection.
   - Eval: at least 6 months, untouched final report.
   - Also report year-by-year and trade-count significance, because recent-only candidates have repeatedly overfit.

### Immediate next implementation unit

Build a train/test/eval router-specialist validation script that freezes all choices selected on test and reports final eval only once.  This should become the main gate before exporting analyzer/trader SFT labels.

## 2026-06-02 risk-sensitive action sweep result

A direct risk-sensitive state/action sweep was added after drift skip/flip overlays.  It selects actions per analyzer bucket from all LONG/SHORT hold candidates using train-only `mean_return - MAE - CVaR/downside` style objectives, selects hyperparameters on test, and reports eval without selection leakage.

Result on `train=2020-2024`, `test=2025H1`, `eval=2025H2-2026-02`:

- Best test-selected candidate: `test` 192 trades, CAGR `~65.9%`, strict MDD `~25.9%`, ratio `~2.55`.
- Untouched eval: 229 trades, CAGR `~-52.4%`, strict MDD `~51.4%`, ratio `~-1.02`.
- Even eval-diagnostic ordering stayed negative.  Coarse risk-sensitive action fitting over analyzer buckets does not generalize.

Implication: the next viable direction is not another bucketed outcome-fit sweep.  The analyzer must produce explicit regime-transition / edge-decay forecasts from richer context, and the trader/RL layer must be evaluated with rolling or online adaptation constraints rather than static train-period bucket averages.

## 2026-06-02 anti-overfit stability gate snapshot

`training/split_stability_report.py` was run over the main recent artifacts:

- stable policy TTE 2020-2022 / 2023-2024 / 2025-2026
- router-specialist TTE 2020-2022 / 2023-2024 / 2025-2026
- stable policy 2020-2024 / 2025H1 / 2025H2-2026
- drift skip overlay
- drift action overlay
- risk-sensitive state policy

Gate criteria: eval trades >= 30, eval CAGR/strict-MDD >= 3, eval strict MDD <= 15%, and ratio gap no worse than -3.  Result: `overall_pass=false`; every tested family failed.  Representative failures:

- Stable 2020-2022→2023-2024→2025-2026: test ratio `~2.21`, eval ratio `~-0.15`.
- Router-specialist: test ratio `~1.86`, eval ratio `~-0.48`.
- 2025H1-selected stable: test ratio `~11.49`, eval ratio `~-0.70`.
- Drift overlay/action overlay: test ratio `~11.81`, eval ratio `~0.26`, but only 15 eval trades.
- Risk-sensitive state policy: test ratio `~2.55`, eval ratio `~-1.02` with large drawdown.

This becomes the current stop condition for static bucket/rule experiments: do not promote a strategy, label set, or fine-tune target unless it passes the stability gate or improves the gate definition with stronger no-leak evidence.

## 2026-06-03 direction change: stop gate optimization, train edge-decay analyzer

The fixed gate-threshold family has failed repeatedly under train-bias, raw-score, and strict OOS replay.  The next stage therefore stops treating `TRADE/NO_TRADE` gate optimization as the main search surface.

New analyzer target:

- predict whether the current past-only regime edge **persists, decays, reverses, or becomes adverse-stress**;
- include macro context from wave_trading DXY / Kimchi / USDKRW features when local caches are available;
- output router hints such as `ALLOW_TREND_SPECIALIST`, `REDUCE_OR_SKIP_TREND_SPECIALIST`, `CONSIDER_REVERSAL_SPECIALIST`, or `RANGE_ROUTER_ONLY`;
- leave actual sizing/entry decisions to trader/RL execution layers.

Implementation unit:

- `training/edge_decay_analyzer_data.py`
  - builds past-only analyzer/router prompts;
  - labels future path diagnostics over short and long horizons;
  - explicitly avoids `TRADE/NO_TRADE` targets;
  - carries leakage guards: prompt is past-only, target uses future path, external features are backward-asof joined, and this is not gate-threshold optimization.
- `tests/test_edge_decay_analyzer_data.py`
  - validates label classification, leakage flags, and CLI output.

Initial real-data sample:

```bash
PYTHONPATH=. uv run python -m training.edge_decay_analyzer_data \
  --market-csv data/2023-01-01_2026-02-28_d2a88c0700504d6a5e15bc3839ad84b6.csv.gz \
  --wave-trading-root ../workspace/wave_trading \
  --output data/edge_decay_analyzer_h144_macro_sample.jsonl \
  --summary-output results/edge_decay_analyzer_h144_macro_sample_summary.json \
  --window-size 144 --short-hold-bars 72 --long-hold-bars 432 --stride-bars 96 \
  --leverage 0.5 --trend-feature trend_96 --trend-threshold 0.0025 \
  --max-records 1000
```

Sample label distribution over the first 1,000 records:

- `EDGE_PERSIST`: 153
- `WEAK_PERSIST`: 163
- `EDGE_DECAY`: 26
- `REVERSAL_RISK`: 68
- `ADVERSE_STRESS`: 242
- `NO_EDGE`: 26
- `NO_CLEAR_TREND`: 288

Router hint distribution:

- `ALLOW_TREND_SPECIALIST`: 153
- `REDUCE_OR_SKIP_TREND_SPECIALIST`: 268
- `CONSIDER_REVERSAL_SPECIALIST`: 68
- `RANGE_ROUTER_ONLY`: 288
- `LOW_CONFIDENCE_ROUTER`: 223

This is now the preferred LLM direction: train Gemma-style analyzer models to detect regime/edge transition states, then let a separate trader/RL layer consume those router states.  Future experiments should benchmark whether these labels improve strict OOS results over the non-LLM trend baseline before any live-candidate promotion.

## 2026-06-03 edge-decay router oracle diagnostic

After creating the edge-decay analyzer target, the next validation was to test whether the labels define a useful router objective before spending GPU time on fine-tuning.  `training/edge_decay_router_backtest.py` maps edge-decay targets/predictions into a strict OHLC route:

- `ALLOW_TREND_SPECIALIST` -> trade with `trend_side`
- `CONSIDER_REVERSAL_SPECIALIST` -> trade opposite `trend_side`
- `REDUCE_OR_SKIP_TREND_SPECIALIST`, `RANGE_ROUTER_ONLY`, `LOW_CONFIDENCE_ROUTER` -> skip

Important: this is an **oracle-label diagnostic** when run on teacher records.  The labels use future path outcomes, so this is not a deployable trading result.  It answers only: "if a model could predict these labels from past-only prompts, would the routing target be economically meaningful?"

Full stride-96 macro dataset:

- records: `data/edge_decay_analyzer_h144_macro_stride96_full.jsonl`
- summary: `results/edge_decay_analyzer_h144_macro_stride96_full_summary.json`
- total records: 3,457 from `2023-01-01 02:55:00` to `2026-02-26 02:55:00`

Oracle router strict result:

- artifact: `results/edge_decay_router_oracle_h144_macro_stride96_split.json`
- execution: hold `432` bars, cooldown `12` bars, leverage `0.5`, entry delay `1`

| Split | Samples | Trades | CAGR | Strict MDD | CAGR/MDD | CI95 lower mean trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,365 | 288 | 347.57% | 5.22% | 66.57 | 0.997% |
| val | 547 | 71 | 324.48% | 4.44% | 73.15 | 0.806% |
| oos | 535 | 70 | 353.84% | 6.87% | 51.47 | 0.849% |

Interpretation:

- The target structure has a very strong oracle upper bound under the strict simulator.
- This confirms the move away from gate thresholds: the economically useful decision is closer to "is this edge persisting, decaying, or reversing?" than "is TRADE score above a fixed margin?"
- The result is not a live candidate until a model predicts these router labels from past-only prompts and is evaluated with no access to teacher targets.

Next required step:

1. Build train/val/oos analyzer SFT splits from `edge_decay_analyzer_h144_macro_stride96_full.jsonl`.
2. Fine-tune/evaluate the Gemma analyzer on `edge_decay_label`, `transition_label`, `risk_label`, and `recommended_router_hint` exact-match/F1.
3. Replace teacher targets with model predictions in `edge_decay_router_backtest.py` and run the same strict split report.
4. Only then connect the trader/RL layer.

## 2026-06-03 model-prediction pipeline preparation

The oracle router is now connected to a model-prediction pipeline so future runs can replace teacher labels with Gemma analyzer outputs.

New utilities:

- `training/split_edge_decay_sft.py`
  - chronologically splits edge-decay records into train/val/oos JSONL files.
  - preserves leakage guards: prompts are past-only; targets are future path labels; no gate-threshold optimization.
- `training/eval_edge_decay_analyzer.py`
  - parses/evaluates full edge-decay JSON outputs across all keys:
    - `trend_side`
    - `edge_decay_label`
    - `transition_label`
    - `risk_label`
    - `recommended_router_hint`
  - writes prediction JSONL with a `prediction` field, directly consumable by `training.edge_decay_router_backtest`.
- `training/train_text_sft.py`
  - SFT dry-run summaries now count edge-decay target labels instead of treating them as generic analyzer rows.

Generated chronological splits from `data/edge_decay_analyzer_h144_macro_stride96_full.jsonl`:

| Split | Records | Period |
| --- | ---: | --- |
| train | 2,365 | `2023-01-01 02:55:00` → `2025-02-27 02:55:00` |
| val | 547 | `2025-03-01 02:55:00` → `2025-08-30 02:55:00` |
| oos | 535 | `2025-09-01 02:55:00` → `2026-02-26 02:55:00` |

Artifacts:

- `data/edge_decay_analyzer_h144_macro_train.jsonl`
- `data/edge_decay_analyzer_h144_macro_val.jsonl`
- `data/edge_decay_analyzer_h144_macro_oos.jsonl`
- `results/edge_decay_analyzer_h144_macro_split_summary.json`

Pipeline smoke checks:

1. Target-echo eval on 64 validation rows wrote predictions to `results/edge_decay_analyzer_val_target_echo_predictions.jsonl` with exact all-key accuracy `1.0`.
2. Those prediction rows were accepted by `edge_decay_router_backtest.py` and produced a strict router smoke report at `results/edge_decay_router_val_target_echo_smoke.json`.
3. Gemma-4-E4B SFT dry-run on 128 balanced train rows succeeded and wrote `checkpoints/edge_decay_analyzer_gemma4_dryrun/sft_summary.json`.

Next command for actual analyzer SFT:

```bash
PYTHONPATH=. uv run python -m training.train_text_sft \
  --train-jsonl data/edge_decay_analyzer_h144_macro_train.jsonl \
  --output-dir checkpoints/edge_decay_analyzer_gemma4_lora \
  --model-name gemma4-e4b \
  --sample-mode balanced \
  --max-samples 0 \
  --max-steps 400 \
  --max-seq-length 3072 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
```

Then evaluate with:

```bash
PYTHONPATH=. uv run python -m training.eval_edge_decay_analyzer \
  --eval-jsonl data/edge_decay_analyzer_h144_macro_val.jsonl \
  --output results/edge_decay_analyzer_val_model_eval.json \
  --prediction-output results/edge_decay_analyzer_val_model_predictions.jsonl \
  --prediction-mode model \
  --model-name gemma4-e4b \
  --adapter-dir checkpoints/edge_decay_analyzer_gemma4_lora
```

The real milestone is not SFT loss; it is strict router performance using `prediction` fields, not teacher targets.

## 2026-06-03 Gemma4 edge-decay analyzer run1 result

Actual Gemma4-E4B LoRA SFT was completed on the chronological train split, then evaluated with model-generated predictions only.  This is the first no-teacher router check for the edge-decay analyzer objective.

Training command/artifact:

- adapter: `checkpoints/edge_decay_analyzer_gemma4_lora_run1`
- train rows: 2,365 (`2023-01-01 02:55:00` → `2025-02-27 02:55:00`)
- model: `google/gemma-4-E4B-it`
- LoRA: r=16, alpha=32, dropout=0.05
- steps: 400, effective batch 8, max sequence length 3072
- runtime: 5,918s (~98.6m)
- final summary: `checkpoints/edge_decay_analyzer_gemma4_lora_run1/sft_summary.json`
- train loss: 0.1787

The original model eval loop was too slow because it generated one sample at a time and wrote predictions only after completion.  `training/eval_edge_decay_analyzer.py` now uses batched deterministic generation, left padding, `torch.inference_mode()`, progress logs, and streaming JSONL writes.  The first unbatched val run was stopped after ~56m with no output; the batched run completed val/oos generation with visible progress.

Model prediction metrics:

| Split | Samples | exact all keys | trend side acc | edge label acc | transition acc | risk acc | router hint acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 547 | 38.21% | 89.21% | 40.04% | 51.74% | 44.97% | 41.68% |
| oos | 535 | 39.25% | 91.03% | 42.43% | 48.97% | 47.48% | 44.30% |

Strict router backtest using model predictions:

| Split | Samples | Trades | CAGR | Strict MDD | CAGR/MDD | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| val | 547 | 54 | 3.61% | 10.77% | 0.335 | `results/edge_decay_router_val_model_run1.json` |
| oos | 535 | 43 | -26.64% | 19.69% | -1.352 | `results/edge_decay_router_oos_model_run1.json` |

Interpretation:

- This run **failed** the profitability target.  The teacher/oracle router upper bound remains strong, but the SFT analyzer does not predict the economically important router hints accurately enough.
- The model learned `trend_side` well, but that is the least valuable key; the router depends on separating `ALLOW_TREND_SPECIALIST`, `CONSIDER_REVERSAL_SPECIALIST`, and skip states.  Those labels are confused heavily with `REDUCE_OR_SKIP_TREND_SPECIALIST` and `RANGE_ROUTER_ONLY`.
- The result supports the user's concern that simply asking the LLM to reconstruct many numeric-derived labels is not enough.  The LLM should be used where it has an advantage: compressing multi-source market context into a small set of semantically robust regime narratives, not copying a brittle five-key teacher JSON.

Next structural change:

1. Replace five-key label imitation with a smaller decision-critical target: `TRADE_TREND`, `FADE_TREND`, `ABSTAIN`, plus a concise natural-language rationale class.
2. Make the analyzer output calibrated uncertainty/evidence quality, not only a class.
3. Train/evaluate with router utility weighting so confusing a profitable allow/fade state with skip is penalized differently from confusing two skip-like states.
4. Keep the batched eval path and require val+oos strict router reports before any further RL/trader stage.

## 2026-06-03 decision-critical analyzer target

The failed Gemma4 edge-decay run showed that a five-key teacher JSON is not the right LLM target.  The model learned the obvious trend side but failed on the economically important router hint.  The next structure compresses the teacher into the smallest decision-critical target:

- `TRADE_TREND`: trade with the current trend.
- `FADE_TREND`: trade against the current trend.
- `ABSTAIN`: no position.

New module:

- `training/decision_analyzer_data.py`
  - converts edge-decay teacher records into `decision_analyzer` SFT records.
  - prompt remains past-only.
  - target is compressed from future path diagnostics, so it is still a teacher label and not deployable until model predictions replace targets.
  - output keys: `decision`, `action_side`, `confidence`, `rationale_class`.

Generated artifacts:

- full records: `data/decision_analyzer_h144_macro_stride96_full.jsonl`
- full summary: `results/decision_analyzer_h144_macro_stride96_full_summary.json`
- split summary: `results/decision_analyzer_h144_macro_split_summary.json`
- dry-run SFT summary: `checkpoints/decision_analyzer_gemma4_dryrun/sft_summary.json`

Full decision distribution over 3,457 records (`2023-01-01 02:55:00` → `2026-02-26 02:55:00`):

| Decision | Count |
| --- | ---: |
| `ABSTAIN` | 2,536 |
| `TRADE_TREND` | 675 |
| `FADE_TREND` | 246 |

Chronological splits:

| Split | Records | Period | TRADE | FADE | ABSTAIN |
| --- | ---: | --- | ---: | ---: | ---: |
| train | 2,370 | `2023-01-01 02:55:00` → `2025-02-28 18:55:00` | 461 | 148 | 1,761 |
| val | 552 | `2025-03-01 02:55:00` → `2025-08-31 18:55:00` | 108 | 54 | 390 |
| oos | 535 | `2025-09-01 02:55:00` → `2026-02-26 02:55:00` | 106 | 44 | 385 |

Strict oracle router check using decision targets:

| Split | Samples | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,370 | 288 | 346.72% | 5.22% | 66.41 |
| val | 552 | 71 | 321.14% | 4.44% | 72.40 |
| oos | 535 | 70 | 353.84% | 6.87% | 51.47 |

This preserves the oracle upper bound while giving the LLM a much smaller and more directly tradable output space than the failed five-key edge-decay target.

Next actual Gemma run:

```bash
PYTHONPATH=. uv run python -m training.train_text_sft \
  --train-jsonl data/decision_analyzer_h144_macro_train.jsonl \
  --output-dir checkpoints/decision_analyzer_gemma4_lora_run1 \
  --model-name gemma4-e4b \
  --sample-mode balanced \
  --max-samples 0 \
  --max-steps 400 \
  --max-seq-length 3072 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
```

The next evaluation must use model predictions, not decision targets, then run `training.edge_decay_router_backtest` on those prediction rows.  Promotion condition remains val+oos strict router evidence, not SFT loss.

## 2026-06-03 decision feature learnability gate

The decision-critical target kept the oracle upper bound, but two Gemma4-E4B SFT runs exposed a lower-level issue: changing class balance moved the failure mode rather than producing a robust predictor.

Gemma decision analyzer runs:

| Run | Train data | Result |
| --- | --- | --- |
| run1 | natural chronological train, 400 LoRA steps | collapsed to `ABSTAIN/NONE` on all 552 val samples; strict router made 0 trades |
| run2 | balanced train 700/700/700, 400 LoRA steps | broke abstain collapse but over-traded; val strict router 101 trades, CAGR `-24.54%`, strict MDD `17.01%`, ratio `-1.44` |

A dependency-free feature learnability gate was added before spending more GPU time:

- `training/decision_feature_learnability.py`
  - parses the same past-only analyzer summary seen by the decision LLM;
  - flattens symbolic fields and binned numeric evidence;
  - trains a categorical Naive Bayes baseline on train only;
  - reports train/val/oos accuracy versus the majority-class baseline;
  - can write router-compatible prediction JSONL.
- `tests/test_decision_feature_learnability.py`
  - verifies prompt summary parsing, feature flattening, toy separability, CLI outputs, and that prediction side uses only past `source_edge_target.trend_side`, never `target.action_side`.

Actual no-leak feature gate on the current train/val/oos decision splits:

| Split | Samples | NB accuracy | Majority baseline | Beats baseline? |
| --- | ---: | ---: | ---: | --- |
| train | 2,370 | 72.41% | 74.30% | no |
| val | 552 | 56.52% | 70.65% | no |
| oos | 535 | 54.02% | 71.96% | no |

Artifacts:

- feature report: `results/decision_feature_learnability_nb_run1.json`
- predictions: `results/decision_feature_nb_run1_predictions/{train,val,oos}_predictions.jsonl`
- strict val router: `results/decision_router_val_feature_nb_run1.json`
- strict oos router: `results/decision_router_oos_feature_nb_run1.json`

Strict router using the no-leak NB decision predictions and past-only trend side:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| val | 79 | -31.82% | 24.41% | -1.30 |
| oos | 75 | -14.72% | 13.44% | -1.09 |

Interpretation:

- The current analyzer summary features are not proving decision-label learnability even for a cheap baseline; the LLM failures are therefore not just a Gemma capacity/training issue.
- The decision prompt was also simplified to remove nested edge-decay instructions.  It now feeds only the past-only analyzer summary JSON instead of wrapping the previous edge prompt, preventing conflicting instructions such as edge-label output versus decision output.
- Promotion rule: do not run another long Gemma SFT on this target unless a train-only baseline beats majority on both val and oos or the target/feature schema is redesigned.

Next structural fix:

1. Redesign analyzer labels away from direct future-path oracle classes and toward past-computable state/risk descriptions plus separately evaluated execution outcomes.
2. Keep macro/price sequence features, but preserve more local structure than the current coarse symbolic summary.
3. Use the learnability gate as a cheap stop/go check before every LLM SFT target.

## 2026-06-03 train-selected analyzer state edge report

After the decision-label learnability gate failed, the next diagnostic stopped asking the LLM to predict future oracle classes and instead tested whether past-only analyzer states have stable economic edge.

New module:

- `training/analyzer_state_edge_report.py`
  - extracts past-only analyzer state buckets from the same summary JSON;
  - chooses bucket actions (`TREND` or `FADE`) on train only using path diagnostics;
  - freezes that bucket policy for val/oos;
  - exports router-compatible predictions whose side comes only from past `source_edge_target.trend_side`, never from target action side.
- `tests/test_analyzer_state_edge_report.py`
  - verifies train-only bucket selection, fixed val/oos application, CLI outputs, and side leakage protection.

Run 1 bucket fields:

```text
regime,trend_alignment,location,volatility_level,risk_state,sequence_stats.wide_or_extreme
```

Selection: train only, min train count `20`, min mean return `0.1%`, max `64` buckets.

Offline path-diagnostic report:

| Split | Trades | Mean return/trade | Win rate | CI95 mean return |
| --- | ---: | ---: | ---: | ---: |
| train | 369 | 0.240% | 52.6% | [0.089%, 0.392%] |
| val | 100 | -0.011% | 41.0% | [-0.268%, 0.247%] |
| oos | 83 | -0.313% | 43.4% | [-0.627%, -0.000%] |

Strict router using the frozen state policy:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| val | 58 | 7.99% | 10.38% | 0.77 |
| oos | 51 | -23.57% | 20.83% | -1.13 |

A stricter train positive-CI variant selected only one bucket:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| val | 8 | 2.72% | 2.35% | 1.16 |
| oos | 3 | -2.88% | 4.58% | -0.63 |

Interpretation:

- Coarse analyzer states do not carry stable enough edge across the current val/oos periods.
- This narrows the failure: the issue is not only LLM class imitation; the current state abstraction itself is too weak/nonstationary for execution.
- The next redesign must preserve more temporal/local structure or introduce adaptive/online regime memory.  Simply mapping current symbolic state buckets to fixed actions is another overfit path.

## 2026-06-03 leakage-safe online state memory diagnostic

Fixed train-selected state buckets failed OOS, so the next diagnostic tested an adaptive regime-memory idea without lookahead.  The goal was to see whether similar recent analyzer states can guide action selection better than a static bucket map.

New module:

- `training/online_state_memory_report.py`
  - keeps a recency-aware memory of prior analyzer states;
  - at each decision, a prior example is eligible only after `example.signal_pos + hold_bars <= current.signal_pos`;
  - chooses `TREND`, `FADE`, or `SKIP` from top-k similar matured examples;
  - exports router-compatible predictions using only past `source_edge_target.trend_side` for side selection.
- `tests/test_online_state_memory_report.py`
  - verifies delayed memory maturity, similar-memory action choice, CLI output, and side leakage protection.

Default run config:

```text
similarity_fields = regime,trend_alignment,location,volatility_level,risk_state,sequence_stats.wide_or_extreme,sequence_stats.rally_or_up,sequence_stats.drop_or_down
top_k = 64
min_similarity = 0.625
min_neighbors = 20
min_mean_return = 0.1%
recency_halflife_bars = 8640
hold_bars = 432
```

Offline online-memory report:

| Split | Trades | Mean return/trade | Win rate | CI95 mean return |
| --- | ---: | ---: | ---: | ---: |
| train | 1,030 | 0.071% | 50.1% | [-0.023%, 0.165%] |
| val | 201 | -0.117% | 45.8% | [-0.314%, 0.080%] |
| oos | 233 | -0.007% | 51.1% | [-0.195%, 0.182%] |

Strict router using online-memory predictions:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| val | 76 | -44.68% | 26.89% | -1.66 |
| oos | 76 | -19.36% | 15.83% | -1.22 |

A single risk-sensitive variant (`top_k=32`, `min_similarity=0.75`, `min_mean_return=0.2%`, `mae_penalty=0.05`, `recency_halflife_bars=4320`) also failed:

| Split | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| val | 82 | -30.70% | 19.68% | -1.56 |
| oos | 81 | -26.79% | 17.47% | -1.53 |

Interpretation:

- Online/adaptive memory over the current symbolic summary does not solve the problem.
- The failure survives no-lookahead maturity rules, so the issue is not only fixed train-bucket overfit.
- The current summary representation lacks enough predictive state information for the 432-bar execution horizon.  Next work should either shorten/condition horizons, enrich the analyzer with direct temporal path prototypes, or move the LLM role toward generating multi-horizon uncertainty/risk narratives rather than action selection.

## 2026-06-03 multi-horizon edge viability report

After fixed buckets and online memory failed at the 432-bar execution horizon, the next diagnostic tested whether the horizon itself was the main problem.  The script recomputes TREND/FADE path outcomes from OHLC for multiple hold lengths using only past `source_edge_target.trend_side` for action orientation.

New module:

- `training/multi_horizon_edge_report.py`
  - recomputes same-trend and fade/opposite outcomes for multiple hold horizons;
  - reports static `TREND`, static `FADE`, and non-deployable `ORACLE` upper bound per split;
  - uses future OHLC only for offline target design, never for inference side selection.
- `tests/test_multi_horizon_edge_report.py`
  - verifies horizon parsing, path summary behavior on toy OHLC, oracle upper bound, and CLI output.

Run config:

```text
hold_bars_list = 36,72,144,288,432
entry_delay_bars = 1
leverage = 0.5
fee = 0.04%, slippage = 0.01%
```

Summary table from `results/multi_horizon_edge_report_run1.json`:

| Hold bars | Train best static mean | Val best static mean | OOS best static mean | Val oracle mean | OOS oracle mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | -0.043% | -0.049% | -0.033% | 0.231% | 0.262% |
| 72 | -0.034% | -0.045% | -0.009% | 0.368% | 0.422% |
| 144 | -0.028% | -0.041% | -0.022% | 0.529% | 0.595% |
| 288 | -0.049% | 0.007% | -0.028% | 0.786% | 0.892% |
| 432 | 0.013% | -0.029% | -0.032% | 0.930% | 1.045% |

Interpretation:

- Shortening the horizon alone does not create a usable static TREND/FADE edge; static action means are negative or near zero across val/oos.
- The oracle upper bound is strong and increases with horizon, so the opportunity exists, but it requires choosing between trend/fade/skip from richer state information.
- The next target should not be a single action label at one horizon.  It should expose multi-horizon path-shape/risk information to the LLM, e.g. direction stability, reversal pressure, adverse-excursion bucket, and horizon-specific uncertainty.  The trader/RL layer can then learn when that path-shape narrative is actionable.

## 2026-06-03 multi-horizon path-shape analyzer target

The multi-horizon viability report showed that static action labels are weak while oracle upper bounds remain strong.  The next LLM target therefore stops asking for a final action and instead asks the analyzer to describe path shape and risk across horizons.

New module:

- `training/multi_horizon_path_shape_data.py`
  - builds `multi_horizon_path_shape_analyzer` SFT records;
  - prompt uses only the past-only analyzer summary JSON;
  - target uses future OHLC path outcomes to describe horizon-wise trend/fade return buckets, adverse-excursion buckets, relative edge, best path, and tradable path count;
  - top-level target keys: `trend_side`, `direction_stability`, `reversal_pressure`, `risk_profile`, `horizons`, `summary_counts`;
  - this is explicitly not a final action target.
- `tests/test_multi_horizon_path_shape_data.py`
  - verifies target derivation, leakage guards, summary counts, and CLI output.

Generated artifacts:

- full records: `data/multi_horizon_path_shape_h36_72_144_288_432_full.jsonl`
- full summary: `results/multi_horizon_path_shape_h36_72_144_288_432_full_summary.json`
- split summary: `results/multi_horizon_path_shape_h36_72_144_288_432_split_summary.json`
- dry-run SFT summary: `checkpoints/multi_horizon_path_shape_gemma4_dryrun/sft_summary.json`

Full dataset distribution over 3,457 records:

| Target key | Distribution |
| --- | --- |
| `direction_stability` | `HORIZON_CONFLICT=1461`, `FADE_STABLE=679`, `TREND_STABLE=567`, `NO_STABLE_EDGE=750` |
| `reversal_pressure` | `HIGH=1767`, `LOW=1302`, `MEDIUM=388` |
| `risk_profile` | `EXTREME_PATH_RISK=1410`, `MIXED_PATH_RISK=1002`, `HIGH_PATH_RISK=801`, `LOW_PATH_RISK=244` |
| `trend_side` | `LONG=1443`, `SHORT=1295`, `NONE=719` |

Chronological splits:

| Split | Records | Period |
| --- | ---: | --- |
| train | 2,370 | `2023-01-01 02:55:00` → `2025-02-28 18:55:00` |
| val | 552 | `2025-03-01 02:55:00` → `2025-08-31 18:55:00` |
| oos | 535 | `2025-09-01 02:55:00` → `2026-02-26 02:55:00` |

Dry-run SFT on 128 balanced train rows resolved `gemma4-e4b` to `google/gemma-4-E4B-it`, with prompt length mean `2572` chars and target length mean `1254` chars.  This fits the existing `max_seq_length=3072` path but is close enough that actual SFT should monitor truncation.

Next step:

1. Add an evaluator for path-shape JSON keys before long SFT.
2. Run target-echo smoke to ensure parsing works.
3. Fine-tune Gemma on this target only if the evaluator can report top-level and horizon-level accuracy, then pass predicted path-shape narratives into a separate trader/RL layer.

## 2026-06-03 path-shape evaluator and target-echo smoke

Before long Gemma SFT, the path-shape target now has a dedicated evaluator.

New module:

- `training/eval_multi_horizon_path_shape_analyzer.py`
  - parses model/target JSON robustly, including fenced or noisy text around the JSON object;
  - normalizes top-level keys: `trend_side`, `direction_stability`, `reversal_pressure`, `risk_profile`;
  - normalizes each horizon's `trend_return_bucket`, `fade_return_bucket`, `trend_mae_bucket`, `fade_mae_bucket`, `relative_edge`, `best_path`, and `tradable_path_count`;
  - reports exact top-level accuracy, exact all-key accuracy, per-top-key accuracy, horizon-key micro accuracy, and per-horizon confusion matrices;
  - supports target-echo and model generation modes.
- `tests/test_eval_multi_horizon_path_shape_analyzer.py`
  - verifies invalid/default parsing, nested horizon normalization, target-echo metrics, and prediction JSONL export.

Target-echo smoke on full validation split:

```bash
PYTHONPATH=. uv run python -m training.eval_multi_horizon_path_shape_analyzer \
  --eval-jsonl data/multi_horizon_path_shape_h36_72_144_288_432_val.jsonl \
  --output results/multi_horizon_path_shape_val_target_echo_eval.json \
  --prediction-output results/multi_horizon_path_shape_val_target_echo_predictions.jsonl \
  --prediction-mode target_echo \
  --hold-bars-list 36,72,144,288,432
```

Result:

- samples: `552`
- exact top-level accuracy: `1.0`
- exact all-key accuracy: `1.0`
- horizon-key micro accuracy: `1.0` for all horizon keys

This locks the evaluation bridge.  The next safe step is an actual Gemma path-shape LoRA run, followed by model-mode eval on val/oos.  Because target length averages ~1,254 chars, use a larger generation budget than decision targets, e.g. `--max-new-tokens 1536`, and monitor truncation.

## 2026-06-04 Gemma4 multi-horizon path-shape LoRA result

A real Gemma4 LoRA SFT was run on the multi-horizon path-shape analyzer target.  This target asks the analyzer to describe future path-shape classes, not to emit a direct trade action.  The intent is to test whether an LLM can learn a richer router state before attaching a trader/RL execution layer.

Training setup:

- model: `google/gemma-4-E4B-it` via repo alias `gemma4-e4b`
- adapter: `checkpoints/multi_horizon_path_shape_gemma4_lora_run1`
- train data: `data/multi_horizon_path_shape_h36_72_144_288_432_train.jsonl`
- train rows: `2,370`, chronological range `2023-01-01` to `2025-02-28`
- val rows: `552`, chronological range `2025-03-01` to `2025-08-31`
- OOS rows: `535`, chronological range `2025-09-01` to `2026-02-26`
- sequence length: `4096`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- optimization: `400` steps, batch `1`, grad accumulation `8`, `lr=2e-5`, balanced sample mode
- runtime: `3,494s` (`~58.2m`), final train loss `0.06021`

Leakage boundary:

- model-mode eval uses only the prompt text;
- the path-shape target uses future OHLC and is therefore only a supervised label;
- target-echo is marked as oracle-only and is not a trading result;
- this run evaluates label learnability, not profitability.

Validation / OOS analyzer accuracy:

| Split | Samples | Exact top-level | Exact all keys | Trend side | Direction stability | Reversal pressure | Risk profile | Best-path micro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 552 | 25.54% | 19.20% | 88.59% | 55.80% | 55.07% | 40.40% | 50.54% |
| OOS | 535 | 22.24% | 18.50% | 91.03% | 58.32% | 57.01% | 37.20% | 53.31% |

Artifacts:

- `results/multi_horizon_path_shape_val_model_run1_eval.json`
- `results/multi_horizon_path_shape_val_model_run1_predictions.jsonl`
- `results/multi_horizon_path_shape_oos_model_run1_eval.json`
- `results/multi_horizon_path_shape_oos_model_run1_predictions.jsonl`
- `logs/multi_horizon_path_shape_gemma4_lora_run1_train.log`
- `logs/multi_horizon_path_shape_val_model_run1_eval.log`
- `logs/multi_horizon_path_shape_oos_model_run1_eval.log`

Interpretation:

- Positive: the model learned `trend_side` very well and OOS did not collapse, so the text prompt carries stable direction information.
- Weak: exact structured path-shape reconstruction is low, especially risk/path bucket fields.  Low train loss plus weak structured OOS accuracy suggests the target is too long/nested and partly memorization-friendly.
- Current conclusion: do **not** promote this analyzer to trading yet.  The next useful step is to compress the analyzer target into fewer decision-relevant states and/or make step/horizon selection a first-class label, then test whether those states improve strict no-leak trader/RL backtests.

## 2026-06-04 compact path-shape router target

The first Gemma4 path-shape run showed that the model can learn direction but struggles with long nested horizon JSON.  The next target therefore compresses the teacher path-shape labels into a short router state:

- `trend_side`
- `action_path`: `TREND` / `FADE` / `NONE`
- `horizon_bars`: `0,36,72,144,288,432`
- `horizon_policy`: `SHORT_STEP` / `MID_STEP` / `LONG_STEP` / `SKIP_STEP`
- `edge_quality`: `STRONG` / `MODERATE` / `WEAK` / `NO_EDGE`
- `risk_budget`: `AGGRESSIVE_OK` / `NORMAL` / `SMALL` / `AVOID_OR_TINY`
- `score_bucket`, `direction_stability`, `reversal_pressure`

This makes the analyzer responsible for the step/horizon/router decision while still leaving final order execution to trader/RL.  The label is derived from future path-shape teacher targets, so the same leakage boundary applies: prompts are past-only, targets are supervised future-path labels, and target-echo is oracle-only.

Generated split summaries:

| Split | Rows | Target chars mean | Action path distribution | Horizon policy distribution | Risk budget distribution |
| --- | ---: | ---: | --- | --- | --- |
| train | 2,370 | 233.8 | FADE 915 / TREND 914 / NONE 541 | LONG 1,256 / SKIP 541 / SHORT 321 / MID 252 | AVOID 1,549 / SMALL 512 / NORMAL 206 / AGGRESSIVE 103 |
| val | 552 | 232.5 | FADE 227 / TREND 217 / NONE 108 | LONG 317 / SKIP 108 / SHORT 70 / MID 57 | AVOID 291 / SMALL 147 / NORMAL 74 / AGGRESSIVE 40 |
| OOS | 535 | 232.7 | FADE 227 / TREND 206 / NONE 102 | LONG 298 / SKIP 102 / SHORT 73 / MID 62 | AVOID 319 / SMALL 142 / NORMAL 47 / AGGRESSIVE 27 |

Dry-run SFT summary:

- artifact: `checkpoints/compact_path_shape_gemma4_dryrun/sft_summary.json`
- model alias: `gemma4-e4b` -> `google/gemma-4-E4B-it`
- train rows: `2,370`
- prompt chars mean: `2,647.1`
- target chars mean: `233.8`, down from the previous path-shape target mean of about `1,266`
- suggested first real SFT: `max_seq_len=3072`, `max_steps=300-400`, `max_new_tokens=384`

Next gate: train this compact analyzer and evaluate val/OOS exact JSON accuracy.  Promote to trader/RL backtest only if OOS action_path/horizon_policy/risk_budget accuracy is materially better than the nested target run and does not collapse into a static majority policy.

## 2026-06-04 compact Gemma4 router-state LoRA result

A full Gemma4 LoRA SFT was run on the compact path-shape router target.

Training setup:

- model: `google/gemma-4-E4B-it` via repo alias `gemma4-e4b`
- adapter: `checkpoints/compact_path_shape_gemma4_lora_run1`
- train data: `data/compact_path_shape_h36_72_144_288_432_train.jsonl`
- train rows: `2,370`
- sequence length: `3072`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- optimization: `400` steps, batch `1`, grad accumulation `8`, `lr=2e-5`, balanced sample mode
- runtime: `6,028s` (`~100.5m`), final train loss `0.102`

Validation / OOS compact analyzer accuracy:

| Split | Samples | Exact all keys | Exact primary keys | Trend side | Action path | Horizon policy | Edge quality | Risk budget | Direction stability | Reversal pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 552 | 22.83% | 31.52% | 88.95% | 54.71% | 68.84% | 84.96% | 52.90% | 55.80% | 58.70% |
| OOS | 535 | 19.63% | 29.72% | 91.03% | 52.90% | 68.79% | 86.54% | 59.63% | 56.45% | 58.32% |

Artifacts:

- `checkpoints/compact_path_shape_gemma4_lora_run1/sft_summary.json`
- `logs/compact_path_shape_gemma4_lora_run1_train.log`
- `results/compact_path_shape_val_model_run1_eval.json`
- `results/compact_path_shape_val_model_run1_predictions.jsonl`
- `results/compact_path_shape_oos_model_run1_eval.json`
- `results/compact_path_shape_oos_model_run1_predictions.jsonl`

Interpretation:

- The compact schema improved the learnability of high-level router fields versus nested path-shape reconstruction: `horizon_policy` is stable at ~69% and `edge_quality` is ~85-87% on val/OOS.
- `trend_side` remains strong at ~89-91%.
- `action_path` is only ~53-55%, and confusion shows many FADE targets predicted as TREND.  That means the model still has a trend/fade discrimination problem.
- `risk_budget` is not genuinely learned; predictions collapse almost entirely to `AVOID_OR_TINY`, so the apparent OOS accuracy reflects class imbalance rather than useful sizing control.
- This is not yet a trading candidate.  The next work unit should remove or rebalance the collapsed `risk_budget` label, make trend-vs-fade discrimination easier, and backtest only fields that show genuine OOS learnability (`trend_side`, `horizon_policy`, `edge_quality`) before further SFT.

## 2026-06-04 compact router strict backtest result

The compact analyzer was next tested as a real model-prediction router on strict OHLC execution.  This uses model predictions only, not target echo.  Strict MDD is marked bar-by-bar including adverse intrabar high/low movement, with one-bar entry delay.

Conservative learned-fields route:

- route: ignore weak `action_path` and collapsed `risk_budget`; trade `trend_side` only when `edge_quality >= STRONG`; map `horizon_policy` to hold bars (`SHORT=72`, `MID=144`, `LONG=432`); cooldown `12`; leverage `0.5`.
- val: `101` trades, CAGR `-36.85%`, strict MDD `21.80%`, CAGR/MDD `-1.69`, CI95 mean trade `[-0.459%, 0.018%]`.
- OOS: `99` trades, CAGR `8.67%`, strict MDD `11.92%`, CAGR/MDD `0.73`, CI95 mean trade `[-0.193%, 0.290%]`.

Val-only sweep:

- candidates: `72` combinations across route mode (`learned_fields`/`action_path`), min edge (`MODERATE`/`STRONG`), cooldown (`0/12/36`), and hold maps.
- selection was based on val only; OOS was not used for parameter choice.
- selected config: `action_path`, `min_edge_quality=MODERATE`, cooldown `0`, holds `SHORT=36`, `MID=72`, `LONG=288`.
- selected val: `121` trades, CAGR `-0.89%`, strict MDD `16.55%`, CAGR/MDD `-0.05`.
- untouched OOS: `120` trades, CAGR `-35.67%`, strict MDD `27.12%`, CAGR/MDD `-1.32`.

Conclusion:

- Compact analyzer outputs are not yet monetizable.  Even the val-selected route cannot produce positive validation economics and fails badly on OOS.
- The issue is not just execution tuning; the model's current `action_path`/trend-vs-fade signal is not economically reliable.
- Next target repair should remove the collapsed risk label, avoid forcing FADE/TREND from weak path buckets, and train a binary/ordinal question that the LLM can actually learn: e.g. `trend_continuation_quality` + `fade_warning` + `skip_reason`, with class balancing and a baseline comparison before another full 100-minute SFT.

## 2026-06-04 repaired router-state target and learnability gate

After compact run1 failed strict execution, the next target repair removed the collapsed `risk_budget` label and decomposed the overloaded `action_path` into simpler questions:

- `trend_continuation_quality`: `CONTINUE_STRONG` / `CONTINUE_WATCH` / `NO_CONTINUATION`
- `fade_warning`: `FADE_STRONG` / `FADE_WATCH` / `NO_FADE_WARNING`
- `skip_reason`: `TRADEABLE_TREND` / `TRADEABLE_FADE` / `CONFLICTING_HORIZONS` / `ADVERSE_RISK` / `LOW_CONFIDENCE` / `NO_EDGE`
- `primary_route`: `TREND` / `FADE` / `SKIP`
- `horizon_policy`: `SHORT_STEP` / `MID_STEP` / `LONG_STEP` / `SKIP_STEP`

Generated repaired split distributions:

| Split | Rows | Primary route | Fade warning | Skip reason highlights |
| --- | ---: | --- | --- | --- |
| train | 2,370 | SKIP 1,739 / FADE 339 / TREND 292 | NO 914 / STRONG 750 / WATCH 706 | ADVERSE 1,009 / NO_EDGE 522 / FADE 339 / TREND 292 |
| val | 552 | SKIP 325 / FADE 130 / TREND 97 | NO 204 / STRONG 204 / WATCH 144 | ADVERSE 183 / FADE 130 / NO_EDGE 107 / TREND 97 |
| OOS | 535 | SKIP 360 / FADE 97 / TREND 78 | STRONG 190 / NO 186 / WATCH 159 | ADVERSE 217 / NO_EDGE 100 / FADE 97 / TREND 78 |

Key-wise categorical Naive Bayes learnability baseline:

| Key | Val acc | Val majority | OOS acc | OOS majority | Pass? |
| --- | ---: | ---: | ---: | ---: | --- |
| `trend_continuation_quality` | 47.64% | 54.53% | 44.11% | 54.39% | no |
| `fade_warning` | 49.64% | 36.96% | 48.79% | 35.51% | yes |
| `skip_reason` | 39.86% | 33.15% | 43.18% | 40.56% | weak yes |
| `primary_route` | 50.91% | 58.88% | 54.39% | 67.29% | no |
| `horizon_policy` | 44.93% | 58.88% | 50.28% | 67.29% | no |

Interpretation:

- The repaired target did not make full route/horizon selection learnable.  `primary_route` and `horizon_policy` still fail majority on val/OOS.
- `fade_warning` is the clearest learnable signal and generalizes above majority on both val and OOS.
- `skip_reason` barely clears majority and may be useful as an auxiliary explanation, but not as a direct trading policy.
- Next SFT should be narrower than previous full JSON targets: train/evaluate a `fade_warning`-centric analyzer first, optionally with `skip_reason` auxiliary output, before another strict trading backtest.

Dry-run SFT summary:

- artifact: `checkpoints/repaired_router_state_gemma4_dryrun/sft_summary.json`
- train rows: `2,370`
- prompt chars mean: `2,671.1`
- target chars mean: `246.0`
- no model loaded; this only verifies SFT schema/sampling readiness.

## 2026-06-04 fade-warning narrow SFT target

The repaired key-wise baseline showed that only `fade_warning` clearly beats majority on both val and OOS.  The next SFT target is therefore narrowed to a fade-risk analyzer instead of another full route/horizon JSON.

Target keys:

- `trend_side`
- `fade_warning`: `FADE_STRONG` / `FADE_WATCH` / `NO_FADE_WARNING`
- `skip_reason`: auxiliary explanation, not direct routing
- `trend_continuation_quality`: auxiliary context

Generated split summaries:

| Split | Rows | Target chars mean | Fade warning distribution | Skip reason highlights |
| --- | ---: | ---: | --- | --- |
| train | 2,370 | 127.7 | NO 914 / STRONG 750 / WATCH 706 | ADVERSE 1,009 / NO_EDGE 522 / FADE 339 / TREND 292 |
| val | 552 | 127.9 | STRONG 204 / NO 204 / WATCH 144 | ADVERSE 183 / FADE 130 / NO_EDGE 107 / TREND 97 |
| OOS | 535 | 127.8 | STRONG 190 / NO 186 / WATCH 159 | ADVERSE 217 / NO_EDGE 100 / FADE 97 / TREND 78 |

Validation bridge:

- target-echo val: exact all keys `1.0`, exact primary key `1.0`
- target-echo OOS: exact all keys `1.0`, exact primary key `1.0`
- dry-run artifact: `checkpoints/fade_warning_gemma4_dryrun/sft_summary.json`
- prompt chars mean: `2,522.1`
- target chars mean: `127.7`

Next gate: run a real Gemma4 LoRA on this narrow target and evaluate model-mode `fade_warning` on val/OOS.  A later trading test should use it as a fade-risk veto/filter, not as a standalone action policy.
