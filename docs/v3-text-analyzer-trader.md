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
