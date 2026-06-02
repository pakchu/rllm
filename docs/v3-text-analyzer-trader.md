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
