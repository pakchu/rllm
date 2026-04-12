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
