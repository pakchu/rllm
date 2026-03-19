# V2 Hierarchical LLM+RL Trading Redesign

## Why V1 failed

The V1 research branch repeatedly produced acceptable classification metrics but failed the actual objective: leakage-safe out-of-sample trading profitability under strict drawdown control.

Confirmed failure patterns:
- `target_horizon=3`, `symbolic/hybrid`, raw and bias-calibrated branches all failed 6m val -> 6m OOS direct split.
- `target_horizon=1`, `symbolic/hybrid`, raw and bias-calibrated branches also failed; several branches produced **zero relaxed/strict validation candidates** across the full direct-split search grid.
- MDD-constrained selection (`strict_mdd_pct <= 15`) produced **zero validation candidates** for the seqfull raw reports.
- Strong label metrics (balanced recall / directional recall) did **not** translate to a profitable action-score trading policy.

## Root causes

### 1. Monolithic 3-way action formulation is wrong for the problem
The current model is trained to emit `BUY/HOLD/SELL` in one step. This mixes two logically distinct decisions:
1. whether a trade should be taken at all
2. if a trade is taken, which direction it should take

That entangles no-trade filtering with direction classification and makes threshold search dominate the final behavior.

### 2. Validation selection overfits post-processing rather than model skill
The current pipeline relies on a downstream parameter search (`spread_mode`, `spread_threshold`, `hold_bars`, `cooldown_bars`, `inverse`).
This means the final system quality is too dependent on backtest parameter fitting rather than robust model outputs.

### 3. The HOLD label dominates the data but is not the same as "no edge"
In real trading data, `HOLD` is massively overrepresented. That pushes the model toward a distorted objective where raw label accuracy can improve while trading usefulness degrades.

### 4. Classification metrics are disconnected from PnL structure
Accuracy, balanced recall, and directional recall gap are helpful diagnostics, but they are not sufficient trading objectives. The current setup can score well on those metrics and still select catastrophic high-turnover or high-drawdown policies.

### 5. Regime awareness is present in features but not enforced in decision structure
Feature engineering improved context, but the policy still has to compress trade gating, direction, and implicit regime handling into one token decision.

## V2 design goals

The new architecture should optimize for **real trading utility first**, not label accuracy first.

### Goal A: Separate trade gating from direction
Replace the monolithic 3-way formulation with a hierarchical policy:
- **Stage 1: Trade Gate** -> `TRADE` vs `NO_TRADE`
- **Stage 2: Trade Side** -> `LONG` vs `SHORT` (only evaluated when Stage 1 says TRADE)

### Goal B: Reduce dependence on post-hoc parameter search
Model outputs should directly encode a more actionable policy surface. Downstream search should be narrower and lower leverage.

### Goal C: Put drawdown and turnover into the training objective more explicitly
The training target and reward path should discourage high-churn, high-drawdown behaviors rather than filtering them only after the fact.

### Goal D: Make regime state explicit in prompts and labels
The new path should explicitly feed regime summaries as first-class context, and allow gate behavior to depend on them.

## V2 execution plan

### Phase 1: Hierarchical data/prompt surfaces
- Add configurable action schemas.
- Add `trade_gate` schema (`TRADE`, `NO_TRADE`).
- Add `trade_side` schema (`LONG`, `SHORT`) for directional examples.
- Generalize prompts, parsing, evaluation, and calibration away from hard-coded `BUY/HOLD/SELL`.

### Phase 2: Gate-first model validation
- Train and validate a `trade_gate` model on the existing leak-safe spans.
- Primary metric: precision/recall trade-off for tradable opportunities, not raw 3-class accuracy.

### Phase 3: Side model validation
- Train a directional model only on gated tradable examples.
- Evaluate whether directional quality remains stable after removing HOLD from the label space.

### Phase 4: Hierarchical policy composition
- Compose gate + side into a final trading policy.
- Backtest with strict drawdown and turnover accounting.
- Keep the existing seqfull 6m val -> 6m OOS protocol.

### Phase 5: Trading-first selection
- Select candidates with explicit constraints:
  - strict MDD <= target
  - minimum trade count
  - positive CI lower bound
  - no leakage

## First implementation unit

The first code unit for V2 will not try to solve everything at once.
It will introduce:
- configurable action schemas
- `trade_gate` support through prompt generation, dataset building, training, and evaluation
- tests that lock the new surface

That allows the next commits to build the hierarchical policy incrementally, with each work unit committed separately.
