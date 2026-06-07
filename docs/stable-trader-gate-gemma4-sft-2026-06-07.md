# Stable trader gate Gemma4 SFT (2026-06-07)

## Purpose
The single-stage trader learned abstention only partially and had poor side quality. This run trains only the gate task (`TRADE` vs `NO_TRADE`) to test whether Gemma4 can learn the stable baseline's abstention behavior separately.

## Training
- Model: `gemma4-e4b` (`google/gemma-4-E4B-it`).
- Dataset: `data/stable_trader_policy_h144_t1p8_s1p5_split_gate_train.jsonl`.
- Sampling: random 512 rows.
- Steps: 16.
- Runtime: 115.2s.
- Final train loss: 0.6316.
- Local adapter: `checkpoints/stable_trader_gate_gemma4_e4b_h144_t1p8_s1p5_step16` (not committed).

## Val128 gate metrics
- Accuracy: 68.75%.
- Confusion:
  - target NO_TRADE / pred NO_TRADE: 81
  - target NO_TRADE / pred TRADE: 20
  - target TRADE / pred NO_TRADE: 20
  - target TRADE / pred TRADE: 7

## Eval128 gate metrics
- Accuracy: 80.47%.
- Confusion:
  - target NO_TRADE / pred NO_TRADE: 97
  - target NO_TRADE / pred TRADE: 16
  - target TRADE / pred NO_TRADE: 9
  - target TRADE / pred TRADE: 6

## Interpretation
The gate model learns the majority abstention behavior better than the single-stage action model, especially on Eval128. However, TRADE recall is still weak: it catches only 6/15 eval trade targets and 7/27 val trade targets in the sampled windows.

## Decision
Gate/side split remains the right direction, but gate training needs recall control. The next step should tune the gate with class-weighted or balanced-with-threshold selection, then combine with a side specialist only if gate precision/recall is acceptable.
