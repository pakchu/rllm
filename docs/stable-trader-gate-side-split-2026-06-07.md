# Stable trader gate/side task split (2026-06-07)

## Purpose
Gemma4 single-JSON trader SFT improved abstention with random-prior sampling, but still failed side/economic generalization. This split separates the problem into:
1. gate model: `TRADE` vs `NO_TRADE`, preserving abstention,
2. side specialist: `LONG` vs `SHORT` only after the gate approves a trade.

## Source
`data/stable_trader_policy_h144_t1p8_s1p5_all.jsonl`

## Outputs
Gate:
- `data/stable_trader_policy_h144_t1p8_s1p5_split_gate.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_split_gate_train.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_split_gate_val.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_split_gate_eval.jsonl`

Side:
- `data/stable_trader_policy_h144_t1p8_s1p5_split_side.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_split_side_train.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_split_side_val.jsonl`
- `data/stable_trader_policy_h144_t1p8_s1p5_split_side_eval.jsonl`

Summary:
- `data/stable_trader_policy_h144_t1p8_s1p5_split.summary.json`

## Counts
Gate rows:
- total: 2362
- `NO_TRADE`: 1802
- `TRADE`: 560
- split: train 1275 / val 552 / eval 535

Side rows:
- total: 560
- `LONG`: 341
- `SHORT`: 219
- split: train 372 / val 92 / eval 96

## Decision
Use this split for the next Gemma iteration. Gate should be trained with original trade prior; side can be trained on trade-only rows without diluting direction learning by the NO_TRADE majority.
