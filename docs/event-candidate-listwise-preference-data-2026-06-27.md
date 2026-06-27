# Event candidate listwise preference data — 2026-06-27

## Purpose

The focused categorical Gemma adapter and linear pairwise ranker both failed to produce a stable deployable signal.
This dataset changes the RLLM task shape: instead of predicting absolute labels independently, the model sees the
same signal-time context and must prefer one action among `LONG`, `SHORT`, and `NO_TRADE`.

## Generator

Script: `training/export_event_candidate_listwise_preference.py`

Inputs:

- Train candidates: `results/event_candidate_ranking_wavefull_ext_micro_c72_s2_train.jsonl`
- Eval candidates: `results/event_candidate_ranking_wavefull_ext_micro_c72_s2_eval.jsonl`

Outputs:

- Train preferences: `data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_train_2026-06-27.jsonl`
- Eval preferences: `data/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_eval_2026-06-27.jsonl`
- Summary: `results/event_candidate_listwise_pref_wavefull_ext_micro_c72_s2_2026-06-27/summary.json`

## Contract

- Prompt contains only signal-time state, selected price-action/micro/external features, event triggers, and
  LONG/SHORT candidate metadata.
- `chosen`/`rejected` use realized reward/utility for training only.
- Each preference row compares actions from the same signal timestamp.
- `NO_TRADE` has utility 0. A trade is chosen only when the best trade utility clears `min_trade_utility=0.25`.
- Pairs require `utility_gap>=0.25`.

## Dataset stats

| Split | Rows | Signals | Chosen NO_TRADE | Chosen LONG | Chosen SHORT | Mean utility gap | Mean prompt chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 12,507 | 6,880 | 2,871 | 5,130 | 4,506 | 4.15 | 1,837 |
| Eval | 3,431 | 1,907 | 799 | 1,344 | 1,288 | 3.01 | 1,830 |

## Dry-run check

`training.train_text_dpo` dry-run on 256 gate-balanced samples passed with Gemma 4 E4B config:

- chosen: NO_TRADE 128, LONG 71, SHORT 57
- rejected: NO_TRADE 71, LONG 90, SHORT 95
- prompt chars: 1,786..1,990, mean 1,839

## Decision

This is the next RLLM-compatible training surface. It better matches the trading decision than absolute categorical
reward labels because the model learns relative action preference at the same timestamp. Next step is a small Gemma
DPO PoC, then score chosen/rejected margins on held-out eval before any trading backtest.
