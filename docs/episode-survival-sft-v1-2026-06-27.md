# Episode survival SFT v1 (2026-06-27)

## Purpose

Direct symbolic execution rules failed long-history robustness. The next RLLM step is therefore not to train a model to execute event rules directly, but to train a compact survival filter:

> Given a causal setup/context and one candidate event-side-horizon, decide whether the trade should be `TRADE` or `NO_TRADE` based on path survival.

The target is derived from future path labels for offline training only. The prompt contains no future path fields.

## Export command

```bash
.venv/bin/python -m training.export_episode_survival_sft_data \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output-dir data/episode_survival_sft_v1_2026-06-27 \
  --train-start 2020-01-01 --train-end '2023-12-31 23:59:59' \
  --test-start 2024-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --windows 576,2016,4032 \
  --horizons 72,144,288 \
  --event-types failed_breakdown_long,low_sweep_reclaim,reclaim_mid_from_below,failed_mid_loss_long,downtrend_pullback_reject,failed_mid_reclaim_short \
  --min-trade-net-pct 0.25 \
  --max-trade-mae-pct 2.0 \
  --min-mfe-to-mae 1.25 \
  --mae-penalty 0.2 \
  --max-negative-per-positive 4 \
  --max-rows-per-split 20000
```

## Outputs

Directory size: 2.8MB.

| Split | Raw rows | Sampled rows | TRADE | NO_TRADE | LONG | SHORT | File |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Train 2020-2023 | 76,956 | 20,000 | 10,000 | 10,000 | 12,020 | 7,980 | `data/episode_survival_sft_v1_2026-06-27/episode_survival_train.jsonl.gz` |
| Test 2024-2025 | 41,274 | 20,000 | 10,000 | 10,000 | 12,129 | 7,871 | `data/episode_survival_sft_v1_2026-06-27/episode_survival_test.jsonl.gz` |
| Eval 2026-06 | 8,304 | 8,304 | 2,520 | 5,784 | 4,959 | 3,345 | `data/episode_survival_sft_v1_2026-06-27/episode_survival_eval.jsonl.gz` |

## Prompt fields

Each prompt includes only causal inputs:

- candidate: event, event_type, episode, side, horizon;
- setup_quality:
  - risk bucket/value;
  - range bucket/value;
  - body bucket/value;
  - favorable wick bucket/value;
  - close quality bucket/value;
- macro_context:
  - `dxy_z`, `usdkrw_z`, `kimchi_z`, `kimchi_chg`.

Quantile buckets are fit on train only and reused for test/eval.

## Target label

Target JSON contains only:

```json
{"decision":"TRADE|NO_TRADE","confidence":"HIGH|MID|LOW","reason":"..."}
```

`target_audit` stores numeric label evidence for diagnostics:

- net pct;
- MAE pct;
- MFE pct;
- MFE/MAE;
- utility pct.

The target is `TRADE` only if:

- net return >= 0.25%;
- MAE <= 2.0%;
- MFE/MAE >= 1.25;
- utility = net - 0.2 * MAE is positive.

## Decision

1. This is the first useful RLLM-compatible dataset after the symbolic rule failures.
2. It should be used to train/evaluate a small Gemma survival classifier before any RL portfolio stage.
3. The immediate model metric should be eval precision/recall for `TRADE`, plus backtest of model-accepted candidates.
4. Do not treat this as final trading alpha. It is a filtering pretext task designed to teach path-risk survival and abstention.
