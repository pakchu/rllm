# Online episode template gate audit (2026-06-27)

## Purpose

Static train/test template selection repeatedly failed 2026. This pass tested a causal online paper-performance gate:

- fixed templates come from a pre-existing `price_action_episode_policy` report;
- every selected template runs hypothetical paper trades;
- paper outcomes enter the rolling ledger only after their hypothetical exit;
- a live eval trade is allowed only when that same template has enough recent paper trades and positive recent expectancy;
- eval trade decisions therefore use only prior outcomes.

Script: `training/online_episode_template_gate.py`

Policy source:

- `results/price_action_episode_policy_wavefull_seqmacro_train2024_test2025_eval2026jm_2026-06-27/report.json`

## Results

All runs warm up from 2024-2025 and trade only from 2026-01-01 through 2026-06-01.

| gate | CAGR | strict MDD | ratio | trades | side | skipped | p-value |
|---|---:|---:|---:|---:|---|---:|---:|
| min 5 / lookback 20 / mean >= 0 / loss <= 0.65 | -16.46 | 11.50 | -1.43 | 60 | LONG 58 / SHORT 2 | 252 | 0.4293 |
| min 8 / lookback 30 / mean >= 0.05 / loss <= 0.60 | -31.93 | 15.71 | -2.03 | 56 | LONG 54 / SHORT 2 | 286 | 0.0341 |
| min 10 / lookback 40 / mean >= 0.10 / loss <= 0.55 | -19.09 | 10.21 | -1.87 | 41 | LONG 38 / SHORT 3 | 385 | 0.2492 |
| min 5 / lookback 20 / mean >= 0.20 / loss <= 0.55 | -19.02 | 9.52 | -2.00 | 39 | LONG 37 / SHORT 2 | 406 | 0.2022 |

## Diagnosis

Online paper-performance gating reduces trade count but does not recover 2026 profitability. The selected templates remain dominated by long entries, while short entries stay too sparse to hedge or profit. Stricter gates cut exposure but still allow the wrong long trades through.

This narrows the failure mode:

1. The problem is not only static selection.
2. Recent same-template paper expectancy is not a sufficient regime detector.
3. The missing component is likely broader market-regime state, not per-template recent PnL alone.

## Next direction

Move from per-template gate to explicit market-regime classifier:

- classify market as trend-up / trend-down / chop / crash-recovery / squeeze using higher timeframe returns, realized volatility, DXY/Kimchi/funding/OI state;
- allow long dip-buying only in regimes where it historically survives;
- search for short templates separately inside trend-down / macro-risk regimes;
- use the classifier output as the compact LLM context later, not raw numeric arrays.
