# Episode survival-quality audit (2026-06-27)

## Purpose

The previous strict backtests showed that event names and simple setup-quality buckets are not enough. This audit reframes the problem in the direction needed for RLLM: can causal setup-quality buckets predict executable path survival?

A candidate is evaluated with future path labels only as labels:

- delayed next-open entry;
- fixed hold path;
- net return after fees/slippage/leverage;
- MAE and MFE from actual high/low path;
- survival rate: net positive, MAE below threshold, and useful MFE/MAE.

Inputs remain causal signal-bar/setup attributes. Bucket thresholds are fit on train triggers only.

## Strict survival audit

Command:

```bash
.venv/bin/python -m training.audit_episode_survival_quality \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/episode_survival_quality_nonseq_train2020_2023_test2024_2025_2026-06-27/report.json \
  --train-start 2020-01-01 --train-end '2023-12-31 23:59:59' \
  --test-start 2024-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --windows 288,576,2016,4032 \
  --horizons 72,144,288,432 \
  --min-split-triggers 40 \
  --max-survival-mae-pct 2.0 \
  --min-mfe-to-mae 1.25 \
  --top-k 100
```

Results:

- Feature columns: 116
- Candidates: 5,120
- Robust train/test count: 0

Top candidates were not robust because either train utility stayed negative, test sample count was too small, or eval collapsed. Example:

| Rule | Train mean net / MAE / survival / utility | Test mean net / MAE / survival / utility | Eval diagnostic |
| --- | --- | --- | --- |
| `pae_w4032_downtrend_pullback_reject@144:range_bps=mid` SHORT | 0.043 / 1.932 / 0.385 / -0.923 | 0.206 / 0.883 / 0.543 / -0.236 | eval utility 0.085, but train/test utility fail |
| `pae_w4032_low_sweep_reclaim@72:risk_bps=low` LONG | -0.312 / 2.877 / 0.377 / -1.750 | 0.250 / 1.182 / 0.524 / -0.341 | train/test utility fail |

## Loose sensitivity check

Command used looser survival assumptions:

```bash
.venv/bin/python -m training.audit_episode_survival_quality \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/episode_survival_quality_nonseq_train2020_2023_test2024_2025_loose_2026-06-27/report.json \
  --train-start 2020-01-01 --train-end '2023-12-31 23:59:59' \
  --test-start 2024-01-01 --test-end '2025-12-31 23:59:59' \
  --eval-start 2026-01-01 --eval-end 2026-06-01 \
  --windows 288,576,2016,4032 \
  --horizons 72,144,288,432 \
  --min-split-triggers 40 \
  --max-survival-mae-pct 3.0 \
  --min-mfe-to-mae 1.0 \
  --mae-penalty 0.2 \
  --top-k 20
```

After fixing robust selection to require test `n >= min_split_triggers`, robust train/test count was still 0.

The apparent top candidate before sample filtering was `pae_w4032_failed_breakdown_long@72:range_bps=high`, but it had only 10 test trades and 5 eval trades, then eval utility was negative. It is not statistically useful.

## Decision

1. Current episode surface does not contain a robust direct symbolic survival rule under long-history train/test validation.
2. The failure is not just PnL; MAE/utility survival also fails. This supports moving RLLM labels away from raw event selection and toward path-survival classification.
3. For LLM/RLLM training, the label should be a path-survival/abstention target, not direct fixed event execution.
4. Next implementation should generate a compact text dataset where each example includes:
   - setup-quality descriptors;
   - candidate side/horizon;
   - causal regime/context;
   - target survival class derived from net/MAE/MFE utility.
5. The model should first learn `TRADE` vs `NO_TRADE` survival filtering before being asked to optimize portfolio-level returns.
