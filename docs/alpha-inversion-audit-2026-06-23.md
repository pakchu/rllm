# Alpha inversion audit — 2026-06-23

## Purpose

Several prior RLLM failures looked like direction/sign instability. Before
building new LLM/RL policy layers, this audit checks whether the top rolling alpha
candidates simply need sign inversion.

## Method

Tool: `training.alpha_inversion_audit`

For each selected rolling candidate:

1. rebuild the causal market/wave feature frame,
2. for each chronological fold, fit the original quantile rule using data before
   that fold only,
3. replay the original rule with strict bar-by-bar MDD,
4. replay an inverted rule where `LONG`/`SHORT` side mapping is swapped,
5. pass both variants through `training.alpha_candidate_gate`.

The inversion is diagnostic only. A production inversion selector would need its
own no-leak train/test/eval selection protocol.

## Current run

Command:

```bash
.venv/bin/python -m training.alpha_inversion_audit \
  --input-report results/rolling_alpha_feature_discovery_report.json \
  --input-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/alpha_inversion_audit_rolling_top6_2026-06-23.json \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --external-tolerance 30min \
  --max-candidates 6 \
  --leverage 1.0
```

Result:

- candidates checked: `6`
- inverted candidates passed: `0`
- decision: `NO_GO`

Representative examples:

| candidate | original positive folds | inverted positive folds | original worst CAGR | inverted worst CAGR | original worst MDD | inverted worst MDD |
|---|---:|---:|---:|---:|---:|---:|
| `mkt__usdkrw_zscore`, h144 | 4/7 | 2/7 | -77.76% | -79.19% | 54.94% | 61.18% |
| `mkt__htf_1w_return_4`, h288 | 3/7 | 0/7 | -14.79% | -47.56% | 18.67% | 29.10% |
| `mkt__htf_1w_return_1`, h288 | 2/7 | 1/7 | -41.13% | -59.48% | 26.12% | 51.04% |

## Interpretation

Simple sign inversion does not rescue the current alpha pool. The failure is not
just a one-bit direction mistake; the candidates are structurally unstable and
high-drawdown across folds.

Next search should broaden the alpha source/environment, not invert existing
feature rules or fine-tune LLMs on these labels.
