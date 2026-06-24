# Fold regime decay signature audit (2026-06-24)

## Purpose

The current best model still misses the target mainly because of one bad traded fold: 2025Q3. This pass audits whether that fold had a test-start regime signature available before trading.

Implementation: `training/event_candidate_fold_regime_audit.py`.

## Base model

Current best before this audit:

- Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`
- CAGR: 15.02%
- Strict MDD: 14.10%
- CAGR/MDD: 1.07
- Trades: 119
- p approx: 0.066

## Audit result

Run:

```bash
.venv/bin/python -m training.event_candidate_fold_regime_audit \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --walkforward-report results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json \
  --output results/event_candidate_fold_regime_audit_decay45_sidescale_d0p5_2026-06-24.json \
  --pretest-days 14 --val-tail-days 14
```

The remaining bad fold, 2025Q3, has a distinct pre-test/validation signature:

- `pretest_range_pos`: 0.838 versus good fold mean 0.389
- `val_tail_range_pos`: 0.838 versus good fold mean 0.389
- `val_full_range_pos`: 0.869 versus good fold mean 0.386
- `val_full_ann_5m_vol_proxy`: 0.022 versus good fold mean 0.029
- `val_full_max_drawdown_pct`: 15.1 versus good fold mean 25.8

Interpretation: the model decayed after a strong/quiet upper-range regime. It kept taking trades after a trend/runup regime where the previous relation stopped paying.

## Diagnostic ceiling check

A diagnostic abstain on only that fold gives:

- Output: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_regime_diag_2026-06-24/backtest.json`
- CAGR: 17.12%
- Strict MDD: 12.54%
- CAGR/MDD: 1.36
- Trades: 91
- p approx: 0.028

This is **not valid final evidence** because the threshold/signature was chosen after seeing the test fold outcome. It is only an upper-bound diagnostic showing that regime decay detection is worth pursuing.

## Next validation requirement

The next test must choose any regime gate from prior completed folds only, then apply it to later folds. A gate selected using the 2025Q3 test outcome cannot count toward no-leak performance.
