# Negative sweeps after current best protocol (2026-06-24)

## Current valid best

`PA-ext + 6M/3M/3M + stats gate + pair half-life 45d + light side scaling`

- Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`
- CAGR: 15.02%
- Strict MDD: 14.10%
- CAGR/MDD: 1.07
- Trades: 119
- p approx: 0.066

## Sweeps run after best

### More meta-folds

Goal: get more bad/good fold examples for regime-gate learning.

| Protocol | CAGR | Strict MDD | CAGR/MDD | Trades | Result |
|---|---:|---:|---:|---:|---|
| 4M fit / 2M val / 2M test | -5.62% | 50.23% | -0.11 | 143 | fail |
| 6M fit / 2M val / 2M test | -5.27% | 47.87% | -0.11 | 137 | fail |

Conclusion: shorter validation/test cadence destabilizes selection; fold count cannot be increased this way.

### Max hold cap on current best predictions

| Max hold bars | CAGR | Strict MDD | CAGR/MDD | Trades |
|---:|---:|---:|---:|---:|
| 72 | -4.74% | 25.24% | -0.19 | 192 |
| 144 | -2.59% | 22.62% | -0.11 | 175 |
| 216 | 2.22% | 19.13% | 0.12 | 159 |
| 288 | 7.00% | 19.34% | 0.36 | 139 |
| 360 | 12.25% | 16.21% | 0.76 | 131 |

Conclusion: the alpha currently depends on long holds; capping hold length destroys edge.

### Stop/take-profit diagnostic scan on fixed best predictions

Best diagnostic post-hoc result was `take_profit=8`:

- CAGR: 19.32%
- Strict MDD: 14.10%
- CAGR/MDD: 1.37
- p approx: 0.022

This is not final evidence because the exit parameter was chosen on aggregate OOS.

### Fixed take-profit inside walk-forward validation selection

After adding fixed exit parameters to the walk-forward runner, TP=8 was applied consistently to validation selection and test backtests.

- Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_tp8_2026-06-24/report.json`
- CAGR: 9.17%
- Strict MDD: 29.16%
- CAGR/MDD: 0.31
- Trades: 141

Conclusion: TP=8 changes validation q/margin choices and opens new bad folds; the post-hoc TP result should not be trusted.

### High-quantile candidate selection

- Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_hiq_2026-06-24/report.json`
- CAGR: 14.04%
- Strict MDD: 19.42%
- CAGR/MDD: 0.72
- Trades: 131

Conclusion: high-q reduces some exposure but opens other weak folds; not better than current best.

### Stricter validation evidence gates

- Reports: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_valratio*_2026-06-24/report.json`
- CAGR: 1.54%
- Strict MDD: 27.53%
- CAGR/MDD: 0.06
- Trades: 106

Conclusion: stronger validation filters over-prune good folds and do not control drawdown.

## Code change

`training/event_candidate_pairwise_walkforward.py` now supports fixed risk-exit parameters:

- `--trade-stop-loss-pct`
- `--trade-take-profit-pct`

These are applied consistently to validation selection, fold test backtests, and aggregate backtest.

## Overall conclusion

The current best remains unchanged. Recent failures indicate that simple execution knobs and stricter selection gates are not enough. The next useful alpha work should add new information or model structure, not keep sweeping gates.
