# Online prior-fold regime gate validation (2026-06-24)

## Purpose

The fold-regime audit identified a diagnostic upper-range quiet-regime signature for the 2025Q3 loss. This pass tests whether such a gate can be selected without leakage, using only prior completed folds.

Implementation: `training/event_candidate_online_regime_gate.py`.

## Protocol

For each fold:

1. Candidate threshold rules are scored only on earlier traded folds with completed outcomes.
2. The selected rule is applied to the current fold using metrics known before the current test period.
3. All original fold prediction files, including abstain folds, are preserved in the aggregate backtest.

Candidate rule family:

- Features: `pretest_range_pos`, `val_full_range_pos`, `val_tail_range_pos`
- Thresholds: `0.70,0.75,0.80,0.85`
- Direction: abstain if feature is above threshold

## Result with strict prior improvement requirement

Run uses `--min-prior-improvement 0.01`.

Report: `results/event_candidate_online_regime_gate_decay45_sidescale_d0p5_strict_2026-06-24/report.json`

- Abstained folds: none
- CAGR: 15.02%
- Strict MDD: 14.10%
- CAGR/MDD: 1.07
- Trades: 119

This equals the current best base model because the gate cannot be legitimately selected before 2025Q3.

## Important invalid/tie-break case

If `--min-prior-improvement 0` is allowed, the code can choose a threshold with zero prior improvement due tie-break ordering and abstain 2025Q3:

- CAGR: 17.12%
- Strict MDD: 12.54%
- CAGR/MDD: 1.36
- p approx: 0.028

This is not acceptable as final evidence. Before 2025Q3, prior completed folds contain no bad example under the current best model, so the selected threshold is not learned from negative prior evidence.

## Conclusion

The upper-range signature is useful diagnostically, but current fold history is too sparse for a no-leak online meta-gate to learn it. The valid current best remains:

- `PA-ext + 6M/3M/3M + stats gate + pair half-life 45d + light side scaling`
- CAGR 15.02%, strict MDD 14.10%, CAGR/MDD 1.07, p approx 0.066

Next work should either increase the number of meta-validation folds or improve the base alpha directly; do not count the tie-break regime gate as a real result.
