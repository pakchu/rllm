# Pre-2026 anti-persistence validation (2026-06-25)

## Goal
Validate whether the 2026 observation — validation spikes often invert next month — also appears before 2026.

## Report parsing
`monthly_selector_inversion_audit.py` was generalized to parse both:
- monthly symbolic selector reports (`selected.backtest`, `eval.backtest`)
- feature-filter walk-forward reports (`selected.val`, `eval`)

## Pre-2026 feature-filter inversion audit
Input: `results/pre2026_feature_filter_validation_2026-06-25/filter_wf_2025h2_p02_report.json`  
Output: `results/pre2026_feature_filter_inversion_audit_2026-06-25/report.json`

Summary:
- eval months: 5
- validation-positive / eval-negative: 4
- validation-positive / eval-positive: 1
- mean eval CAGR when validation positive: -39.45%

Examples:
- 2025-09: validation CAGR 107.6%, eval CAGR -56.0%
- 2025-11: validation CAGR 215.6%, eval CAGR -42.0%
- 2025-12: validation CAGR 26.3%, eval CAGR -81.5%

## Anti-persistence overlay on 2025H2
Input predictions: `results/pre2026_feature_filter_validation_2026-06-25/filter_wf_2025h2_p02_work/combined_predictions.jsonl`

Blocking validation ratio spikes:
- ratio > 3: CAGR -4.63%, strict MDD 6.60%, 48 trades
- ratio > 5/10/20: CAGR -31.63%, strict MDD 16.71%, 109 trades

## Conclusion
Anti-persistence is not a 2026-only artifact. It also appears in pre-2026 feature-filter reports. However, it only reduces damage; it does not create a profitable strategy.

## Implication
The current system has a useful abstention warning: validation spikes are often unstable. But a separate positive alpha source is still required. The next search should focus on new alpha generation, not on further gates over the current weak pool.
