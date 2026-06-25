# Monthly selector inversion audit (2026-06-25)

## Goal
Quantify the repeated validation-spike → next-period inversion pattern and test whether validation spikes can be used as an abstention signal.

## Inversion audit
Script: `training/monthly_selector_inversion_audit.py`  
Report: `results/monthly_selector_inversion_audit_2026-06-25/report.json`

Using the 2026 monthly symbolic selector reports:
- eval months with trades: 5
- validation-positive / eval-negative cases: 4
- validation-positive / eval-positive cases: 1
- validation CAGR → eval CAGR correlation: -0.814
- validation ratio → eval ratio correlation: -0.683

Interpretation: for this selector, strong validation performance is not persistence; it is often a warning sign.

## Anti-persistence overlay diagnostic
Script: `training/monthly_selector_anti_persistence_overlay.py`

Rules tested on fixed selector predictions:
- block months with validation ratio above threshold, or validation CAGR/t-stat spike

Results:
- Blocking only the extreme April validation spike: CAGR 4.98%, strict MDD 9.05%, 81 trades, ratio 0.55
- Blocking validation ratio > 5: CAGR 8.61%, strict MDD 6.81%, 45 trades, ratio 1.26

This improves the failed baseline but remains far below target and statistically weak.

## Conclusion
Validation spike detection is useful as a risk warning, not alpha. It can reduce damage, but it does not create enough positive expectancy.

## Next validation required
The anti-persistence rule itself must be validated on pre-2026 selector reports before it can be used as a standing rule. Otherwise it is another 2026-specific repair.
