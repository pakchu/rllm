# Regime symbolic monthly selector threshold-cache — 2026-06-26

## Change

`training/regime_symbolic_monthly_selector.py` now fits/scoring the rolling symbolic expert policy once per `(month, target)` at the lowest requested threshold, then derives higher-threshold prediction files from cached `predicted_utility` rows.

This avoids repeating the expensive rolling expert fit for every threshold in the same target/month.

## Verification

Re-ran the prior reduced smoke with:

- history: `data/event_action_verifier_text_v3k8_2024_2025.jsonl`
- eval: `data/event_action_verifier_text_v3k8_2026_jan_may.jsonl`
- eval: 2026-01 through 2026-02
- target: `utility`
- thresholds: `0,0.001,0.003`

Artifacts:

- old: `results/regime_symbolic_monthly_selector_utility_smoke_2026-06-26/report.json`
- cached: `results/regime_symbolic_monthly_selector_utility_smoke_cached_2026-06-26/report.json`

Result equivalence:

- aggregate sim identical: 0 trades, 0.00% CAGR, 0.00% strict MDD
- month decisions identical:
  - 2026-01: `ABSTAIN`, selected `utility@0.0`, same reject reasons
  - 2026-02: `ABSTAIN`, selected `utility@0.003`, same reject reasons

Disk status after run: `/` remained around 292GB used, below the 300GB limit.

## Limitation

This only removes threshold-level duplicate fits. Different targets still need separate labels and separate fits. Broader target grids are now practical enough to retry, but further caching would be needed for very large grids.
