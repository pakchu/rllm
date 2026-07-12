# Extrema-arrival bidirectional alpha scan (2026-07-12)

- Causal features only: bars since trailing high/low event, rolling high/low hit counts, range position, recovery/fade speed and high-minus-low hit imbalance.
- Train `<2024` thresholds; test2024 selection; eval2025/YTD2026 reporting.
- 6bp/side, 0.5x, strict intrabar MDD, both directions required.

Result: 5,616 eligible variants, zero test/eval ratio>=2.5 qualifiers. Best balanced candidate `hit_imbalance_follow_576` had test/eval ratios 0.94/0.72 and failed 2026. This exact standalone family is rejected.

Artifacts:
- `training/search_extrema_arrival_bidirectional_alpha.py`
- `results/extrema_arrival_bidirectional_alpha_scan_2026-07-12.json`
