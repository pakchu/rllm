# Pairwise event-candidate walk-forward validation (2026-06-24)

## Purpose

Previous direct LLM/Q-code selectors and one-shot ranker validation were too easy to overfit to a single validation period. This pass adds a stricter no-leakage rolling protocol:

1. Train a pairwise winner-vs-loser candidate ranker on a past fit window.
2. Select only `score quantile` and `full-size margin` on the next validation window.
3. Trade the following test window only if validation passes evidence gates.
4. Refit on `fit + validation` before the test window, because those labels are known before live deployment.
5. Aggregate only fold test predictions.

Implementation: `training/event_candidate_pairwise_walkforward.py`.

## Verification run

Command shape:

```bash
.venv/bin/python -m training.event_candidate_pairwise_walkforward \
  --input-jsonl data/event_action_compressor_ranker_all_2022_2026_2026-06-24.jsonl \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --fit-months 12 --val-months 6 --test-months 6 --step-months 6 \
  --quantiles 0.85,0.90,0.95 --full-margins 0,0.5,1.0 \
  --min-fit-signals 200 --min-val-trades 20 --min-test-signals 50 \
  --min-val-cagr-pct 0 --min-val-ratio 0 --max-val-strict-mdd-pct 25 \
  --epochs 80 --leverage 1.0 --entry-delay-bars 1
```

Report: `results/event_candidate_pairwise_walkforward_2026-06-24/report.json`.

Aggregate OOS result over fold test windows:

- Trades: 166
- CAGR: -15.15%
- Strict MDD: 49.43%
- CAGR / strict MDD: -0.31
- Mean trade return: -0.260%
- p-value approximation: 0.155

Fold summary:

| Fold test window | Gate | Validation ratio | Test CAGR | Test strict MDD | Test ratio | Test trades |
|---|---:|---:|---:|---:|---:|---:|
| 2023-07-01..2024-01-01 | trade | 0.53 | -11.92% | 13.98% | -0.85 | 44 |
| 2024-01-01..2024-07-01 | trade | 4.11 | -59.76% | 49.28% | -1.21 | 51 |
| 2024-07-01..2025-01-01 | abstain | -1.61 | n/a | n/a | n/a | 0 |
| 2025-01-01..2025-07-01 | trade | 0.58 | 30.16% | 9.61% | 3.14 | 35 |
| 2025-07-01..2026-01-01 | trade | 2.94 | -22.38% | 13.72% | -1.63 | 36 |
| 2026-01-01..2026-05-30 | abstain | -1.34 | n/a | n/a | n/a | 0 |

A stricter statistics-gated run (`--min-val-t-stat 1.0 --max-val-p-value 0.25 --max-val-power-gap 250`) did not fix the issue:

- Report: `results/event_candidate_pairwise_walkforward_statsgate_2026-06-24/report.json`
- Trades: 87
- CAGR: -18.01%
- Strict MDD: 54.51%
- CAGR / strict MDD: -0.33
- Mean trade return: -0.621%
- p-value approximation: 0.043, but in the wrong direction.

## Conclusion

The rolling protocol validates the failure mode rather than solving it. Strong validation windows still invert or decay in the next OOS window. This means the current feature/ranker stack is not capturing a stable causal edge; it is finding transient regime coincidences.

The best fold (`2025-01-01..2025-07-01`) meets the target locally, but it is surrounded by failed folds and cannot be treated as a deployable model.

## Next implication

Do not keep optimizing the same gate grid. The next useful work is feature-side alpha discovery with regime-stability tests before returning to LLM compression/fine-tuning. Candidate features should be accepted only if their sign/rank relation survives rolling fit/validation/test windows, not just one static split.
