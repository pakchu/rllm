# Relation-flip audit and online risk overlay (2026-06-24)

## Purpose

The PA-ext 6M/3M/3M walk-forward finally turned aggregate return positive but still had excessive MDD. This pass asked two questions:

1. Are losing folds explained by simple feature/reward sign flips?
2. Can a live-usable completed-trade risk overlay cut regime decay losses without using future labels?

## Relation-flip audit

Implementation: `training/event_candidate_relation_flip_audit.py`.

Run:

```bash
.venv/bin/python -m training.event_candidate_relation_flip_audit \
  --input-jsonl data/event_action_compressor_ranker_all_2022_2026_paext_2026-06-24.jsonl \
  --walkforward-report results/event_candidate_pairwise_walkforward_paext_6m3m3m_statsgate_2026-06-24/report.json \
  --output results/event_candidate_relation_flip_audit_paext_6m3m3m_statsgate_2026-06-24.json \
  --top-n-features 80 --min-rows 500 --quantile 0.2
```

Findings:

- Across all fold-feature checks:
  - fit→val IC flips: 322
  - val→test IC flips: 363
  - fit→val spread flips: 410
  - val→test spread flips: 471
- Simple sign-flip counts do not cleanly separate winners from losers.
- Some losing folds had many val→test flips, but 2025Q3 lost badly even though the top relations mostly retained sign and only decayed in magnitude.

Conclusion: relation flip is part of the failure, but not sufficient. The live system also needs fast loss containment for relation decay.

## Online risk overlay

Input predictions: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_statsgate_2026-06-24/*/fold*_test_predictions.jsonl`.

Run:

```bash
.venv/bin/python -m training.online_risk_overlay_backtest \
  --predictions-jsonl <fold prediction files comma-separated> \
  --market-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/event_candidate_pairwise_walkforward_paext_6m3m3m_statsgate_online_overlay_2026-06-24.json \
  --leverage 1.0 --entry-delay-bars 1 \
  --pause-after-losses 2 --pause-bars 864 \
  --rolling-window-trades 10 --rolling-loss-stop-pct 8 --rolling-drawdown-stop-pct 8 \
  --monthly-loss-stop-pct 6 \
  --trade-stop-loss-pct 4
```

Result:

| Variant | Trades | CAGR | Strict MDD | CAGR/MDD | Mean trade | p approx |
|---|---:|---:|---:|---:|---:|---:|
| PA-ext 6M/3M/3M stats-gate | 126 | 5.39% | 17.36% | 0.31 | +0.183% | 0.410 |
| + online risk overlay | 74 | 5.92% | 11.96% | 0.49 | +0.316% | 0.290 |

The overlay uses only completed prior trades and current/previous trade path stops; it does not inspect future outcomes before deciding to pause.

## Conclusion

This is not enough for the target, but it is the first path that satisfies strict MDD < 15 on the broad rolling OOS run. The remaining gap is return/edge strength, not drawdown containment alone.

Next work should improve the adaptive ranker alpha while keeping this online overlay as a risk layer candidate.
