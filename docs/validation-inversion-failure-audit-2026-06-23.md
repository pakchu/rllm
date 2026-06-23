# Validation inversion failure audit (2026-06-23)

## Purpose

The h288 pairwise event-context ranker showed a suspicious 2026 behavior: the selected side often lost while the opposite side looked better. This audit checks whether that is a deployable causal inversion rule or only an eval-period symptom.

## Added tools

- `training/event_side_failure_audit.py`
  - Audits chosen side, inverted same-entry side, and oracle best same-entry side by period.
  - Groups failures by event/state tokens.
  - Audit-only: realized labels are used only after predictions exist and are not a selector.
- `training/nested_validation_inversion_selection.py`
  - Tests causal prior-validation-score gates that can `pass`, `block`, or `invert` generated sides.
  - Ranks threshold/action/overlay configs on 2024-01 to 2025-12 only.
  - Replays top configs on untouched 2026-01 to 2026-05 eval.
  - Deletes per-candidate intermediate files by default to avoid WSL disk growth; use `--keep-artifacts` only when debugging.

## Side failure audit

Inputs:

- h288 predictions: `results/rolling_event_context_preference_predictions_2026-06-23.jsonl`
- h288 context rows: `data/llm_context_regime_events_2026-06-23.jsonl`
- output: `results/event_side_failure_audit_h288_2026-06-23.json`

Key h288 findings:

| Period | Trades | Chosen mean | Inverted same-entry mean | Oracle best mean | Matched best | Opposite best | Wait best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-01..2025-12 selection | 1,154 | +0.022% | -0.122% | +0.858% | 575 | 523 | 56 |
| 2026-01..2026-05 eval | 201 | -0.137% | +0.037% | +0.865% | 92 | 99 | 10 |

Interpretation:

- Selection period does **not** support global inversion; inverted same-entry performance is worse.
- Eval period does show direction collapse; opposite side is slightly positive on average.
- The model is not finding stable side alpha. It is sometimes finding useful entry timing, but side mapping breaks across regimes.

## Nested validation inversion test

Full nested command used for h288:

```bash
.venv/bin/python -m training.nested_validation_inversion_selection \
  --predictions-jsonl results/rolling_event_context_preference_predictions_2026-06-23.jsonl \
  --rolling-summary-json results/rolling_event_context_preference_summary_2026-06-23.json \
  --market-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/nested_validation_inversion_selection_h288_2026-06-23.json \
  --work-dir results/nested_validation_inversion_h288_2026-06-23 \
  --top-k 20
```

Selection/eval split:

- Selection: 2024-01 through 2025-12, 2,924 rows.
- Eval: 2026-01 through 2026-05, 598 rows.
- Candidate configs ranked only on selection.

Top clean selection result:

| Config | Selection trades | Selection CAGR | Selection strict MDD | Selection ratio | Eval trades | Eval CAGR | Eval strict MDD | Eval ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold `0.5`, below=`block`, above=`pass`, TP `3%` | 316 | 28.70% | 12.81% | 2.24 | 0 | 0.00% | 0.00% | 0.00 |

Best notable inversion config inside top clean selection set:

| Config | Selection trades | Selection CAGR | Selection strict MDD | Selection ratio | Eval trades | Eval CAGR | Eval strict MDD | Eval ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold `-500`, below=`invert`, above=`pass`, TP `3%` | 418 | 26.51% | 11.92% | 2.22 | 80 | 14.29% | 10.22% | 1.40 |

## Decision

This is **not** a target-achieving solution.

What was learned:

1. The 2026 side inversion is real enough to reduce damage when selected causally by prior validation score.
2. It does not meet the target: eval ratio is 1.40, not 3+, and CAGR is 14.29%, not 50%.
3. The strongest selection rule still prefers blocking bad-validation months, which produces zero 2026 trades.
4. The core failure remains side/market-regime non-stationarity, not just risk overlay tuning.

Next research direction:

- Stop treating side inversion as a fixed rescue rule.
- Build a causal regime classifier whose label is not same-month PnL but prior-window side-map reliability.
- Feed that reliability state into the RLLM context as an explicit uncertainty/action-space token: `side_map=normal|inverse|unreliable`, where `unreliable` can force WAIT.
