# Monthly family state-card expansion (2026-07-02)

## Why

The first position-aware state-card export used 6-month folds and produced only 7 rows.  That is too small for any meaningful Gemma/LLM fine-tune.  This pass expands the scoreboard/state-card surface to monthly folds and exports chronological train/test/eval JSONL splits.

## Monthly selector result

Report: `results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json`

Final stitched replay:

| CAGR | strict MDD | CAGR/MDD | trades | p-value |
| ---: | ---: | ---: | ---: | ---: |
| 3.85% | 17.94% | 0.21 | 412 | 0.523 |

Interpretation: monthly folds increase sample count but make family selection noisier.  This is not a better strategy than the 6-month abstention replay.  Use it as prompt/data expansion only.

## Exporter update

`training/build_event_candidate_family_state_cards.py` now supports date filtering:

- `--fold-start` inclusive;
- `--fold-end` exclusive.

This allows chronological train/test/eval JSONL splits from one selector report.

## Generated state-card splits

Commands:

```bash
.venv/bin/python -m training.build_event_candidate_family_state_cards \
  --selector-report results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json \
  --output-jsonl data/event_candidate_family_state_cards_rex_core_1m_train_2023_2024_2026-07-02.jsonl \
  --max-options 5 --split-name train_2023_2024 --fold-start 2023-01-01 --fold-end 2025-01-01

.venv/bin/python -m training.build_event_candidate_family_state_cards \
  --selector-report results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json \
  --output-jsonl data/event_candidate_family_state_cards_rex_core_1m_test_2025_2026-07-02.jsonl \
  --max-options 5 --split-name test_2025 --fold-start 2025-01-01 --fold-end 2026-01-01

.venv/bin/python -m training.build_event_candidate_family_state_cards \
  --selector-report results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json \
  --output-jsonl data/event_candidate_family_state_cards_rex_core_1m_eval_2026h1_2026-07-02.jsonl \
  --max-options 5 --split-name eval_2026h1 --fold-start 2026-01-01 --fold-end 2026-06-01
```

Outputs:

| split | rows | targets |
| --- | ---: | --- |
| train 2023-2024 | 24 | `A=18`, `ABSTAIN=6` |
| test 2025 | 12 | `A=7`, `ABSTAIN=5` |
| eval 2026H1 | 5 | `A=4`, `ABSTAIN=1` |

Every row includes explicit `position_state` and every prompt includes `Current position:`.

## Verification

- `py_compile` passed for exporter/test.
- Manual tests passed for:
  - explicit position state;
  - JSONL writing;
  - fold date filtering.
- Generated monthly split files were read back and checked for `position_state` and `Current position:`.

## Next action

The monthly fold dataset is still small and target labels are dominated by option `A` because the selector's top option is usually the chosen option.  Before SFT, generate richer listwise examples by either:

1. randomizing option order to remove option-position bias; or
2. creating pairwise chosen/rejected records from the scoreboard; and
3. expanding beyond fold-level records into per-signal/per-week state cards if needed.
