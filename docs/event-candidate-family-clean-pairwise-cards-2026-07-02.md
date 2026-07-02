# Clean pairwise family cards (2026-07-02)

## Why

The first randomized/listwise and pairwise family-card datasets still used the selector's pre-fold choice as the label.  That made the target too noisy: it taught the LLM to imitate the current heuristic selector, including its `rex_multiscale_location_revert` over-selection, instead of learning which pre-fold evidence patterns preceded a genuinely profitable next fold.

## Change

Added `training/build_event_candidate_family_clean_pairwise_cards.py`.

It builds DPO-compatible pairwise `A > B` training rows directly from the monthly selector report (with response fields `chosen` / `rejected` set to the preferred letter and option metadata stored separately as `chosen_option` / `rejected_option`):

1. **Prompt/options**: only `pre_fold_scoreboard` evidence plus explicit `position_state`.
2. **Label**: chosen from `top_fold_diagnostic_not_for_selection`, but only when the diagnostic family also existed in the pre-fold options and passed clean-label filters.
3. **Leakage guard**: target-fold metrics are kept in metadata as `diagnostic_target`, never in the prompt.
4. **DPO schema**: the trainable response is the letter only (`A` preferred over `B`), while option metadata is retained outside the response fields.
5. **Abstain**: if no diagnostic family passes filters, label is `ABSTAIN`.
6. **Order augmentation**: every pair is emitted twice, once with the clean winner as A and once with the clean winner as B. This prevents the LLM from learning an A-position shortcut.
7. **Randomization**: non-ABSTAIN option letters are shuffled per row to reduce option-position bias.

Default clean-label filters:

- `min_diagnostic_trades=12`
- `min_diagnostic_ratio=0.25`
- `min_diagnostic_cagr_pct=0.0`
- `max_diagnostic_mdd_pct=25.0`
- score used for label ranking: `cagr_to_strict_mdd * sqrt(min(trades, 30) / 30)`

The 12-trade threshold was selected over 8 because 8 still admitted more small-sample spike labels and more `location_revert` dominance; 12 keeps usable monthly labels while filtering many one-off winners.

## Generated split summaries

Source report:

- `results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json`

Generated files:

- `data/event_candidate_family_clean_pairwise_cards_rex_core_1m_train_2023_2024_random_2026-07-02.jsonl`
  - 24 folds, 144 order-augmented pairs
  - target families: `ABSTAIN=48`, `rex_compression_breakout=36`, `rex_compression_fakeout=24`, `rex_multiscale_location_revert=18`, `rex_htf_context_pullback_resume=12`, `rex_htf_pullback_resume=6`
  - chosen response balance: `A=72`, `B=72`
- `data/event_candidate_family_clean_pairwise_cards_rex_core_1m_test_2025_random_2026-07-02.jsonl`
  - 12 folds, 72 order-augmented pairs
  - target families: `ABSTAIN=12`, `rex_multiscale_location_revert=24`, `rex_compression_breakout=24`, `rex_compression_fakeout=12`
  - chosen response balance: `A=36`, `B=36`
- `data/event_candidate_family_clean_pairwise_cards_rex_core_1m_eval_2026h1_random_2026-07-02.jsonl`
  - 5 folds, 30 order-augmented pairs
  - target families: `rex_compression_fakeout=12`, `rex_htf_pullback_reclaim=6`, `ABSTAIN=6`, `rex_compression_breakout=6`
  - chosen response balance: `A=15`, `B=15`

Compared with the previous pairwise labels, clean-label train distribution is less dominated by `location_revert` and contains more `ABSTAIN`/compression cases.  Test still has `location_revert`, so this is not a final alpha claim; it is a cleaner supervised target for the next Gemma/LLM preference-tuning stage.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile training/build_event_candidate_family_clean_pairwise_cards.py tests/test_build_event_candidate_family_clean_pairwise_cards.py
PYTHONPATH=. .venv/bin/python - <<'PY'
from pathlib import Path
import tempfile
from tests.test_build_event_candidate_family_clean_pairwise_cards import (
    test_clean_target_uses_diagnostic_label_not_prefold_winner,
    test_clean_target_falls_back_to_abstain_when_no_valid_diagnostic,
    test_clean_pairwise_prompt_excludes_target_fold_metrics,
    test_clean_pairwise_run_writes_summary,
)
for fn in [
    test_clean_target_uses_diagnostic_label_not_prefold_winner,
    test_clean_target_falls_back_to_abstain_when_no_valid_diagnostic,
    test_clean_pairwise_prompt_excludes_target_fold_metrics,
    test_clean_pairwise_run_writes_summary,
]:
    with tempfile.TemporaryDirectory() as d:
        fn(Path(d))
print('manual clean pairwise tests passed')
PY
```

`pytest` could not run because this venv currently does not have `pytest` installed. After schema/order correction, generated JSONL was checked to confirm balanced `chosen` responses, `chosen_option` exists, and `diagnostic_target` is absent from prompts.

## Remaining risk

This dataset uses target-fold diagnostics to create supervised labels.  That is acceptable for train/test/eval label construction, but eval labels must not be used for model selection or strategy parameter tuning.  Any trading metric still requires a later chronological train -> test choose -> untouched eval backtest.
