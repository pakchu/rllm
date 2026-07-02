# Randomized family state-card options (2026-07-02)

## Why

The monthly state-card split expanded rows but target completions were position-biased: most chosen options were `A` because the selector's best family was emitted first.  This is the same label-position bias that previously broke listwise/binary LLM experiments.  Before any SFT, option order must be randomized and ids reassigned.

## Implementation

`training/build_event_candidate_family_state_cards.py` now supports:

- `--randomize-options`
- `--random-seed`

Behavior:

- shuffles non-`ABSTAIN` options per record using `random_seed + record_index`;
- reassigns option ids after shuffling;
- keeps `ABSTAIN` id stable as `ABSTAIN`;
- recomputes the target id from selected family after shuffling;
- marks `leakage_guard.option_order_randomized=true`.

## Verification

Manual tests passed:

- explicit `position_state` remains in prompt;
- JSONL writer works;
- fold date filtering works;
- randomized option order preserves selected family but moves the target away from fixed `A` under seeded shuffle.

## Generated randomized splits

Commands used the monthly scoreboard report:

`results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_1m_2023_2026h1_2026-07-02.json`

Outputs:

| split | output | rows | target ids |
| --- | --- | ---: | --- |
| train 2023-2024 | `data/event_candidate_family_state_cards_rex_core_1m_train_2023_2024_random_2026-07-02.jsonl` | 24 | `A=3`, `B=3`, `C=3`, `D=5`, `E=4`, `ABSTAIN=6` |
| test 2025 | `data/event_candidate_family_state_cards_rex_core_1m_test_2025_random_2026-07-02.jsonl` | 12 | `A=3`, `B=1`, `C=1`, `E=2`, `ABSTAIN=5` |
| eval 2026H1 | `data/event_candidate_family_state_cards_rex_core_1m_eval_2026h1_random_2026-07-02.jsonl` | 5 | `A=1`, `D=2`, `E=1`, `ABSTAIN=1` |

Every row was read back and verified to include `position_state`, `Current position:`, and randomized-order leakage guard.

## Remaining issue

Option-position bias is fixed, but family-label quality is still weak:

- train target families include `rex_multiscale_location_revert=10/24`;
- eval target families include `rex_multiscale_location_revert=4/5`;
- monthly selector replay itself is weak and should not be considered deployable.

Before SFT, the target generation should be improved so the LLM learns robust veto/abstain behavior rather than memorizing noisy selector choices.

## Next action

Build pairwise chosen/rejected records from randomized state cards or derive cleaner targets from fold diagnostic outcomes with strict train/test/eval separation.  Prefer pairwise/DPO-style data because it better fits LLM comparative reasoning and reduces one-token class imbalance.
