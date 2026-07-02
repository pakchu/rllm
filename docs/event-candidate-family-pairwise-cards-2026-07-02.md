# Pairwise family state-card records (2026-07-02)

## Why

Randomized listwise state cards removed option-position bias, but direct one-token SFT still has two issues:

1. small row count;
2. noisy family targets from the monthly selector.

This pass converts each state card into pairwise chosen/rejected examples so a model can learn comparative family validity from compact evidence and position state.

## Implementation

New script: `training/build_event_candidate_family_pairwise_cards.py`

Behavior:

- reads randomized family state-card JSONL;
- finds the target option;
- pairs it against top rejected options by pre-fold score;
- includes `position_state` in every prompt;
- prompt asks for exactly `A` or `B`;
- chosen option is always option A within the pair, making it suitable for SFT-style or DPO conversion.

Leakage guard:

- source options come from pre-fold scoreboard;
- target fold metrics are not placed in the prompt;
- position state is included explicitly.

## Generated pairwise splits

Inputs are the randomized monthly state cards.

| split | output | rows | target families |
| --- | --- | ---: | --- |
| train 2023-2024 | `data/event_candidate_family_pairwise_cards_rex_core_1m_train_2023_2024_random_2026-07-02.jsonl` | 72 | `location_revert=30`, `ABSTAIN=18`, `context_pullback=9`, `compression_fakeout=9`, `deep_pullback=3`, `compression_breakout=3` |
| test 2025 | `data/event_candidate_family_pairwise_cards_rex_core_1m_test_2025_random_2026-07-02.jsonl` | 36 | `compression_breakout=18`, `ABSTAIN=15`, `location_revert=3` |
| eval 2026H1 | `data/event_candidate_family_pairwise_cards_rex_core_1m_eval_2026h1_random_2026-07-02.jsonl` | 15 | `location_revert=12`, `ABSTAIN=3` |

All generated rows were read back and verified to include `position_state` and `position_state` in the prompt.

## Verification

- `py_compile` passed for exporter and tests.
- Manual tests passed for:
  - chosen/rejected contrast generation;
  - position state in prompt;
  - JSONL writing.

## Limitation

This fixes data shape and pairwise contrast, not target quality.  The monthly selector still over-labels `rex_multiscale_location_revert`, especially in eval.  Before spending GPU time on Gemma, the target should be cleaned by using target-fold diagnostic outcomes under strict train/test/eval separation, or by generating labels from a stability rule that explicitly demotes location-reversion decay.

## Next action

Build a clean-target pairwise exporter:

- choose the best target-fold diagnostic family only for train/test label construction;
- keep eval fully held out for final validation;
- or derive a train-only stability teacher and apply it without peeking at eval.
