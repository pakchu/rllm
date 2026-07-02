# Clean pairwise learnability audit (2026-07-03)

## Context

After fixing pair-local option ids and switching responses to JSON choices, Gemma4 SFT still failed to generalize:

- JSON-choice SFT 32-step generation:
  - train: 49.3% (A=49, B=95)
  - test: 45.8% (A=21, B=51)
  - eval: 20.0% (A=11, B=19)
- Choice-token logprob:
  - train: 50.0%
  - test: 52.8%
  - eval: 50.0%

The model is therefore not extracting a stable ranking function from the visible prompt features.

## Audit added

Added `training/audit_clean_pairwise_learnability.py` to score simple prompt-visible rules against clean pairwise labels:

- higher/lower `pre_fold_score`
- higher `evidence_count`
- prefer/avoid `ABSTAIN`
- a hand-written combo of prompt-visible latest evidence fields

## Results

Reports:

- `results/clean_family_pairwise_train_learnability_audit_2026-07-03.json`
- `results/clean_family_pairwise_test_learnability_audit_2026-07-03.json`
- `results/clean_family_pairwise_eval_learnability_audit_2026-07-03.json`

Key rule accuracies:

| Split | higher pre-fold score | lower pre-fold score | best simple rule |
| --- | ---: | ---: | --- |
| train 2023-2024 | 47.2% | 52.8% | prefer ABSTAIN 56.9% |
| test 2025 | 55.6% | 44.4% | higher score / avoid ABSTAIN / evidence count 55.6% |
| eval 2026H1 | 20.0% | 80.0% | lower score 80.0% |

The direction of `pre_fold_score` flips across splits.  A model trained chronologically cannot safely infer whether high or low score is better without extra regime information or a different label construction.

## Interpretation

Current clean pairwise labels are not a stable supervised target for the prompt features.  The target uses target-fold diagnostics, but the visible state card only contains pre-fold family summary features.  If the diagnostic winner is often a low-scoring family in one regime and a high-scoring family in another, the LLM sees contradictory patterns.

This means the next useful step is not more SFT/DPO steps.  The target/data structure should change:

1. Build labels around **regime-conditioned transformations** rather than raw diagnostic winners.
2. Include explicit regime descriptors that can explain score direction flips.
3. Or train the LLM to output a **hypothesis/rationale/regime class** first, then let a separate causal selector map that to family choice.

## Verification

- `py_compile` for the audit script and test.
- Manual unit test for prompt-visible rule auditing.
- Generated train/test/eval audit reports.

`pytest` is still unavailable in the current venv.
