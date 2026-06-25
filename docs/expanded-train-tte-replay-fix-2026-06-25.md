# Expanded train TTE and sparse replay fix (2026-06-25)

## What changed

The 2020-2024 train / 2025 test / 2026H1 eval run exposed a replay bug in sparse setup validation.

`_candidate_events` recomputed thresholds for every replay fold instead of respecting fold-level thresholds and zero-sample skips already stored in the sparse discovery report. This made `train_2020h1` produce live replay events even though discovery marked it as `not_enough_train` / zero samples. In one representative config this created 241 artificial trades in `2020H1`.

Fix:

- For folds present in `candidate.strict_folds`, replay stored thresholds and stored side.
- If the stored fold result has zero samples/trades, emit no events for that fold.
- For new folds not present in the candidate report, continue fitting thresholds from past-only data.

## Expanded split

- Train discovery: `2020H1` through `2024H2`
- Test selection: `2025H1`, `2025H2`
- Eval holdout: `2026H1` through `2026-06-01`

Artifacts are under:

- `results/sparse_setup_tte_2020train_combined_pa_macro_2026-06-25/`

## Strict TTE result after replay fix

Small config sweep:

- `candidate_limits`: `1,2,4,8`
- `ensemble_sizes`: `1,2`
- selected by test score only
- selected config: `candidate_limit=4`, `max_ensemble_size=2`

| Period | CAGR | Strict MDD | CAGR/MDD | Trades | p-value approx | Power gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 2020-2024 | 30.39% | 18.88% | 1.61 | 315 | 0.0009 | 0 |
| Test 2025 | -12.17% | 15.05% | -0.81 | 89 | 0.3192 | 616 |
| Eval 2026H1 | 18.12% | 7.37% | 2.46 | 56 | 0.5055 | 936 |
| All | 21.01% | 22.22% | 0.95 | 460 | 0.0035 | 0 |

This fails the objective. Test 2025 is negative and eval is not statistically supported.

## Candidate-level diagnosis

Top individual test-ranked candidates from the 80 train-discovered pool are weak:

- Candidate 34: train `6.03/13.25 ratio 0.46`, test `7.81/3.15 ratio 2.48`, eval `7.80/7.36 ratio 1.06`.
- Candidate 74: train `8.13/21.31 ratio 0.38`, test `9.17/6.38 ratio 1.44`, eval `26.13/8.21 ratio 3.18`.
- Combo `[34,35,74]`: train `14.98/19.93 ratio 0.75`, test `17.61/6.38 ratio 2.76`, eval `15.55/10.36 ratio 1.50`.

The pool contains weak alphas but no robust train/test/eval-valid strategy.

## Interpretation

The expanded train set did not solve generalization. It mostly made the weakness clearer:

1. Current discovery ranking can select candidates with attractive fold medians but poor cross-period deployment behavior.
2. 2025 remains the decisive failure period.
3. Eval-only strength is not sufficient; any live candidate must pass test first without using eval.
4. The sparse setup feature family has weak signal, but not enough standalone edge under strict MDD constraints.

## Next direction

Do not keep tuning selector/gate parameters on this same pool. The next useful step is feature/label redesign:

- Add regime-conditioned labels explicitly, especially for 2025-like chop/down regimes.
- Penalize train candidates by deployment-style continuous MDD, not only fold median CAGR.
- Require candidate train ratio > 1 and train strict MDD < 15 before test selection.
- Treat sparse setups as weak features for a higher-level learner rather than direct strategy triggers.
