# Gross9 fixed-candidate and state-substitution battery preregistration

`G9-FCSS-1` is frozen before the new shared-clock marginal scan.

## Candidate cells

- Add one fixed independent sleeve: `nonpb30_taker` or `oi_divergence_highfreq`, weight 0.25–1.00.
- Or replace 0.25–2.00 of the existing 2.00 `markov_transition_long` allocation with exactly one fixed 6-of-10 state ensemble: Kalman, BOCPD, or Semi-Markov.
- Addition cells must beat a pro-rata Gross9 control at the same gross. State substitutions keep both gross 9.0 and funding/premium family gross 2.0 unchanged.

## Selection boundary

- Rank on train through 2023 plus 2024 only.
- Require both selection windows to beat the relevant comparator by at least 0.05 CAGR/strict-MDD, retain at least 97% of Gross9 absolute return, and reduce strict MDD in at least one window.
- 2025, 2026, and the already inspected July replay cannot rank, repair, or choose another cell. They may only veto frozen top 1.
- All candidates and later windows are research-exposed; any survivor is forward-shadow only.

The complete machine-readable contract, exact source hashes, weight grids, exclusions, diagnostics, and veto rules are in `results/gross9_fixed_candidate_state_substitution_preregistration_2026-07-28.json`.

## Clarifications

- The same-gross pro-rata addition control is a deliberately non-deployable leverage counterfactual and is exempt from family caps; every candidate portfolio itself must respect the caps.
- State substitutions always exclude the residual Markov sleeve from acceptance Jaccard because both gate the same base setup, but exact Markov overlap remains a mandatory diagnostic.
- The frozen July artifact does not contain the 6-of-10 state ensembles. It is used only for the two addition candidates; a state survivor remains forward-shadow even if the 2025/2026 veto passes.
- The evaluator must hard-code this committed preregistration SHA-256 and fail closed on any contract or input drift.
