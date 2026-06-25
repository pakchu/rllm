# Sparse setup action-policy bridge — 2026-06-25

## Purpose

Convert the best sparse setup families into LLM/RL-ready event-action records and test whether a past-only action selector can improve execution without using current-fold rewards.

## Implementation

- Added `training/sparse_setup_action_policy.py`.
- It reconstructs sparse setup events with the same fold-local threshold/side fitting used by `sparse_setup_ensemble_audit.py`.
- Each event becomes an action book over side/hold choices:
  - holds: `24,36,72,144`
  - sides: setup source side plus opposite side
  - outcome labels: executable delayed-entry path utility from `compute_trade_path_outcome`
- Walk-forward action rules are fit only on previous folds.
- Fallback mode `always_when_no_rule` can use the setup source side/horizon when no action rule qualifies. This is still fold-safe because the setup source side is fit before each fold start.

## Validation

```bash
.venv/bin/python -m unittest tests/test_sparse_setup_action_policy.py
```

Result: `Ran 6 tests ... OK`.

## Main run

Input sparse report:

`results/sparse_setup_cached_macro_optimized_2026-06-25/report.json`

Main output files:

- `data/sparse_setup_action_policy_2026-06-25.jsonl`
- `results/sparse_setup_action_policy_2026-06-25/report_fallback_source.json`
- `results/sparse_setup_action_policy_2026-06-25/top_limit_sweep.json`

## Findings

Full top-16 action-rule-only selector was too fragmented and stopped trading from 2025 onward:

- CAGR: `~7.0%`
- strict MDD: `~19.3%`
- CAGR/MDD: `~0.36`

Using source-action fallback and sweeping candidate pool size showed that more candidates were not better. Top-4 was best:

| candidate_limit | trades | CAGR | strict MDD | CAGR/MDD | note |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 187 | 20.14% | 14.67% | 1.37 | clean but low trade count |
| 2 | 203 | 19.61% | 15.32% | 1.28 | MDD above target |
| 4 | 218 | 21.55% | 14.51% | 1.49 | best overall in this run |
| 8 | 401 | 19.54% | 17.78% | 1.10 | noisy candidates add drawdown |
| 16 | 715 | 13.56% | 24.09% | 0.56 | over-expanded pool |

Top-4 fold detail:

| fold | trades | CAGR | strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| 2023H1 | 11 | 33.60% | 3.39% | 9.90 |
| 2023H2 | 11 | 32.80% | 3.13% | 10.48 |
| 2024H1 | 53 | 40.15% | 11.40% | 3.52 |
| 2024H2 | 43 | 32.69% | 14.51% | 2.25 |
| 2025H1 | 36 | 15.71% | 8.59% | 1.83 |
| 2025H2 | 30 | 0.63% | 12.83% | 0.05 |
| 2026H1 | 34 | 34.36% | 14.37% | 2.39 |

## Interpretation

- The sparse setup family has a real positive edge under a strict no-leak replay, but it is not strong enough for the original target.
- Expanding the pool blindly hurts MDD. The useful signal is concentrated in the first few sparse candidates.
- The action-policy bridge is structurally useful for LLM/RL training, but current action labels only vary side/hold. They do not yet include stop/take-profit/trailing-stop actions, which are likely needed to push CAGR/MDD toward 3.

## Next step

Move from a fixed global stop to a regime-conditioned risk-profile selector. The stop/take-profit follow-up below shows that a 3% stop helps 2026H1 but hurts the full period, so risk controls need to be selected from past regime evidence rather than globally fixed.


## Stop/take-profit follow-up

Added fixed `trade_stop_loss_pct` / `trade_take_profit_pct` execution labels to the action book. A top-4 sweep over stop/tp settings is stored at:

- `results/sparse_setup_action_policy_2026-06-25/top4_stop_tp_sweep.json`

Best full-period result remained no fixed stop/take-profit:

- CAGR: `21.55%`
- strict MDD: `14.51%`
- CAGR/MDD: `1.49`
- trades: `218`

The 3% stop-only setting improved the recent 2026H1 fold sharply but hurt the full period:

- Full period: CAGR `18.75%`, strict MDD `14.10%`, CAGR/MDD `1.33`
- 2026H1: CAGR `53.42%`, strict MDD `10.93%`, CAGR/MDD `4.89`

Interpretation: stop behavior is regime-dependent. A fixed global stop is not enough; the next useful action space is a regime-conditioned risk-profile selector, not a single global SL/TP constant.

## Regime-conditioned risk-profile selector

Added simultaneous risk-profile action keys, for example:

- `LONG_36_base`
- `LONG_36_sl3`
- `LONG_36_sl3tp6`
- `LONG_36_sl5`
- `LONG_36_sl5tp8`

The evaluator now uses `action_key` directly, so a past-only rule can select both direction/hold and risk profile. Source-action fallback can also choose a profile from previous evidence via:

- `fallback_risk_profile_mode=base`
- `fallback_risk_profile_mode=recent_fold_best`
- `fallback_risk_profile_mode=history_best`

Validation files:

- `results/sparse_setup_action_policy_2026-06-25/top4_risk_profiles.json`
- `results/sparse_setup_action_policy_2026-06-25/top4_risk_recent_fold_best.json`
- `results/sparse_setup_action_policy_2026-06-25/top4_risk_history_best.json`

Results:

| mode | CAGR | strict MDD | CAGR/MDD | selected profiles |
| --- | ---: | ---: | ---: | --- |
| base selector with risk book | 19.79% | 14.51% | 1.36 | base 210, sl5tp8 8 |
| recent_fold_best | 18.27% | 14.51% | 1.26 | base 95, sl5tp8 53, sl3 70 |
| history_best | 18.34% | 14.51% | 1.26 | base 165, sl5tp8 53 |

Important detail: `recent_fold_best` correctly chose `sl3` for 2026H1 and produced:

- 2026H1 CAGR `53.42%`
- 2026H1 strict MDD `10.93%`
- 2026H1 CAGR/MDD `4.89`

But it also chose weaker stop profiles in earlier folds, so full-period ratio dropped. This rejects naive recent-fold profile chasing. The useful next direction is a richer regime/context model that learns when stop-based risk control is appropriate, not a one-fold winner-take-all selector.
