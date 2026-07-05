# RLLM alpha failure guardrails (2026-07-05)

This note exists because the long-regime midfrequency branch started repeating prior RLLM failures: oracle ceilings, future-best target construction, cheap classifier overfit, and hand-tuned rule drift.  These rules are mandatory for future alpha claims in this repo.

## Non-negotiable definitions

A result is **deployable evidence** only if all are true:

1. Candidate/action rows exist at signal time without future outcome selection.
2. Prompt/model/selector inputs use only signal-time or prior data.
3. Candidate thresholds are fit before the evaluated split.
4. Selector thresholds/margins are selected before the evaluated split.
5. The evaluated split is not used for target shaping, threshold selection, model selection, or report cherry-picking.
6. Backtest uses strict execution:
   - next/open delayed entry,
   - fees and slippage,
   - non-overlap or explicit position-state handling,
   - in-hold OHLC adverse excursion in strict MDD,
   - period-end forced close if still holding,
   - CAGR annualized over the full declared split including no-trade time.

Anything else is diagnostic only.

## Banned as strategy evidence

Do not present these as strategy performance:

- **Oracle ceiling**: choosing the best candidate using future reward.
- **Future-best binary records**: pre-collapsing each signal to `NO_TRADE` vs the future-best executable trade, then replaying that chosen candidate.
- **Eval-shaped targets**: modifying labels/gaps/features after inspecting 2025 eval behavior.
- **Rule-mining loops** that search quantile gates directly on 2024+2025 and then describe the result as RLLM alpha.
- **Trade-span CAGR** when no-trade time was excluded.
- **Trade-to-trade MDD only** when in-hold excursion is available.

These can be used only as:

- candidate-book expressiveness diagnostics,
- upper-bound sanity checks,
- target-noise audits,
- ablation evidence.

They must be labeled `diagnostic_not_deployable`.

## Minimum evidence tiers

### Tier 0 — diagnostic

Examples: oracle ceiling, future-best target, same-split rule scan.

Allowed conclusion: “the candidate book contains possible profitable actions.”

Forbidden conclusion: “the strategy works.”

### Tier 1 — non-leaky cheap selector

A train-only or rolling prior-only selector scores candidates independently and selects actions without future rewards.

Required report fields:

- train/test/eval/ytd strict stats,
- trade count,
- Sharpe or t-stat proxy,
- selected threshold source,
- forced period-end exits,
- exact candidate stream source.

Allowed conclusion: “selector baseline recovers / fails to recover the ceiling.”

### Tier 2 — RLLM/LLM selector proof

A frozen adapter/checkpoint or deterministic model scores candidates with no evaluated-split label access.

Required before any live-candidate claim:

- threshold selected on train or 2024 validation only,
- final 2025 eval untouched until one report,
- 2026 YTD reported as regime drift diagnostic,
- no target/feature changes after looking at eval unless eval is discarded and a new held-out split is defined.

### Tier 3 — live-candidate dry-run

Only after Tier 2:

- dry-run config only,
- manual regime gate,
- stale-data and open-position guards,
- several days/weeks paper stream compared to offline event definitions.

## Current long-regime midfrequency branch status

Artifacts:

- `data/long_midfreq_action_book_2026-07-05/`
- `results/long_midfreq_action_book_fast_ridge_baseline_2026-07-05.json`
- `results/long_midfreq_binary_selector_2026-07-05.json`
- `results/long_midfreq_candidate_level_selector_2026-07-05.json`

Status:

- Action-book format: keep.
- Oracle ceiling: diagnostic only.
- Future-best binary replay: invalid as deployable proof.
- Non-leaky candidate-level selectors: failed to generalize; best quick baseline had strong 2024 but collapsed in 2025.

Best non-leaky quick baseline:

| selector | split | CAGR | strict MDD | CAGR/MDD | trades |
| --- | --- | ---: | ---: | ---: | ---: |
| candidate logistic gap>=0.2 | 2024 test | 52.81% | 10.64% | 4.96 | 43 |
| candidate logistic gap>=0.2 | 2025 eval | 4.06% | 12.80% | 0.32 | 31 |

Conclusion: do not continue by adding more hand-tuned gates.  The next valid work is selector robustness, not more candidate mining.

## Required next-step shape

The next experiment must be one of:

1. **Rolling prior-only selector audit**: train/fit on `<year`, validate next year, advance; no fixed 2025 shaping.
2. **Time-balanced pairwise/listwise preference**: build records without future-best preselection; score every candidate independently or score a complete same-time candidate set.
3. **Teacher distillation from a non-leaky teacher**: labels come from a prior-only selector/policy, not raw future utility.

Every next result must explicitly state whether it is Tier 0, 1, 2, or 3.

## Rolling prior-only audit result — 2026-07-05

Artifact: `results/long_midfreq_rolling_prior_audit_2026-07-05.json`.

Validation-selected outcomes:

| fold | validation | selected model | validation CAGR/MDD | test | test CAGR/MDD | test CAGR | test MDD | trades |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2024 audit | 2023 | logreg_gap0.2 | 0.36 | 2024 | 1.80 | 49.34% | 27.38% | 87 |
| 2025 audit | 2024 | hgb_gap0.2 | 4.82 | 2025 | 2.19 | 36.40% | 16.64% | 73 |
| 2026 YTD audit | 2025 | logreg_gap1.0 | 4.63 | 2026 YTD | 0.07 | 1.58% | 23.51% | 39 |

Guardrail conclusion: no Tier 1 deployable evidence.  The clean prior-only 2025 fold fails the target, and 2026 YTD drift is severe.  Do not continue by expanding hand-tuned candidate gates inside this family unless the experiment is explicitly labeled diagnostic-only.
