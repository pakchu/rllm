# RLLM reset after analyzer/trader failure — 2026-06-23

## Decision

The two-stage `analyzer -> trader` architecture is deprecated as the active RLLM path.
Keep old modules only for reproducing historical experiments. New experiments must use a
single compact policy interface unless a later no-leak holdout result justifies a wider
interface.

## Why the prior path failed

### 1. No stable alpha was proven

The strongest validation candidates repeatedly collapsed on untouched later periods:

- `results/multiasset_cross_sectional_um6_summary.json`: validation selection reached very
  high CAGR, but 2025-2026 eval fell to negative CAGR with very large drawdown.
- `results/multiasset_feature_model_um6_summary.json`: the learned feature model selected on
  2024 validation failed on 2025-2026 eval.
- `results/multiasset_feature_model_rolling_summary.json` and
  `results/multiasset_feature_model_rolling_threshold_summary.json`: rolling retrain and
  abstention reduced some collapse but did not produce meaningful positive CAGR.
- `results/multiasset_candidate_segment_audit_summary.json`: candidate behavior was
  regime-local and often sign-flipping across half-year segments.

Conclusion: the bottleneck was not the LLM interface. The current features/labels did not
contain a stable, deployable action edge under strict no-leak evaluation.

### 2. The two-stage interface amplified noise

The old path decomposed one trading decision into multiple learned artifacts:

1. analyzer summary generation,
2. trade gate,
3. side selection,
4. optional best-side/offline scoring variants,
5. later regime/policy selection.

This expanded the failure surface without solving alpha stability. Gate/side splits also
made it easy for one weak stage to dominate the entire pipeline, including degenerate
all-`NO_TRADE` or biased side outputs.

### 3. LLM fine-tuning mostly learned unstable labels

Historical Gemma/DPO artifacts show low action accuracy, weak side accuracy, and weak
statistical backing. Preference and SFT rows derived from weak bandit/selector outputs are
not an independent edge; they are only distillation of an already weak source policy.

### 4. Earlier high CAGR results were selection-sensitive

Strict MDD, untouched eval periods, and segment audits exposed that several attractive
numbers were validation artifacts, harness optimism, or regime-local spikes. Future reports
must separate train, selection/test, and final eval claims, and must not promote diagnostic
holdout wins that could not have been selected without looking at that holdout.

## Replacement contract

New RLLM work should use one policy surface:

```text
causal market state -> single policy JSON -> simulator/backtest/RL reward
```

Allowed output shape should stay compact and executable by a deterministic adapter, e.g.:

```json
{
  "regime": "TREND_UP|TREND_DOWN|RANGE|CHOP|REVERSAL_RISK",
  "edge_quality": "NONE|WEAK|MODERATE|STRONG",
  "risk": "LOW|MID|HIGH",
  "action": "NO_TRADE|LONG|SHORT",
  "exit_profile": "AVOID|FAST|NORMAL|TRAIL",
  "confidence": "LOW|MID|HIGH"
}
```

Hard rules:

- no live `analyzer -> trader` cascade,
- no separate gate/side LLMs,
- no final exchange order directly from the LLM,
- no future path, reward audit, or eval-selected statistic in prompts,
- no promotion of a candidate unless it survives no-leak train/test/eval or rolling eval
  with enough trades and strict MDD accounting.

## Next experiment shape

1. Build or reuse a causal state dataset that strips legacy analyzer/trader wording from
   prompts.
2. Train/evaluate a single policy model only as a compact action prior.
3. Use the simulator/RL layer to optimize action selection and sizing against strict MDD,
   not to imitate unstable future-derived labels directly.
4. Treat feature/alpha discovery as a prerequisite. If the candidate alpha pool remains weak,
   do not spend more GPU cycles on LLM fine-tuning.
