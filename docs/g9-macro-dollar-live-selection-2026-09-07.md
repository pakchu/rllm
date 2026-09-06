# Operator-approved live portfolio selection — 2026-09-07

The user explicitly accepted the discussed tradeoffs and approved **G9 unchanged + macro1 + dollar-rally short0.5**. This is not approval of the optimizer rank-one replacement or the failed-rebound short.

The decision is recorded in `configs/approved/g9_macro1_dollar_short05_2026-09-07.json`.

Notional/equity coefficients:
- Fresh Kimchi:1.0
- Rank7:1.5
- REX taker:0.2
- REX veto:0.8
- Markov long:1.0
- Macro-flow/regime switch:1.0
- Dollar-rally short:0.5

Cross-sleeve overlap and long/short netting remain approved, with4.5x post-fee net cap at open/rebalance events. This is not a continuous intrabar cap. Original G9 legacy weights are retained separately to prevent a2x conversion error.

## Selection approval is not deployment

The new sleeve configurations are still research records (`runtime_ready:false`). Native macro-target execution, dollar-short signal/lifecycle integration, and aggregate-net live parity require implementation and verification. The approval artifact is deliberately outside the runtime config directory, disabled, and cannot itself authorize order submission.

No existing live configuration was replaced. No service was restarted and no exchange order was submitted. Historical validation records remain unchanged; user acceptance does not convert exposed backtests into pristine OOS or prove execution readiness.
