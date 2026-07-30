# Gross9 structural clock bundle Rank7 authority amendment — 2026-07-31

## Status

This candidate-independent amendment is `G9CB-1A`. It supersedes only the
Rank7 reconstruction and internal-computation clauses of `G9CB-1` in:

`docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md`

Every other `G9CB-1` requirement remains in force, including the domain,
five-sleeve contract, source and environment seals, pre-access claim,
attempt-consumed sentinel, exactly two fresh rebuilds, byte identity,
manifest-last publication, no retry, and the prohibition on candidate or
overlap access.

No source row, model array, history row, comparator clock row, candidate row,
return, PnL, or metric was opened to create this amendment. The amendment is
based only on already authenticated source code, configuration metadata, and
the committed Rank7 bundle manifest metadata.

`G9CB-1A` is not operative merely because this document exists or is
committed. Before the protocol commit `P` may be sealed, the preregistration
producer, production builder, and their tests must implement every requirement
in this amendment. `P` and the preregistration must bind this exact amendment
by repository-relative path, its standalone authority commit, SHA-256, and Git
blob. The claim, sentinel preflight, each fresh worker, each per-pass core, and
the final manifest must authenticate that binding before any generic import or
value-row access. A standalone amendment commit before `P` is metadata-only
authority work and does not permit a claim, sentinel, or production rebuild.

Any protocol commit that omits the bundle load, hourly-history use,
annual-refit/bundle parity, extended counters, or their regression tests is not
an implementation of `G9CB-1A` and must not be used as `P`.

## Contradiction repaired

`G9CB-1` requires an exact historical `frozen_annual_rank7` clock over:

```text
[2023-06-01T00:00:00Z, 2026-06-01T00:00:00Z)
```

The exact historical policy uses annual refit folds. Reproducing those folds
requires the source-owned two-output Rank7 training labels:

```text
net
adverse
```

Those labels are derived from exact trade price, funding, and adverse-path
factors. The original decision simultaneously stated that funding rows were
causal features only and prohibited applying a funding cash flow. Read
literally, that prohibition makes exact historical Rank7 reconstruction
impossible.

The frozen runtime bundle does not resolve the contradiction by itself. Its
models and thresholds are valid from the 2026 annual cutoff; they cannot be
applied retrospectively to 2023–2025 and still be called the exact historical
annual-refit clock.

`G9CB-1A` therefore distinguishes:

1. a narrowly authorized, source-owned Rank7 training-label replay needed to
   reproduce historical activation; from
2. forbidden portfolio economics, candidate economics, and published
   economic values.

## Exact Rank7 reconstruction

The production adapter must use the authenticated generic Rank7 feature,
annual-refit, activation, and source-routed scheduling contracts.

The frozen learner is:

```json
{"max_depth":2,"max_features":0.8,"min_samples_leaf":32}
```

The frozen selection policy is:

```json
{
  "funding_quantile":0.4,
  "premium_quantile":0.55,
  "risk_lambda":0.25,
  "risk_quantile":0.75
}
```

These values are protocol constants sealed by the builder Git blob. Production
must not parse the pre-2025 anchor or a historical economic result to obtain
them.

The exact execution constants used only for the authorized Rank7 labels are:

```json
{"fee_rate":0.0005,"leverage":0.5,"slippage_rate":0.0001}
```

For one successfully replayed training trade, the only authorized label
calculation is:

```text
one_side_cost = fee_rate + slippage_rate
fee_factor = 1 - leverage * one_side_cost
net = fee_factor * price_factor * funding_factor * fee_factor - 1
adverse = max(
    0,
    1 - fee_factor * funding_debit_factor * adverse_price_factor,
)
```

The constants and formula are sealed protocol values, not tunable inputs.
`fee_factor` is constructed once per successfully replayed training trade and
its use is counted by `rank7_fee_factor_values_used`. No other cost,
leverage, stress-cost, return, or label formula is authorized.

For each immutable Rank7 anchor needed by an annual fit, the adapter may:

- replay the exact source-routed long trade;
- examine the exact OHLC path;
- derive the source-owned price, funding, funding-debit, and adverse factors;
- compute exactly one `net` and one `adverse` training label; and
- use those labels only inside the authenticated annual ExtraTrees fit and
  activation contract.

This authorization is limited to reconstructing the already frozen Rank7
decision schedule. It does not authorize selection of a new learner, policy,
cadence, threshold, feature, source, side, hold, barrier, or portfolio weight.

The builder must not call an equity curve, portfolio return, portfolio PnL,
funding-cash amount, CAGR, MDD, rank, correlation, Jaccard, containment, or
overlap helper. It must not aggregate the Rank7 labels into an economic result.

## Frozen 2026 bundle requirement

The adapter must load the authenticated Rank7 bundle through the generic
runtime contract after the sentinel:

```text
artifacts/rank7/frozen_annual_rank7_2026
```

It must open and count:

- all five sealed model files; and
- the sealed completed-hourly-history file.

The bundle is authoritative for its valid 2026 annual window. The adapter must
rebuild the Rank7 feature context using the bundle medians, clipping contract,
delay, and hourly history, score the bundle models with `n_jobs=1` semantics,
and reproduce the bundle source/risk/interaction thresholds.

The historical annual-refit activation and frozen-bundle activation must be
identical throughout the bundle-valid portion of the `G9CB-1` domain. Any
activation, source identity, or scheduled-entry mismatch is terminal after the
sentinel. The canonical 2026 intervals must use that common activation.

The bundle must not be applied before its `valid_from` timestamp.

## Structural barrier replay

Fresh Kimchi and the final Rank7 schedule require only structural barrier
geometry. Their ordinary schedule replay must use OHLC-only stop-before-take
logic and must not compute funding factors, fees, returns, or PnL.

The economically derived execution factors are authorized only for the
Rank7 annual-fit label replay described above.

## Required counters

The permanent access-counter schema is extended with:

```text
rank7_training_trades_replayed
rank7_net_labels_computed
rank7_adverse_labels_computed
rank7_price_factor_values_used
rank7_funding_factor_values_used
rank7_funding_debit_factor_values_used
rank7_adverse_price_factor_values_used
rank7_fee_factor_values_used
rank7_bundle_activation_rows_scored
rank7_bundle_parity_rows_compared
```

The existing counters remain required:

```text
model_files_opened
rank7_hourly_history
prediction_rows_scored
outcome_dependent_ohlc_rows_examined
```

All counters increment when the value is actually opened, computed, scored, or
compared. They must not be inferred from output interval count.

The following assertions remain zero:

```text
portfolio_return_values_computed
portfolio_pnl_values_computed
funding_cash_values_computed
cagr_values_computed
mdd_values_computed
economic_rank_values_computed
candidate_metric_values_computed
overlap_metric_values_computed
```

`rank7_funding_factor_values_used` is a dimensionless source-owned model-label
input and is disclosed separately. `rank7_fee_factor_values_used` counts the
deterministic scalar constructed from the three sealed constants above; it
does not authorize any additional cost input. `funding_cash_values_computed=0`
means the builder never converts a funding factor into a cash amount,
portfolio path, sleeve return, or allocation result.

## Output boundary

No Rank7 feature value, model score, threshold, label, price factor, funding
factor, return, PnL, or cash amount may appear in the canonical CSV.

The Rank7-specific additions to the per-pass core and final manifest may
contain only:

- the added integer counters;
- authenticated model/history provenance;
- activation-parity status and compared-row count; and
- the already authorized structural interval and publication metadata.

They must not contain the underlying values. The complete source,
environment, claim, sentinel, two-pass receipt, and publication schema already
required by `G9CB-1` remains required and is not excluded by this Rank7-specific
list.

## Failure rule

Before the sentinel, any inability to authenticate the amendment, builder,
generic closure, bundle, source, or environment stops without value access.

After the sentinel, any Rank7 label, model, history, prediction, parity,
schedule, source, or barrier failure consumes the one-shot attempt and is
terminal with no retry or repair.

## Decision

With this narrow amendment, exact historical Rank7 reconstruction is
authorized without converting `G9CB-1` into an economic result. The amendment
does not authorize any new alpha identity or any candidate-dependent choice.
