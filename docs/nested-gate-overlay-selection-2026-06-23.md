# Nested gate/overlay selection (2026-06-23)

## Purpose

The month-validation gate plus take-profit diagnostic looked strong on the full rolling window, but gate threshold and overlay were selected on the same result. This test separates selection and eval:

- selection period: `2024-01` through `2025-12`;
- untouched eval period: `2026-01` through `2026-05`;
- source predictions: pairwise event-context preference ranker;
- configs: month-validation thresholds, stop-loss, take-profit, rolling loss stops.

## Implementation

Added:

- `training/nested_gate_overlay_selection.py`
- `tests/test_nested_gate_overlay_selection.py`

The selector ranks configs using only the selection period, then applies the top configs to eval.

## Selection result

Best selected config on 2024-2025:

| Parameter | Value |
| --- | ---: |
| month validation threshold | 0.5 |
| stop loss | 0.0% |
| take profit | 3.0% |
| rolling loss stop | off |

Selection-period performance:

| Metric | Value |
| --- | ---: |
| Trades | 316 |
| CAGR | 28.70% |
| Strict MDD | 12.81% |
| CAGR/MDD | 2.24 |
| p-value approx | 0.014 |

## Untouched 2026 eval result

The selected config blocks all 2026 months because every 2026 month has poor prior-validation health.

| Metric | Value |
| --- | ---: |
| Trades | 0 |
| CAGR | 0.0% |
| Strict MDD | 0.0% |
| CAGR/MDD | 0.0 |

This is not a profitable eval; it is abstention.

## Diagnostic: if eval configs are ranked using eval itself

For diagnosis only, all configs were evaluated on 2026. Among configs that actually traded, the best was still negative:

| Config | Trades | CAGR | Strict MDD | CAGR/MDD | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| threshold=-1000, stop=4%, no TP | 78 | -10.61% | 12.60% | -0.84 | 0.730 |

So 2026 failure is not primarily a gate/overlay-selection issue. The underlying side/ranker edge is absent or inverted in 2026.

## Interpretation

Clean validation invalidates the prior optimistic diagnostic claim.

What remains true:

- 2024-2025 selection period has a real-looking edge.
- Prior-validation health is a useful safety gate.
- The gate correctly refuses to trade 2026 rather than forcing bad trades.

What fails:

- The system does not earn money in untouched 2026.
- No evaluated overlay rescues 2026 traded performance.
- The current event-context ranker lacks regime adaptation for 2026.

## Decision

Do not promote the current pairwise ranker/gate/overlay stack to Gemma SFT as a profitable policy.

Next work must address 2026 side-edge collapse directly:

1. Learn separate regime-specific policies, especially for 2026-like conditions.
2. Add explicit recent-performance memory as row-level features, not only month-level gating.
3. Investigate whether 2026 requires different action family/horizon, not just a gate.
4. Treat abstention as acceptable safety, but not as target achievement.
