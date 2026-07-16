# Causal residual expert switcher development (CRES-1)

## Decision

- Development gate: **PASS**.
- This is not OOS evidence and not deployable. Every 2023-2025 outcome was already research-seen.
- 2026 post-entry outcomes remained unopened by this evaluator; one exact policy must be frozen first.
- The strategy is market-neutral across six alt perpetuals and holds no BTC leg.

## Frozen selected development policy

- Online ridge: min history 52, last 104, alpha 300, no target-mean/intercept drift.
- Confidence: trade only above the 0.825 in-sample absolute fitted-edge quantile.
- Outcome publication lag: prior exit + 5 minutes <= signal.
- Risk scale: 3-day completed 5m max-leg log-range RMS versus prior 52-event median, clipped [0.25, 1.00].
- Cost: 6 bp/side base, 10 bp/side stress; funding included.
- Strict MDD: global pre-entry HWM, funding cash, favorable-before-adverse held OHLC, hypothetical liquidation cost.

## Development metrics

| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 warm-up | -1.03% | -1.03% | 2.84% | -0.36 | 2 |
| 2024 | 20.07% | 20.03% | 5.44% | 3.68 | 25 |
| 2025 | 13.79% | 13.80% | 5.30% | 2.60 | 24 |
| 2024-2025 combined | 36.63% | 16.88% | 5.44% | 3.10 | 49 |

## Controls (2024-2025 combined)

| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 10 bp/side | 32.33% | 15.03% | 5.53% | 2.72 | 49 |
| Direction flip | -34.07% | -18.79% | 35.28% | -0.53 | 49 |
| Entry/exit +5m | 34.52% | 15.97% | 5.44% | 2.94 | 49 |

## Multiple-testing and execution warning

The policy is a post-hoc successor to failed LORE/LORC studies. The disclosed development search included rolling/weighted experts, ridge windows and penalties, confidence levels, regime rules, stop diagnostics, and causal risk scalers. Therefore only a preregistered one-shot 2026 replay can confirm it.

The current live executor is BTC single-symbol oriented. CRES-1 remains research/shadow-only until atomic two-leg alt execution, partial-fill neutralization, per-leg min-notional/slippage, and pair-level reservation are implemented and parity-tested.

## Artifacts

- `results/causal_residual_expert_switcher_development_2026-07-17.json`
- `training/develop_causal_residual_expert_switcher_pre2026.py`
- `tests/test_develop_causal_residual_expert_switcher_pre2026.py`
