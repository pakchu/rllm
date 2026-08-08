# RV20 volatility-alpha search — 2026-08-08

## Outcome

No candidate was promoted. Every official candidate preserved the frozen protocol, and no 2025–2026H1 future window was opened. The strongest pre-2024 mechanism was a two-sided high-RV liquidation-transfer reversal, but it did not satisfy the independent 2024 portfolio-weight gate.

## Official evidence

| Candidate | Official verdict | Key evidence | Result SHA-256 |
|---|---|---|---|
| `rv20_oi_flush_absorption_reversal` | `REJECT_NO_STRUCTURAL_TOP1` | At most one train trade; signal too sparse. | `283093da9e85dd03c60919a477e21a0bbfaae5409aa701f9b1cc8926a4527e61` |
| `rv20_participation_breakout_continuation` | `REJECT_NO_STRUCTURAL_TOP1` | Best train return +6.30%; best CAGR/MDD 0.239; insufficient persistence. | `bc2d25ed66552e3fc79436bb82d48fd53be05802abb26b24546334108bd1a750` |
| `rv20_low_participation_shock_reversal` | `REJECT_NO_STRUCTURAL_TOP1` | Best train +92.04%, but CAGR/MDD 1.256 below the frozen 1.5 gate. | `8cb0cdd20232885575c70de1e150a706254489289d310052f3372c70ce61d934` |
| `rv20_asymmetric_liquidation_reversal` | `REJECT_NO_LEVERAGE` | Structural top1: +87.76%, CAGR 20.81%, MDD 10.59%, CAGR/MDD 1.966, stress +76.11%; frozen leverage gate not met. | `d54ef5eb6a895c16ebea337294bb9b5b2ff0f496e0713086e72e5129a56f4c2f` |
| `rv20_asymmetric_liquidation_reversal_fixed1x_successor2` | `REJECT_NO_2024_WEIGHT` | Exact train replay; 2024 standalone +9.82%, CAGR/MDD 0.740; no weight passed all confirmation gates. | `f14661f6873e642716b9ca84c75052c6d4d655b8ff20f85e6c9e89d5e88b8724` |
| `rv20_low_participation_shock_reversal_breakeven1_fixed1x` | `REJECT_NO_2024_WEIGHT` | Train +89.65%, CAGR/MDD 1.763 after next-bar-only breakeven; 2024 standalone -2.64%, stress -10.06%. | `75375f0090907b849c09a17312abbf6db0d4af8edca35f3b6fc44b79bbc35e53` |

Two fixed-1x attempts failed before economic output and were terminally sealed rather than retried:

- `rv20_asymmetric_liquidation_reversal_fixed1x`: `7ca8cf391f7c8a083dc7211ac7d31074d2a4656ab1cb6fb9efea43506670882d`
- `rv20_asymmetric_liquidation_reversal_fixed1x_successor`: `9d239d786abd58775497830734428fd5f4bd8f0e8c17bcb4b64514e3c2529ec4`

The successor2 repair was implementation-only: exact pre-2025 context cropping plus finite OHLC comparison with `rtol=0`, `atol=1e-10` for deterministic decimal round-trip drift.

## Integrity controls

- Inputs and Gross9 context dependencies were SHA-256 authenticated before economic decoding.
- Signals used completed-hour data, shifted finite-observation thresholds, and prior-calendar RV20 only.
- Entries occurred at the next 5-minute open; gap exits preceded intrabar checks; ambiguous bars resolved adverse-first.
- Evaluation windows scheduled independently from flat state.
- Costs, exact realized funding, full-calendar CAGR, strict global/intratrade MDD, same-gross controls, entry Jaccard, and persistent long-vol residuals were enforced.
- One-shot markers were durably created before official economic evaluation; terminal failures were never retried.
- Future data could only act as a veto after a 2024 weight passed. This condition was never reached.

## Verification

The eight evaluator test modules pass together: **231 passed**. Python compilation and `git diff --check` also pass.

## Decision

Do not deploy or advertise a new alpha from this search. Preserve the asymmetric liquidation-reversal cell as research evidence only. Its pre-2024 structural strength did not generalize strongly enough through the frozen 2024 confirmation gate, and post-confirmation parameter repair would be contaminated.
