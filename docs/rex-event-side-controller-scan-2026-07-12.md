# REX event side-controller scan (2026-07-12)

## Goal
The event-first REX candidate is side/regime fragile. This scan keeps the base REX event fixed, then learns simple causal side allow-filters from train only.

Base event:
- family: `rex_htf_pullback_reclaim`
- hold: 144 bars
- stride: 24 bars
- entry delay: 1 bar
- thresholds: fit on train only

Side filter structure:
- LONG event allowed only if one train-fit feature condition is true.
- SHORT event allowed only if one train-fit feature condition is true.
- Selection uses train+test only; eval is final holdout.

## Recent split: train 2020-2024 / test 2025 / eval 2026H1
Artifact: `results/rex_event_side_controller_scan_2026-07-12.json`

Best selected rule:
- LONG allow: `rsi_norm <= -0.3052479583`
- SHORT allow: `range_vol >= 0.0215245647`

| split | abs return | full-window CAGR | strict MDD | CAGR/MDD | trades | p approx | sides |
|---|---:|---:|---:|---:|---:|---:|---|
| train 2020-2024 | 89.23% | 13.60% | 15.04% | 0.90 | 432 | 0.019 | L184/S248 |
| test 2025 | 17.99% | 18.01% | 2.54% | 7.09 | 35 | 0.0003 | L10/S25 |
| eval 2026H1 | 5.65% | 14.22% | 4.11% | 3.46 | 28 | 0.233 | L1/S27 |

Interpretation:
- This is materially better than the ungated recent event on train risk and recent OOS ratio.
- Still not standalone live-grade because train 2020-2024 CAGR/MDD is only 0.90 and 2026H1 has only 28 trades.
- Useful as an RLLM context/action prior: “deep RSI pullback for longs; high realized range for shorts.”

## Broad split: train 2020-2023 / test 2024 / eval 2025-2026H1
Artifact: `results/rex_event_side_controller_broad_2026-07-12.json`

Best selected rule:
- LONG allow: `return_zscore_48 <= -1.0045715298`
- SHORT allow: `return_zscore_48 <= -0.5528652569`

| split | abs return | full-window CAGR | strict MDD | CAGR/MDD | trades | p approx | sides |
|---|---:|---:|---:|---:|---:|---:|---|
| train 2020-2023 | 78.22% | 15.54% | 11.63% | 1.34 | 241 | 0.0065 | L136/S105 |
| test 2024 | 17.74% | 17.70% | 2.84% | 6.24 | 36 | 0.0038 | L27/S9 |
| eval 2025-2026H1 | 4.70% | 3.31% | 3.51% | 0.94 | 32 | 0.268 | L8/S24 |

Interpretation:
- Broad split fails final eval ratio; it overfits 2024-style pullback behavior.
- This confirms the side-controller is regime-sensitive and cannot be promoted as a universal 3-year alpha.

## Verdict
- Side-controller improves the recent REX candidate substantially.
- It is not the final target alpha: it fails the 3-year+ robust CAGR/MDD>=3 requirement.
- The useful discovery is structural: RLLM should not output trade/no-trade from raw numbers only; it should reason over event + side-context rules:
  - Is this a deep pullback/reclaim event?
  - Is long side in local oversold/reclaim context?
  - Is short side in high realized range / stress continuation context?
  - If neither side context is clean, skip.
