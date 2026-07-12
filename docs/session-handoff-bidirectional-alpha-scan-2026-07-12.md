# Completed-session handoff alpha scan

Date: 2026-07-12

## Protocol

- Entries are considered only at completed 8-hour UTC session boundaries (`00:00`, `08:00`, `16:00`).
- Inputs summarize only the preceding completed session: return, path efficiency, high-low range, taker flow, flow shift and volume ratio.
- All thresholds are frozen on Train (`2020-2023`).
- Test 2024 is the only ranking split; Eval 2025 and 2026 YTD are report-only.
- Entry delay is one 5-minute bar; cost is 6 bp/side; strict MDD includes intraposition adverse excursion.
- Tested combinations: 2,118.

## Result

No candidate passed the alpha-pool or live-grade gate.

Best Test-2024 candidate: `handoff_efficient_sh_boundary_16_0.9_0.6`, entering after the `16:00 UTC` handoff with hold 72 bars, TP 2.5%, SL 1.5%.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long/short |
|---|---:|---:|---:|---:|---:|---:|
| Train | +4.28% | 1.05% | 8.01% | 0.13 | 204 | 110 / 94 |
| Test 2024 | +13.55% | 13.52% | 1.89% | 7.15 | 54 | 25 / 29 |
| Eval 2025 | -1.95% | -1.95% | 5.68% | -0.34 | 54 | 19 / 35 |
| 2026 YTD | +2.63% | 6.44% | 1.59% | 4.06 | 21 | 13 / 8 |

## Interpretation

The 16:00 UTC handoff effect was unusually strong in 2024 but failed in 2025 and missed the 2026 target. The complete-boundary variant also lost in Train, 2025 and 2026. This is a calendar-regime effect rather than a stable standalone alpha.

The family is rejected for live promotion. Session identifiers may remain useful as context for an LLM/RLLM regime explanation, but they must not act as a hard trading rule based on this scan.

## Artifacts

- Script: `training/search_session_handoff_bidirectional_alpha.py`
- Result: `results/session_handoff_bidirectional_alpha_scan_2026-07-12.json`
