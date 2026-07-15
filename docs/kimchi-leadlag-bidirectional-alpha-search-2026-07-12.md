# Kimchi lead/lag bidirectional alpha search (2026-07-12)

> **Superseded for live-parity statistics (2026-07-16):** the original search
> allowed forward-filled unavailable FX/Kimchi rows to authorize entries and
> did not include realized funding.  The frozen-threshold repair is audited in
> `docs/fresh-kimchi-orthogonal-alpha-audit-2026-07-16.md`.  Its corrected
> 2024/2025/2026H1 ratios are 2.85/2.38/4.40; this remains a forward-shadow
> low-correlation candidate, not a live-grade standalone alpha.

## Standalone result
- 5,040 symmetric Kimchi lead/lag variants tested.
- No standalone candidate cleared test/eval ratio>=2.5.
- Best balanced standalone `kimchi_lead_continuation_144`: test/eval ratios 1.60/1.44.

## Successful alphaization gate
The fixed `funding_relief_vs_fx_stress` base policy was then gated without changing its entries/exits.

Selected test-only gate:
- feature: `kl_local_impulse_144`
- definition: zscore(144-bar Kimchi premium change) minus zscore(144-bar USDKRW change), both trailing 576 bars
- condition: `<= -2.2728671108` (train q0.10)
- applies only to the long leg; short leg remains unchanged

| split | return | CAGR | strict MDD | ratio | L/S | win L/S | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 16.30% | 3.85% | 11.37% | 0.34 | 63/105 | 50.8/44.8% | 0.94 |
| test2024 | 11.98% | 11.95% | 3.29% | **3.64** | 9/23 | 55.6/60.9% | 1.63 |
| eval2025 | 10.44% | 10.44% | 3.41% | **3.06** | 15/8 | 73.3/50.0% | 1.70 |
| ytd2026 | 7.86% | 19.95% | 5.58% | **3.58** | 25/8 | 60.0/50.0% | 1.03 |

## Verdict
- Clears test/eval live-grade ratio>=3 with both directions.
- 2026 is positive and materially better than previous bidirectional gates, but ratio 3.58 remains below the requested 5.
- Sample remains modest, especially test long=9 and eval short=8. Candidate-grade, not automatic live promotion.

## Artifacts
- `training/search_kimchi_leadlag_bidirectional_alpha.py`
- `results/kimchi_leadlag_bidirectional_alpha_scan_2026-07-12.json`
- `training/search_bidirectional_kimchi_gate_alpha.py`
- `results/bidirectional_kimchi_gate_alpha_scan_2026-07-12.json`
