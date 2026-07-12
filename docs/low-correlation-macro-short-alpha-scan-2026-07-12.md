# Low-correlation macro short alpha scan (2026-07-12)

## Protocol
- BTCUSDT target; DXY/USDKRW/Kimchi are causal predictor features only.
- Train `<2024` quantile thresholds; rank on test2024; attach eval2025/YTD2026 afterward.
- 6bp/side, 0.5x, next-open entry, strict short intrabar MDD, conservative stop-before-take ordering.
- Candidate activation maximum absolute phi <=0.20 against existing long/short alpha component masks.
- 1,850 eligible entry/exit variants.

## New qualifier: USDKRW risk-off weakness short

Entry:
- `usdkrw_momentum >= 0.0041314411` (train q0.95)
- `htf_1d_return_1 <= -0.0190542563` (train q0.05)

Execution:
- short; stride 12 bars
- TP 2.5%, SL 1.5%, hold cap 216 bars (18h)

| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | p approx |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17.81% | 4.18% | 5.73% | 0.73 | 125 | 48.8% | 0.115 |
| test2024 | 8.71% | 8.69% | 1.48% | **5.86** | 21 | 61.9% | 0.044 |
| eval2025 | 4.66% | 4.66% | 1.46% | **3.19** | 19 | 57.9% | 0.263 |
| ytd2026 | -1.10% | -2.63% | 3.64% | **-0.72** | 10 | 40.0% | 0.669 |

Activation max phi is `0.1284`, nearest to `short_kimchi_unwind`. It is therefore more orthogonal than the prior FX-stress tight-exit candidate (max phi 0.2615), and almost independent of the long squeeze family.

## Verdict
- Mechanically clears alpha-pool and test/eval live-grade ratio rules.
- Not deployable: only 21/19 OOS trades, weak eval significance, and negative 2026.
- Keep as candidate diversification evidence. Do not tune against 2026 or call it robust/live.

## Artifacts
- `training/search_lowcorr_macro_short_alpha.py`
- `results/lowcorr_macro_short_alpha_scan_2026-07-12.json`
