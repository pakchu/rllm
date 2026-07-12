# Bidirectional BTC state alpha scan (2026-07-12)

## Protocol
- One standalone policy can enter BTCUSDT long or short; one position at a time.
- Train `<2024` quantile thresholds, test2024 selection, eval2025/YTD2026 reporting.
- 6bp/side, 0.5x, strict intrabar MDD, next-open entry, conservative stop-before-take ordering.
- Both directions required in test and eval.
- 2,456 eligible variants.

## Qualifier: funding relief vs FX stress

Long entry:
- `funding_rate <= -0.0000167`
- taker-flow acceleration `>= 0.0408248038`

Short entry:
- `usdkrw_momentum >= 0.0026584831`
- `htf_1d_return_1 <= -0.0340374575`

Execution: TP4%, SL2.5%, cap 288 bars, stride 6.

| split | return | CAGR | strict MDD | ratio | trades L/S | win L/S | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 28.90% | 6.55% | 17.08% | 0.38 | 292/93 | 47.6/45.2% | 1.10 |
| test2024 | 19.17% | 19.13% | 4.00% | **4.78** | 34/22 | 55.9/59.1% | 1.97 |
| eval2025 | 15.27% | 15.28% | 5.77% | **2.65** | 54/8 | 55.6/50.0% | 1.69 |
| ytd2026 | -0.54% | -1.29% | 12.17% | **-0.11** | 74/6 | 48.6/50.0% | -0.01 |

Direction ablation shows both isolated legs were positive in 2026 (long +1.15%, short +1.58%), but the one-position-at-a-time combined policy was -0.54%. Signal overlap/opportunity blocking therefore matters and the dual policy is not robust/live-ready.

## Verdict
- Clears alpha-pool test/eval ratio>=2.5.
- Does not clear live-grade because eval ratio is below 3.
- Fails 2026 and becomes heavily long-skewed. Candidate-only; no live promotion.

## Artifacts
- `training/search_bidirectional_state_alpha.py`
- `results/bidirectional_state_alpha_scan_2026-07-12.json`
- `results/bidirectional_state_alpha_direction_ablation_2026-07-12.json`
