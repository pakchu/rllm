# MOMENT + Mamba2 distributional path critic (2026-07-13)

## Verdict

48h return만 예측하던 구조를 `return / long MAE / short MAE`의
25/50/75% quantile을 예측하는 distributional path critic으로 바꿨다.
2023 Top-10 고정 후 `alpha_pool`/`live_grade` 통과자는 **0개**였다.

## Best evidence

| Rank | Model / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | V-REx median utility λ0.5, long, 365d q95 | 2023 select | +9.72% | 9.73% | 4.75% | 2.05 | 42 |
| 1 | same | 2024 Test | +27.29% | 27.23% | 7.55% | 3.60 | 39 |
| 1 | same | 2025 Eval | -6.01% | -6.02% | 13.98% | -0.43 | 52 |
| 1 | same | 2026 YTD | -1.32% | -3.13% | 11.88% | -0.26 | 17 |
| 6 | ERM median utility λ0.5, long, 180d q70 | 2024 Test | +48.87% | 48.75% | 6.26% | 7.79 | 67 |
| 6 | same | 2025 Eval | +7.88% | 7.89% | 10.44% | 0.76 | 73 |
| 6 | same | 2026 YTD | +3.62% | 8.92% | 13.49% | 0.66 | 28 |
| 9 | V-REx median utility λ0.5, both, 180d q95 | 2024 Test | +17.27% | 17.23% | 11.61% | 1.48 | 76 |
| 9 | same | 2025 Eval | -12.04% | -12.04% | 18.97% | -0.63 | 80 |
| 9 | same | 2026 YTD | +12.13% | 31.67% | 5.98% | 5.30 | 30 |

Path-risk target은 2024 strict MDD를 낮추고 ratio를 크게 올렸지만 2025 raw
score Spearman이 다시 음수였다. 따라서 reward 설계만으로는 고정 parametric
model의 relation flip을 해결할 수 없다.

## Strict protocol

- 2020-2022 fit-only component scale 및 pinball training
- phase1에는 exit가 2024 이전인 path target만 생성
- 2024 이전 MOMENT embedding/inference만 수행
- fit-prefix score NaN, 최소 200 prior score threshold
- 2023 actual executable path Top-10 manifest 선기록
- manifest 이후 full target/embedding 생성, fit scale 동일성 검증
- phase2 2023 path hash 검증 후 OOS metric 계산
- MAE quantile은 de-normalize 후 0 이상으로 clamp
- 비용은 `2 * (fee + slippage) = 12bp`를 utility에 반영

## Artifacts

- Search: `training/search_moment_mamba2_path_critic_alpha.py`
- Tests: `tests/test_moment_mamba2_path_critic_alpha.py`
- Frozen manifest: `results/moment_mamba2_path_critic_top10_manifest_2026-07-13.json`
- Result: `results/moment_mamba2_path_critic_alpha_scan_2026-07-13.json`

