# MOMENT + Mamba2 multi-horizon alpha validation (2026-07-13)

## Verdict

MOMENT PCA32 state의 6시간 단위 전이 32/64개를 tiny Mamba2 state-space
model로 처리하고 12h/48h/7d return을 동시에 학습했다. 2023 Top-10 고정 후
2024/2025/2026 평가에서 `alpha_pool`/`live_grade` 통과자는 **0개**였다.

## Best evidence

| Rank | Model / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | seq32 ERM, 48h head, long, 365d q95 | 2023 select | +29.12% | 29.15% | 4.84% | 6.02 | 41 |
| 1 | same | 2024 Test | +19.96% | 19.91% | 5.40% | 3.69 | 46 |
| 1 | same | 2025 Eval | +2.39% | 2.39% | 9.49% | 0.25 | 51 |
| 1 | same | 2026 YTD | -0.23% | -0.54% | 4.05% | -0.13 | 10 |
| 2 | seq32 ERM, 3-horizon mean, long, 365d q90 | 2024 Test | +9.60% | 9.58% | 13.96% | 0.69 | 57 |
| 2 | same | 2025 Eval | -4.41% | -4.41% | 15.44% | -0.29 | 68 |
| 2 | same | 2026 YTD | +10.68% | 27.62% | 1.94% | 14.22 | 14 |

Rank 1은 2024에서 목표 ratio 3을 넘었지만 2025와 2026에 일반화하지 못했다.
모든 선택된 Mamba score의 Spearman은 2024 양수, 2025 음수, 2026 양수였다.
따라서 단순 point return 예측의 병목은 sequence model 용량이 아니라
**연도별 relation flip과 거래 경로 위험을 target에 반영하지 않은 것**이다.

## Strict protocol

- phase 1 MOMENT embedding과 Mamba inference는 2024 이전 anchor만 수행
- 7d 최대 horizon exit가 2023 이전인 sequence만 학습
- fit prefix score는 전부 NaN; rolling threshold에 in-sample score 미사용
- rolling threshold는 최소 200개 prior score 필요
- 2023 actual executable path Top-10을 먼저 파일로 고정
- manifest 이후 미래 embedding/target 생성 및 선택 model inference
- phase2 2023 executable hash가 고정 manifest와 같아야 OOS 평가 진행
- next-bar 5m open, 48h hold, 0.5x, 편도 6bp
- full-window CAGR, strict intratrade MDD

## Sources and artifacts

- Hugging Face Mamba2 docs: <https://huggingface.co/docs/transformers/model_doc/mamba2>
- MambaTS official implementation: <https://github.com/XiudingCai/MambaTS-pytorch>
- Search: `training/search_moment_mamba2_alpha.py`
- Tests: `tests/test_moment_mamba2_alpha.py`
- Frozen manifest: `results/moment_mamba2_top10_manifest_2026-07-13.json`
- Result: `results/moment_mamba2_alpha_scan_2026-07-13.json`

