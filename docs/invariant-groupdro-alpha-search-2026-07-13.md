# Invariant / Group-DRO alpha search (2026-07-13)

## Verdict

2020·2021·2022 각 환경에서 Spearman 방향이 모두 같은 feature만 남기고,
6개 half-year 환경에 ERM/V-REx/Group-DRO tail classifier를 학습했다.
2023 distinct Top-10에서 `alpha_pool` 및 `live_grade` 승격은 **0개**였다.

다만 이전 raw-return/utility online model과 달리, 일부 MLP score는 2024,
2025, 2026에서 모두 양수 Spearman을 유지했다. 약한 예측 신호 자체는 생겼지만
2023 absolute score threshold를 이후 연도에 그대로 적용한 집행이 calibration
drift를 처리하지 못했다.

## Leak-safe protocol

- fit: 2020-2022 only
- train environments: six half-years
- feature admission: 2020/2021/2022 Spearman sign이 모두 같은 feature
- near-duplicate removal: fit absolute Pearson >= 0.95인 feature 제거
- labels: fit 30/70 percentile 기준 short-tail / flat / long-tail 3-class
- models: linear 또는 32-hidden MLP
- objectives: ERM / V-REx 1 / V-REx 10 / Group-DRO
- 2023: score threshold 및 distinct executed-path Top-10 선택
- manifest freeze 후 2024 Test / 2025 Eval / 2026 YTD 계산
- next-bar open entry, 48h hold, 0.5x, 편도 6bp
- full-window CAGR, intratrade adverse excursion strict MDD

총 24개 model/feature 알고리즘과 알고리즘당 12개 방향/threshold policy,
288개 specification을 구성했다. 2023 양수·8회 이상 후보는 133개였다.

## Stable features

상위 8개 train-only invariant feature:

1. `premium_index`
2. `funding_rate`
3. `rex_8640_range_width_pct`
4. `rex_2016_range_pos`
5. `dxy_momentum`
6. `rex_2016_max_to_cur_pct`
7. `trend_96`
8. `htf_3d_range_1`

16/24 feature 집합은 DXY/USDKRW, 단·중기 rolling extrema, candle shape,
taker imbalance, premium/funding z-score 등을 추가한다. 모든 ranking과
중복 제거는 fit 2020-2022에서만 수행했다.

## Top-10 trading result

| Rank | Model / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | MLP V-REx1 stable24, long q90 | 2023 select | +41.28% | 41.31% | 5.27% | 7.84 | 74 |
| 1 | same | 2024 Test | +32.03% | 31.95% | 16.10% | 1.98 | 87 |
| 1 | same | 2025 Eval | +13.66% | 13.67% | 10.93% | 1.25 | 82 |
| 1 | same | 2026 YTD | +4.48% | 11.11% | 11.94% | 0.93 | 39 |
| 2 | MLP V-REx10 stable24, long q90 | 2024 Test | +34.48% | 34.40% | 10.55% | 3.26 | 89 |
| 2 | same | 2025 Eval | -0.30% | -0.30% | 18.37% | -0.02 | 88 |
| 10 | MLP Group-DRO stable24, long q95 | 2024 Test | +25.39% | 25.33% | 10.23% | 2.48 | 60 |
| 10 | same | 2025 Eval | +13.86% | 13.87% | 6.71% | 2.07 | 58 |
| 10 | same | 2026 YTD | -2.90% | -6.82% | 9.03% | -0.76 | 33 |

## Important learnability improvement

Top trading policy는 실패했지만 score ranking 자체는 여러 모델에서 OOS 양수였다.

| Model | 2024 Spearman | 2025 Spearman | 2026 Spearman |
|---|---:|---:|---:|
| MLP ERM stable8 | +0.0786 | +0.0452 | +0.1051 |
| MLP Group-DRO stable8 | +0.0922 | +0.0434 | +0.1301 |
| MLP V-REx1 stable8 | +0.0815 | +0.0431 | +0.1132 |
| MLP Group-DRO stable16 | +0.0825 | +0.0445 | +0.1112 |

이는 이전 River outcome head의 대부분 ±0.1 무작위권 결과보다 안정적이다.
현재 실패의 다음 가설은 feature learnability가 아니라 **연도별 score scale 및
tail calibration drift**다.

## Next branch

모델과 feature 집합은 그대로 두고, 2023 absolute threshold를 재사용하지 않는다.
각 시점의 현재 score를 제외한 과거 180/365일 score rolling percentile만으로
long/short tail을 판정한다. 알고리즘과 percentile은 다시 2023 Top-10으로
고정하고 2024+를 report-only로 유지한다.

## Artifacts

- Search: `training/search_invariant_groupdro_alpha.py`
- Tests: `tests/test_invariant_groupdro_alpha.py`
- Frozen manifest: `results/invariant_groupdro_top10_manifest_2026-07-13.json`
- Result: `results/invariant_groupdro_alpha_scan_2026-07-13.json`
