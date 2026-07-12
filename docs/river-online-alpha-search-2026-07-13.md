# River delayed-feedback online alpha search (2026-07-13)

## Verdict

River 0.25.0의 ARF+ADWIN, Hoeffding Adaptive Tree, online linear model을
48시간 지연 라벨로 prequential 평가했다. 2023에서 고정한 Top-10 가운데
`alpha_pool` 및 `live_grade` 승격 후보는 **0개**였다.

정적 TabICLv2의 연도 간 drift를 온라인 적응으로 해결할 수 있는지 확인한
실험이며, 결과상 **raw 48h return regression 자체가 충분히 학습 가능한
안정 신호가 아니었다**. 모델 업데이트 주기만 온라인으로 바꾸는 것은 근본
해결책이 아니었다.

## Leak-safe protocol

- 신호 간격: 6시간 (`72 x 5m`)
- 진입: 신호 다음 5분봉 시가
- 라벨: 진입 시가부터 48시간 뒤 시가까지의 log return
- 라벨 공개: `signal_pos + 1 + 576` 이후에만 `learn_one`
- 순서: 공개 완료 라벨 학습 -> 현재 시점 예측 -> 현재 샘플 pending 등록
- 모델 warm-up: 완료 라벨 300개
- gate: 현재 점수를 `shift(1)`로 제외한 과거 180일/365일 rolling quantile
- 선택: 2023 strict CAGR/MDD로 distinct Top-10 고정
- 중복 제거: **2023 선택 구간 신호만 hash**; 2024+ 신호는 hash에 미포함
- 평가: 2024 Test, 2025 Eval, 2026 YTD prequential OOS
- 비용: 편도 6bp
- 수익률: 무거래 기간을 포함한 전체 평가창 CAGR
- 위험: 진입 전 equity와 포지션 보유 중 adverse OHLC excursion을 포함한 strict MDD

`results/river_online_top10_manifest_2026-07-13.json`은 2024+ trading metric을
계산하기 전에 기록됐고 `later_metrics_included=false`를 가진다.

## Models

| Family | Feature groups | Adaptation |
|---|---|---|
| Adaptive Random Forest | compact / price / full | ADWIN drift=0.001, warning=0.01 |
| Hoeffding Adaptive Tree | compact / full | ADWIN drift=0.002 |
| Online Linear Regression | compact / full | rolling StandardScaler + SGD |

총 7개 model/feature 조합, 2개 score window, 5개 quantile, 3개 방향
policy를 구성했다. 2023에서 양수 수익 및 8회 이상 거래를 만족한 후보는
100개였고, 중복 제거 후 Top-10을 고정했다.

## Top candidates

| Rank | Model / policy | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | linear compact long, 365d q95 | 2023 select | +28.69% | 28.72% | 3.86% | 7.44 | 19 |
| 1 | linear compact long, 365d q95 | 2024 Test | +5.31% | 5.30% | 1.14% | 4.63 | 2 |
| 1 | linear compact long, 365d q95 | 2025 Eval | +10.46% | 10.47% | 1.99% | 5.27 | 8 |
| 1 | linear compact long, 365d q95 | 2026 YTD | -4.21% | -9.83% | 10.53% | -0.93 | 17 |
| 6 | ARF compact long, 180d q95 | 2024 Test | +18.77% | 18.72% | 10.53% | 1.78 | 58 |
| 6 | ARF compact long, 180d q95 | 2025 Eval | +19.84% | 19.85% | 5.58% | 3.56 | 43 |
| 6 | ARF compact long, 180d q95 | 2026 YTD | -12.13% | -26.71% | 13.43% | -1.99 | 17 |

Rank 1은 2024 거래가 2회라 통계 기준을 충족하지 못했고, 모든 Top-10이
2026에서 음수였다. ARF compact는 전체 stream에서 drift 18회, warning
43회를 감지했지만 적응 이후에도 2023/2024/2026 score Spearman이 각각
-0.043/-0.051/-0.049였다.

## Interpretation and next branch

온라인 적응 실패는 “더 자주 재학습하면 된다”는 가설을 기각한다. 다음
실험은 raw return 회귀를 반복하지 않는다.

1. 동일한 causal state에서 `long / short / flat`의 실행 가능 보상을 각각 계산한다.
2. 48시간 path가 끝난 뒤에만 행동별 net return과 MAE 기반 utility를 공개한다.
3. delayed-feedback contextual bandit/utility model은 방향과 abstain을 공동 학습한다.
4. 2023 Top-10 family를 먼저 고정하고 2024+는 동일한 strict simulator로 평가한다.
5. LLM은 숫자 가격 예측기가 아니라 event/state ontology 및 후보 가설 생성기로만 사용한다.

## Reproduction

```bash
uv run --no-project --python /usr/bin/python3.12 --script \
  training/search_river_online_alpha.py \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --manifest-output results/river_online_top10_manifest_2026-07-13.json \
  --output results/river_online_alpha_scan_2026-07-13.json
```

Source: <https://github.com/online-ml/river>
