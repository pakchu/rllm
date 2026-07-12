# Chronos-2 zero-shot alpha search (2026-07-13)

## Verdict

Amazon Chronos-2 120M을 repo 데이터로 fine-tune하지 않고, 완료된 1시간봉
720개와 8개 past covariate로 48시간 log-price quantile forecast를 생성했다.
2023에서 고정한 Top-10의 `alpha_pool`/`live_grade`는 **0개**였다.

Raw forecast를 가격 방향 그대로 거래하는 방식은 유효하지 않았다. 그러나
모든 score stream의 fit 2020-2022 Spearman이 음수였고 2023, 2024, 2026도
대부분 같은 부호였다. 다음 branch는 OOS가 아니라 fit 부호만 사용해 forecast
orientation을 사전 보정한다.

## Why Chronos-2

- 120M encoder-only time-series foundation model
- native univariate, multivariate, past/future covariate support
- maximum context 8192, prediction length 1024
- Apache-2.0
- repo data fine-tuning 없음
- 실행 peak VRAM 약 2.9GiB

Official sources:

- <https://huggingface.co/amazon/chronos-2>
- <https://github.com/amazon-science/chronos-forecasting>

## Causal input

- 5분봉 timestamp는 candle open으로 간주
- hourly bin `[H-1h, H)`를 `H` 시점 완료 데이터로 label
- hour boundary에서 새로 열린 5분봉은 이전 hourly candle에 포함하지 않음
- signal bar close 시점과 completed hour가 일치하는 6시간 anchor
- target: hourly `log_close`
- context: 720h (30일)
- horizon: 48h
- past covariates:
  - log quote volume
  - taker imbalance
  - hourly range
  - funding rate
  - premium index
  - DXY z-score
  - kimchi premium z-score
  - USDKRW z-score

총 9,362 anchors 중 9,244개를 forecast했다. `cross_learning=false`라 batch
구성에 따라 live prediction이 달라지지 않는다.

## Model identity

- Model: `amazon/chronos-2`
- Revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Package: `chronos-forecasting==2.3.1`
- Fine-tuned on repo data: no

## Top raw policy

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 select | +37.54% | 37.57% | 8.88% | 4.23 | 88 |
| 2024 Test | +15.31% | 15.28% | 18.46% | 0.83 | 102 |
| 2025 Eval | -9.65% | -9.66% | 13.96% | -0.69 | 86 |
| 2026 YTD | -2.43% | -5.74% | 19.19% | -0.30 | 39 |

Configuration: `median_24h_48h_consensus`, long-only, prior 365d q70.

## Forecast rank diagnostics

| Score | Fit 2020-22 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|
| median terminal | -0.0296 | -0.0582 | -0.0005 | +0.0123 | -0.0413 |
| central terminal | -0.0271 | -0.0592 | -0.0300 | +0.0244 | -0.0311 |
| median path mean | -0.0342 | -0.0486 | -0.0127 | +0.0021 | -0.0501 |
| interval SNR | -0.0355 | -0.0578 | -0.0166 | +0.0151 | -0.0649 |

## Next branch

각 score의 orientation을 fit 2020-2022 Spearman 부호로만 고정한다.
fit correlation이 음수면 `oriented_score = -raw_score`, 양수면 그대로 둔다.
이후 rolling percentile과 2023 Top-10 selection을 새 manifest에서 다시 수행한다.

## Artifacts

- Search: `training/search_chronos2_zero_shot_alpha.py`
- Tests: `tests/test_chronos2_zero_shot_alpha.py`
- Frozen manifest: `results/chronos2_zero_shot_top10_manifest_2026-07-13.json`
- Result: `results/chronos2_zero_shot_alpha_scan_2026-07-13.json`
