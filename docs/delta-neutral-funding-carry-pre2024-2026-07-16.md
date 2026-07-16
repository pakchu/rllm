# Delta-neutral funding carry: pre-2024 frozen search

## 결론

- 상태: **reject_pre2024**
- 2024년 이후 행은 열지 않았다(`oos_rows_opened=0`).
- 구조: Binance BTCUSDT 현물 롱 + USD-M 무기한 숏의 BTC 수량을 정확히 동일하게 맞추고 합산 gross를 1배로 유지.
- 정책 입력: 이미 정산된 funding-rate의 trailing mean만 사용하고 다음 5분봉 open에 집행.
- DB 과거 현물/펀딩은 backfill된 비-PIT 스냅샷이므로 live forward proof 전에는 운영 승격 금지.
- 분리 지갑에서 선물 담보가 고갈될 수 있으므로 통합마진/자동 담보이체 없이는 운영 승격 금지.

## 선택 정책

```json
{
  "entry_threshold": 5e-05,
  "exit_threshold": 0.0,
  "lookback_events": 21,
  "min_hold_events": 3
}
```

## Pre-2024 성능

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020h1 | 3.3911% | 6.9216% | 11.2793% | 0.6137 | 2 | 129 | 384 |
| fit_2020h2 | 4.0676% | 8.2361% | 4.7612% | 1.7298 | 3 | 171 | 507 |
| fit_2021h1 | 11.4852% | 24.5324% | 9.6856% | 2.5329 | 2 | 172 | 514 |
| fit_2021h2 | 3.2355% | 6.5251% | 6.4653% | 1.0092 | 1 | 152 | 455 |
| fit_2022h1 | -0.2162% | -0.4357% | 4.7095% | -0.0925 | 5 | 128 | 374 |
| fit_2022h2 | 0.2257% | 0.4484% | 7.8061% | 0.0574 | 3 | 122 | 358 |
| fit_2020_2022 | 24.4777% | 7.5697% | 11.2793% | 0.6711 | 13 | 874 | 2592 |
| select_2023h1 | 1.3748% | 2.7937% | 3.2708% | 0.8541 | 1 | 181 | 542 |
| select_2023h2 | 1.4600% | 2.9191% | 4.3945% | 0.6643 | 2 | 123 | 363 |
| select_2023 | 3.0352% | 3.0373% | 4.3945% | 0.6912 | 2 | 304 | 906 |

## 통제군

### always_carry

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020_2022 | 25.7988% | 7.9488% | 18.8048% | 0.4227 | 1 | 1096 | 3287 |
| select_2023 | 3.2971% | 3.2994% | 6.8088% | 0.4846 | 1 | 365 | 1094 |

### inverted_gate

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020_2022 | -2.0696% | -0.6945% | 18.3726% | -0.0378 | 8 | 74 | 200 |
| select_2023 | 0.0000% | 0.0000% | 0.0000% | 0.0000 | 0 | 0 | 0 |

### decision_delayed_1_event

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020_2022 | 24.0332% | 7.4415% | 11.2793% | 0.6598 | 13 | 878 | 2591 |
| select_2023 | 3.0205% | 3.0226% | 4.4081% | 0.6857 | 2 | 303 | 906 |

### decision_delayed_3_events

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020_2022 | 23.9126% | 7.4067% | 11.2793% | 0.6567 | 13 | 873 | 2589 |
| select_2023 | 3.0309% | 3.0330% | 4.3985% | 0.6895 | 2 | 304 | 906 |

### basis_only_zero_funding

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020_2022 | -4.1663% | -1.4082% | 13.8546% | -0.1016 | 13 | 874 | 2592 |
| select_2023 | -0.7572% | -0.7577% | 4.5929% | -0.1650 | 2 | 304 | 906 |

### double_execution_cost

| 구간 | 절대수익 | CAGR | strict MDD | CAGR/MDD | 에피소드 | active days | funding events |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit_2020_2022 | 19.4228% | 6.0938% | 11.2829% | 0.5401 | 13 | 874 | 2592 |
| select_2023 | 2.2697% | 2.2713% | 4.5751% | 0.4965 | 2 | 304 | 906 |

## 엄격성/누수 계약

- 모든 창은 flat equity=1로 시작하며 과거 funding gate 상태만 전달한다.
- funding event는 해당 시각보다 엄격히 뒤의 첫 5분봉 open에서만 gate를 바꾼다.
- event 당시 이미 보유한 short만 funding을 받는다; 그 event로 진입한 포지션은 받지 않는다.
- funding mark는 event 시각까지 완전히 끝난 마지막 선물 5분봉 close를 일관되게 사용한다.
- strict MDD는 1분 내 비동시 basis dislocation까지 포함해 spot-high/perp-low HWM 뒤 spot-low/perp-high adverse를 적용한다.
- 현물 누락/부분봉은 직전 완성 basis와 선물 OHLC로 복원하고 high/low를 고정 cushion만큼 확대한다.
- 진입·청산·일일 리밸런싱 모두 두 leg의 실제 변경 notional에 fee+slippage를 부과한다.
- CAGR 분모는 거래/보유일이 아니라 전체 달력 기간이다.

## 직교성 판단

이 단계에서는 방향성 알파와 다른 경제 메커니즘 및 일별 BTC beta를 확인한다. 기존의 entry/position Jaccard gate는 장시간 보유하는 market-neutral sleeve에 그대로 적용할 수 없으므로, frozen OOS 이후 동일 일별 손익의 Pearson 상관과 포트폴리오 한계기여를 주 판정으로 사용한다.

## 소스 진단

```json
{
  "complete_spot_bars": 2101493,
  "funding_events": 4383,
  "funding_mark_policy": "use actual reported funding mark when finite and positive; otherwise use the last fully completed USD-M one-minute close whose end <= funding_time",
  "funding_missing_reported_mark": 4198,
  "funding_settlement_marks_fallback": 4198,
  "funding_settlement_marks_reported": 185,
  "funding_without_causal_fallback": 1,
  "market_rows": 2103840,
  "max_consecutive_proxy_bars": 354,
  "missing_spot_bars": 2347,
  "oos_rows_opened": 0,
  "partial_spot_bars": 0,
  "proxy_rule": "last complete prior one-minute spot/perp close basis; futures OHLC; high/low widened by 0.002500; observed partial points retained in extrema",
  "proxy_spot_bars": 2347,
  "reported_vs_causal_mark_comparisons": 185,
  "reported_vs_causal_mark_median_abs_error_bps": 0.2092908144935013,
  "reported_vs_causal_mark_p99_abs_error_bps": 2.442995094408084
}
```
