# OFBR-1: OI failed-breakdown recoil preregistration

## 목표

Gross9의 단순 배율 변경이 아니라, **하락 충격 뒤 신규 숏 재고가 남아 있는데
가격이 더 내려가지 못하고 체결 흐름이 회복되는 순간**을 별도 롱으로 거래한다.
핵심 평가는 단독 수익률이 아니라 Gross9의 절대수익과
`CAGR / strict MDD` 한계기여다.

## 사전 고정 메커니즘

앵커는 완료된 5분봉에서 다음 네 조건을 모두 만족한다.

1. 4시간 BTC 수익률이 calibration q20 이하
2. 한 봉 지연한 4시간 OI-가격 divergence가 q75 또는 q90 이상
3. 1시간 taker imbalance가 q20 또는 q35 이하
4. 현재 종가의 4시간 range position이 q25 이하

앵커 30분 또는 60분 뒤 완료봉에서 다음을 확인한다.

- 종가가 앵커 종가를 회복
- 최근 15분 taker flow가 앵커 시점 1시간 flow보다 개선
- 한 봉 지연 OI가 `-0.5%` 또는 `0%` retention 기준 이상
- 앵커 이후 저점이 사전 ATR의 1.5배보다 깊지 않음
- 종가가 앵커 이후 범위의 상단 65% 또는 80% 이상

확인 다음 5분봉 시가에 `0.5x LONG` 진입하고 3/6/12시간 고정 보유한다.
TP/SL은 쓰지 않는다. OI 결측은 stale fill하지 않고 신호를 차단한다.

## 동결된 96-cell 검색

| 축 | 값 |
|---|---|
| OI-price divergence | q75, q90 |
| sell-flow tail | q20, q35 |
| 확인 대기 | 6, 12 bars |
| OI retention | -0.5%, 0% |
| post-anchor range position | 0.65, 0.80 |
| hold | 36, 72, 144 bars |

가격 q20, range q25, 30분 anchor stride, ATR 1.5배는 고정한다.

## 데이터 격리

- threshold calibration: `2020-09-01 .. 2022-12-31`
- pre-2024 후보 강건성: calendar 2023
- top-1 선택: calendar 2024
- 2025/2026: 고정 top-1 veto만 가능
- 2025/2026 결과를 본 뒤 threshold, hold, 대표 후보, weight를 바꾸지 않는다.

현재 연구 이력상 2025/2026은 전역적으로 pristine하지 않다. 따라서 통과해도
즉시 live 승격하지 않고 forward shadow 후보로만 취급한다.

## 회계

- 완료봉 판단, 다음 5분봉 시가 진입
- OI는 한 완료봉 지연
- 0.5x, 비용 6bp/notional/side
- stress 10bp/notional/side
- split-contained exit
- 무포지션 기간도 전체 calendar CAGR에 포함
- same-BTC OHLC `upper-before-lower` global strict MDD

## 통과 기준

2024 top-1은 단독 `CAGR/MDD >= 3`, MDD `<=15%`, 12건 이상이어야 한다.
또한 weight 0.25를 Gross9에 더했을 때 2024 절대수익을 낮추지 않으면서
포트폴리오 `CAGR/MDD`를 개선해야 한다.

고정 top-1은 2025와 2026 각각 양수, `CAGR/MDD >= 3`, MDD `<=15%`,
10bp stress 양수여야 하며 Gross9의 각 미래 구간 절대수익과
`CAGR/MDD`를 모두 훼손하지 않아야 한다.

실행 시점 독립성은 entry Jaccard, ±6시간 overlap, position-bar Jaccard,
일별 MTM 상관, Gross9 최악 drawdown bar 기여로 별도 검증한다.

기계 판독 원문:
`results/oi_failed_breakdown_recoil_preregistration_2026-07-28.json`.
