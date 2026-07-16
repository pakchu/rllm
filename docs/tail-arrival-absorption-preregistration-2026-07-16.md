# Tail–Arrival Absorption/Release Alpha — Preregistration

## 목적

현재 live portfolio의 OI·funding/premium·REX와 다른 거래 축을 사용한다.
레포 검색상 `event_notional_p50/p90/p99/max`, `event_notional_mean/std`,
`interarrival_mean_ms/std_ms`는 데이터 감사·생성 외에는 알파 입력으로 사용된 적이
없다. 이 family는 **공격 거래 이벤트 크기의 꼬리 모양과 도착시간 불규칙성**만을
주요 상태로 사용한다.

과거 전체 시장수익은 다른 연구에서 이미 알려져 있으므로 2023을 globally pristine이라고
주장하지 않는다. 다만 이 exact 4-policy family의 2023 결과는 아직 열지 않고,
2020–2022 selection 입력을 물리적으로 잘라 커밋한 뒤 통과 정책 하나만 2023에서
단 한 번 검증한다. 최종 증거는 forward live다.

## 가설

5분 동안 일부 공격 거래만 극단적으로 커지고 도착 간격도 불규칙해지면 평범한
flow imbalance와 다른 packetized-liquidity state가 된다.

- 큰 packet 방향으로도 가격이 움직이지 못하면 passive side가 흡수한 것으로 보고
  큰 packet의 반대 방향을 1시간 또는 3시간 보유한다.
- 큰 packet 방향으로 가격 반응까지 강하면 liquidity release로 보고 같은 방향을
  1시간 또는 3시간 보유한다.

participant identity, hidden order, resting-book liquidity를 직접 관측한다고 주장하지
않는다. 모두 공개 aggTrade의 완료된 통계에서 얻는 proxy다.

## 고정 피처와 정책

- tail span: `log(p99/p50) + 0.5*log(max/p99)`
- event dispersion: `log1p(std/mean)`
- arrival CV: `log1p(interarrival_std_ms/interarrival_mean_ms)`
- packet direction: buy/sell 평균 event size log-ratio의 부호
- signed response: packet direction × 현재 5분 micro return

threshold는 현재 봉을 제외한 이전 30일의 95/75/75/80 분위수이며 최소 7일
관측치를 요구한다. 같은 branch가 최근 12봉에 없을 때만 episode start로 인정한다.
정책은 absorption/release × hold 12/36봉의 정확히 4개다.

## 라이브·누수 계약

- 공식 Binance UM daily aggTrades archive를 사용하고 checksum/gap manifest를 검증한다.
- 2020–2022 selection 파일은 원본의 첫 date 필드만 읽다가 2023 sentinel에서
  non-date CSV parsing 전에 중단한다.
- source-gap day, 결측 feature bar와 이후 24봉을 격리하고 채우지 않는다.
- 완료봉 `t` 직후 정확한 next-open fill을 가장하지 않는다. 계산·전송 시간을 위해
  한 봉을 비우고 **`t+2` open**에 진입한다.
- 0.5×, notional 편도 6bp, 8bp stress, realized funding과 strict held-path MDD를 쓴다.

## 공개 순서

1. 이 문서와 4-policy manifest를 커밋한다.
2. 2020–2022 feature/market/funding prefix와 support clock을 결과 없이 동결한다.
3. policy별 총 120회·연 25회·양 방향 20%·월 집중 20% gate를 통과한 경우에만
   2020·2021 fit + 2022 selection 수익률을 연다.
4. 연도별 양수, 6반기 중 5개 양수, 연 MDD ≤10%, combined CAGR/MDD ≥2,
   비용 stress 및 Bonferroni weekly sign-flip을 모두 통과한 한 정책만 고정한다.
5. 고정 정책 commit 뒤에만 2023을 연다. 목표는 절대수익 양수,
   CAGR/strict MDD ≥3, MDD ≤10%, 거래 ≥20, H1/H2 비음수다.
6. 통과 후에만 기존 3개 sleeve와 entry/position/daily-PnL 직교성을 측정한다.

정확한 계약과 해시는
`training/preregister_tail_arrival_absorption_alpha.py`가 권위 원본이다.
