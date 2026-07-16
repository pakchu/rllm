# POWR-12 사전등록 — Perp-Only Wick Rejection

## 가설

Binance USD-M perp에서만 큰 꼬리가 발생하고 Binance spot은 같은 꼬리를 확인하지 않으며, perp 봉이 꼬리 반대 방향으로 닫히면 derivative-local 강제 체결 또는 유동성 probe가 현물 수요·공급으로 이어지지 않았을 가능성이 있다. POWR-12는 이 사건 이후 한 시간 동안 꼬리 반대 방향을 보유한다.

이는 basis 수준/압축, funding carry, spot-perp absorption grid, phase slip, transfer entropy와 다른 사건이다. OI, funding, 김프, REX, Markov, aggTrade HHI도 신호에 쓰지 않는다.

## 데이터와 한계

- perp 1m: `data/binance_perp_btc_1m_2020_2023.csv.gz`, SHA256 `0b55bb0c3b845a90da738e746c769b19c1de4ac230ca8f1fccb6c361c4a9a41f`
- spot 1m: `data/binance_spot_btc_1m_2020_2023.csv.gz`, SHA256 `bc6e0fd6b773ab6458a5de88fb9589161d1adf4ac1d0e7024f252515909f4a54`
- 기간: 2020-01-01 ~ 2023-12-31
- DB snapshot은 historical backfill이며 point-in-time snapshot이 아니다. exchange timestamp를 의미적 가용 시점으로 사용하고 해시로 현재 snapshot을 고정한다. 실제 승격 전 live forward parity가 필요하다.
- spot 1분 봉 누락은 보간하지 않는다.

각 UTC 5분 버킷은 정확히 연속된 1분 봉 5개로 open=첫 open, high=max, low=min, close=마지막 close를 만든다. 라벨 `t`는 버킷 시작이며 값은 `t+5m`에만 완성된다.

## 단일 정책

```text
upper_v = log(high_v / max(open_v, close_v))
lower_v = log(min(open_v, close_v) / low_v)
body_perp = log(close_perp / open_perp)
upper_excess = max(0, upper_perp - upper_spot)
lower_excess = max(0, lower_perp - lower_spot)
```

excess q95는 직전 complete joint 5분 관측 8,640개만 사용하며 최소 2,016개를 요구한다.

- SHORT: `upper_excess >= prior q95`, `upper_perp >= 6bp`, `upper_spot <= 0.5*upper_perp`, `body_perp <= 0`
- LONG: `lower_excess >= prior q95`, `lower_perp >= 6bp`, `lower_spot <= 0.5*lower_perp`, `body_perp >= 0`
- 두 branch가 동시에 발생하면 거래하지 않는다.

## 실행

- signal bucket `t` 완료: `t+5m`
- 그 다음 joint spot/perp 5분 latency bucket도 정상 완료돼야 함
- latency bucket `t+5m..t+10m`의 완료를 확인한 뒤 진입: 다음 tradable perp open `t+15m`, 즉 signal position +3
- 12봉(60분) 고정 보유 후 예정 perp open 청산
- non-overlap, stop/TP/동적 청산 없음
- 0.5x, 기본 6bp/notional/side, 스트레스 8bp/notional/side
- funding: `entry_time <= funding_time < exit_time`
- future hold 구간의 spot 누락 여부로 과거 진입을 사후 제거하지 않는다.

## 지원도 게이트

- 2020–2022 train 500건 이상, 각 연도 80건 이상
- 2023 40건 이상, H1 20건 이상, H2 10건 이상
- long/short 각각 35–65%
- upper/lower branch 각각 20% 이상
- 단일 월 최대 12%

## 성능 게이트

train과 2023 각각 절대수익 >0, CAGR/strict MDD >=3, strict MDD <=15%, weekly cluster sign-flip p<=0.10, 비용 전 평균 움직임 >12bp, 8bp 스트레스 절대수익 >0을 요구한다. 2023 H1/H2도 각각 양수여야 하며, 진입을 한 봉 더 늦춰도 train/2023 절대수익이 양수여야 한다.

CAGR는 미거래 현금 기간을 포함한다. strict MDD는 global/pre-entry HWM, 보유 중 유리한 극값 후 불리한 극값, funding, 진입/실제 청산/가상 청산 비용을 포함한다.

## 사전 대조군

- 동일 클록 방향 반전: 진단 전용이며 primary 수리·대체·기각 게이트로 사용하지 않음
- spot-only wick
- spot과 perp가 함께 긴 common wick
- spot anchor를 제거한 perp wick
- 진입 1봉 추가 지연
- spot wick을 1시간/1일 stale시킨 비교

spot-only/common/basis-free/stale control 중 하나가 primary 전체 게이트를 독립적으로 통과하면 perp-only 현물 비확인 메커니즘을 기각한다. 실패한 POWR-12의 q95, 6bp, 0.5 비율, 지연, 방향, hold를 수정하지 않는다.

2024·2025·2026 YTD는 봉인한다. pre-2024 통과 후에만 기존 live/shadow 클록과 entry Jaccard, position overlap, 일별 PnL 상관 및 포트폴리오 한계 개선을 검사한다.
