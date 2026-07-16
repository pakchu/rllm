# Wikimedia Attention-Divergence Alpha — Preregistration

## 목적

기존 OI·funding/premium·REX 가격행동 축과 다른 **사람의 외부 관심도**를
거래 시계로 사용한다. 검색 결과를 보기 전에 문서 집합, 지연, 파생식,
14개 정책, 선택 기준과 실패 기준을 고정한다.

## 공식 데이터 계약

- API: [Wikimedia Analytics API — page views](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)
- 가용성 주의: [Wikimedia Analytics API — troubleshooting](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/documentation/troubleshooting.html)
- 접근 정책: [Wikimedia Analytics API — access policy](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/documentation/access-policy.html)
- 고정 집합: `en.wikipedia.org`의 `Bitcoin`, `Ethereum`, `Cryptocurrency`,
  `Blockchain`
- 필터: `all-access/user/daily`; 자동화·spider 트래픽은 제외한다.
- redirect 조회는 원문서에 합산되지 않으므로 제목은 고정하고 중간 변경을
  허용하지 않는다.
- 누락 일자는 0이나 보간값으로 바꾸지 않고 해당 신호를 차단한다.

공식 [pageview_hourly 변경·장애 이력](https://wikitech.wikimedia.org/wiki/Data_Platform/Data_Lake/Traffic/Pageview_hourly)은
2021-06-04~2022-01-26에 전체 webrequest/pageview가 평균 약 2.80~4.34%
유실됐다고 기록한다. 이 공통 측정 충격과 Wikipedia 전체 계절성을 줄이기
위해 각 문서 조회수를 **동일 일자의 en.wikipedia 전체 user pageviews**로
나눈 백만 조회당 비율로 바꾼 뒤 특징을 만든다.

공식 문서는 일별 데이터가 보통 수 시간 내 적재되지만 문제 시 24시간
이상 걸릴 수 있다고 명시한다. 역사적 API 스냅샷에는 실제 적재 시각이
없으므로 D일 값은 **D+2 12:05 UTC**에만 사용하고 다음 5분봉 시가인
D+2 12:10에 진입한다. 이 36시간 지연도 역사적 point-in-time 증명은
아니므로, 성공하더라도 retrieval timestamp를 남기는 forward shadow 전에는
실거래 승격하지 않는다.

## 고정 가설군

1. `broad_attention_reversal`: 4개 문서의 광범위한 관심도 충격과 큰 1일
   가격 이동이 동시에 나타나면 늦은 군중의 소진으로 보고 가격 방향을
   반전한다.
2. `bitcoin_share_reversal`: 전체 crypto 관심 중 Bitcoin 비중이 비정상적으로
   높고 3일 가격 이동이 크면 BTC narrative crowding으로 보고 반전한다.
3. `silent_impulse_continuation`: 큰 가격 이동에도 외부 관심이 평범하거나
   낮으면 retail crowding보다 정보 주도 이동으로 보고 추세를 따른다.

정규화 관심도와 Bitcoin 비중은 현재 D값을 **엄격히 이전 90일**의 median과
`1.4826 * MAD`로 표준화한다. 최소 45일이 없으면 신호를 만들지 않는다.

## 공개 순서

1. 2020–2021 fit + 2022 selection 데이터만 물리적으로 내보낸다.
2. 14개 고정 정책 중 사전 기준을 통과한 1개를 결정하고 manifest를 커밋한다.
3. 그 뒤에만 2023 데이터를 받아 완전 홀드아웃으로 한 번 평가한다.
4. 2023의 CAGR/strict MDD ≥ 3, strict MDD ≤ 15%, 거래 ≥ 6, H1/H2 비음수,
   family-wise block-bootstrap 기준을 모두 통과해야 2024+ 단일 OOS를 연다.

정확한 정책·비용·통계·직교성 기준은
`training/preregister_wikimedia_attention_divergence_alpha.py`가 생성하는
해시된 manifest가 권위 원본이다.

## 실거래 전 필수 직교성

standalone 생존 후에만 현재 3개 sleeve와 실제 entry/position/daily-PnL을
비교한다. exact entry Jaccard ≤ 0.02, ±6시간 근접 진입 ≤ 0.25,
position Jaccard ≤ 0.15, |daily PnL Pearson| ≤ 0.30을 모두 통과하고 기존
포트폴리오의 strict risk-adjusted 성과를 실제로 개선해야 후보가 된다.
