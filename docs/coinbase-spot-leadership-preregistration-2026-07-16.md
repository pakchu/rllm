# Coinbase–Binance Venue-Leadership Proxy — Preregistration

## 목적과 주장 한계

현재 포트폴리오의 OI, Binance funding/premium, REX 가격행동과 다른
**미국 USD 현물 거래소의 상대 활동·가격 변화**를 신호 원천으로 사용한다.
레포 검색에서 Coinbase 기반 family는 발견되지 않았다.

공개 과거 캔들에는 taker side와 order book이 없으므로 이 실험은 진정한
가격발견, 주문흐름 불균형 또는 호가 압력을 측정한다고 주장하지 않는다.
오직 완료된 Coinbase `BTC-USD` 5분봉이 다음 Binance perpetual 봉보다 먼저
나타내는 **venue-leadership proxy**가 재현 가능한지를 검증한다.

## 공식 소스와 라이브 동등성

- [Coinbase Exchange: Get product candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
- [Coinbase Exchange REST rate limits](https://docs.cdp.coinbase.com/exchange/rest-api/rate-limits)
- [Coinbase Exchange WebSocket overview](https://docs.cdp.coinbase.com/exchange/websocket-feed/overview)
- [Coinbase Exchange WebSocket channels](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels)
- [Coinbase Exchange: Get product trades](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-trades)

`BTC-USD` 5분봉은 요청당 최대 300개다. API가 요청 범위 밖 봉을 반환할 수
있고 no-tick 구간은 누락될 수 있으므로 exporter는 timestamp를 정렬·중복 제거한
뒤 정확한 구간만 남긴다. 결측은 채우지 않고 해당 봉과 다음 12개 신호봉을
격리한다. selection 파일에는 2023-01-01 이후 값을 넣지 않는다.

라이브에서는 candle 채널이 없으므로 `matches`로 5분봉을 합성하고 `heartbeat`
및 trade ID 연속성을 감시한다. 간극은 public trades REST로 복구하며, 이 forward
WebSocket parity 검증 전에는 라이브 승격하지 않는다.

## 고정 피처

- `ZR`: Coinbase per-bar return − Binance per-bar return의 prior-only robust z
- `ZP`: `log(Coinbase USD close / Binance USDT perp close)`에서 strictly-prior
  3일 median을 제거한 residual change의 prior-only robust z
- `ZV`: 양 거래소 quote notional 중 Coinbase 비중을 logit 변환한 prior-only
  robust z. Coinbase quote notional은 BTC volume × close, Binance는 원본
  `quote_asset_volume`을 사용한다.
- `ZCB`, `ZBN`: 각 venue의 per-bar return prior-only robust z

모든 robust 기준은 현재 봉을 제외한 30일 median/MAD, 최소 14일 관측치다.
USD spot 대 USDT perpetual 비교에는 futures basis와 USD/USDT 변동이 섞이므로
raw ratio를 순수 Coinbase premium이라고 부르지 않는다. premium을 쓰지 않는
return/activity control도 별도로 요구한다.

## 정책과 공개 순서

방향(long/short)과 1·3봉 hold를 미리 고정한 16개 정책만 평가한다. 다섯 family는
relative return, premium shock, activity-confirmed relative return, activity-premium
confluence, return-premium confluence다. 정확한 threshold는 생성 코드가 권위 원본이다.

1. Coinbase와 Binance 입력을 2022-12-31에서 물리적으로 종료한다.
2. forward 수익률 계산 전에 family별 비중첩 event 120개 이상, 연도별 25개 이상,
   양 방향 각각 20% 이상, 월 집중도 20% 이하, 결측/격리 비율을 검사·커밋한다.
3. 2020·2021 fit + 2022 selection에서 연도별 양수, 6개 반기 중 5개 양수,
   연도별 strict MDD ≤ 10%, combined CAGR/strict MDD ≥ 2, 8bp/notional-side
   비용 stress 양수, Bonferroni 보정 weekly cluster sign-flip p < 0.10을 요구한다.
4. 통과 정책 하나를 해시·커밋한 뒤에만 2023을 단일 홀드아웃으로 연다.
5. 2023 절대수익 양수, CAGR/strict MDD ≥ 3, strict MDD ≤ 10%, 거래 ≥ 20,
   H1/H2 비음수, 비용·지연·보정 유의성 gate를 모두 통과해야 한다.
6. 그 뒤에만 현재 sleeve와 entry/position/daily-PnL 직교성 및 포트폴리오 한계
   기여를 측정한다. 2024+는 계속 봉인한다.

## 비용과 strict MDD

0.5× exposure에서 fee 5bp + slippage 1bp는 **notional 기준 편도 6bp**이며
account 기준 편도 3bp다. 진입은 다음 5분봉 open, 비중첩 fixed-time exit다.
strict MDD는 진입 전/global HWM, 보유 중 favorable-before-adverse OHLC 경로,
진입·가상청산 비용과 실현 funding debit을 포함한다. CAGR은 거래하지 않은 기간도
포함한 전체 달력 시간으로 계산한다.

정확한 고정값과 해시는
`training/preregister_coinbase_spot_leadership_alpha.py`가 생성하는 manifest가
권위 원본이다.
