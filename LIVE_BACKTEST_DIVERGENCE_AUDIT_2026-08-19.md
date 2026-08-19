# 라이브·백테스트 신호 괴리 감사 보고서

- 감사 기준일: 2026-08-19 UTC
- 감사 대상 커밋: `c71b76bf`
- 라이브 설정: `configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json`
- 조사 방식: 현재 PostgreSQL 데이터와 현재 라이브 scorer를 frozen 연구 결과 및 manifest와 비교
- 변경 범위: 본 감사에서는 라이브 프로세스, 설정 및 거래 상태를 변경하지 않음

## 1. 결론

현재 라이브 신호 계산과 frozen 연구 백테스트 사이에는 **실제 괴리가 존재한다**.

가장 명확한 증거는 다음 세 가지다.

1. `frozen_annual_rank7`의 현재 activation schedule이 frozen manifest와 일치하지 않는다.
2. 2026년 동일 구간 재생에서 총 거래 수는 우연히 108회로 같지만, 슬리브별 거래 수와 포트폴리오 손익·MDD가 다르다.
3. 라이브 tail feature cache는 calendar-aligned 3일·주간 피처에서 full compute와 수치적으로 동일하지 않다.

현재 코드와 현재 DB로 2026-07-19 이후 신호 0은 재현된다. 그러나 이는 현재 구현이 자기 자신과 일관된다는 뜻일 뿐, frozen 연구 계약을 정확하게 재현한다는 증거는 아니다. 특히 Rank7 OI 입력 계약과 REX taker frozen source가 재현되지 않아 최근 장기 무거래를 정상이라고 확정할 수 없다.

## 2. 현재 라이브 구성

현재 live config의 gross exposure는 8.0이며 다음 5개 슬리브로 구성된다.

| Sleeve | Weight |
|---|---:|
| `fresh_kimchi_fx` | 2.0 |
| `frozen_annual_rank7` | 2.0 |
| `rex_taker_low_range_position` | 0.4 |
| `cand_rex_veto_7` | 1.6 |
| `markov_transition_long` | 2.0 |

해당 config에는 `research_contaminated: true`가 명시되어 있으므로, 저장된 연구 성과를 완전한 out-of-sample 성과로 해석하면 안 된다.

## 3. 확정된 괴리

### 3.1 Rank7 activation schedule 불일치

Frozen bundle manifest:

- raw selected events: 16개
- research accepted trades: 12개

현재 PostgreSQL 데이터와 현재 runtime 재구성:

- raw active events: 18개
- accepted trades: 13개
- 18개 모두 source-ready 상태이므로 freshness gate 탈락으로 설명되지 않음

Expected-only timestamp:

- `2026-03-04 01:00 UTC`
- `2026-04-09 17:00 UTC`
- `2026-04-29 10:00 UTC`

Current-only timestamp:

- `2026-02-14 16:00 UTC`
- `2026-03-04 00:00 UTC`
- `2026-04-09 16:00 UTC`
- `2026-04-14 16:00 UTC`
- `2026-04-17 16:00 UTC`

공통 timestamp는 13개다. 일부 신호는 1시간 이동했고 추가·누락 신호가 함께 존재한다.

가장 강한 원인은 OI source contract drift다.

- frozen 연구는 hash가 고정된 market/OI cache를 사용했다.
- 현재 라이브는 PostgreSQL historical OI와 live snapshot을 5분 bin 단위로 혼합한다.
- mutable `/tmp` OI CSV도 frozen 연구 source와 hash 및 state value가 달라 원본 대체재가 아니다.
- 원래 frozen OI source 파일이 현재 checkout에 없어 정확한 행 단위 원인까지는 복원할 수 없다.

관련 근거:

- `artifacts/rank7/frozen_annual_rank7_2026/manifest.json`
- `execution/rank7_runtime.py:589-790`
- `execution/portfolio_live.py:1482-1531`
- `training/backtest_added_alpha_month.py:529-591`

### 3.2 동일 총 거래 수가 숨긴 포트폴리오 차이

2026-01-01부터 2026-06-01까지 현재 DB와 현재 live adapter를 재생한 결과를 frozen 연구 결과와 비교했다.

| Metric | Current DB replay | Frozen research | Difference |
|---|---:|---:|---:|
| Return | +66.92484% | +69.23856% | -2.31372%p |
| Strict MDD | 17.52464% | 15.00097% | +2.52367%p |
| Trades | 108 | 108 | 0 |
| Win rate | 64.8148% | 65.7407% | -0.9259%p |

슬리브별 거래 수:

| Sleeve | Current DB replay | Frozen research |
|---|---:|---:|
| Fresh | 28 | 28 |
| Rank7 | 13 | 12 |
| REX taker | 22 | 23 |
| cand REX | 22 | 22 |
| Markov | 23 | 23 |

총 거래 수가 동일한 것은 Rank7 `+1`과 REX taker `-1`이 상쇄됐기 때문이다. 따라서 aggregate trade count만으로 parity를 판정하면 실제 schedule drift를 놓친다.

Frozen 기준 결과:

- `results/portfolio_added_alpha_update_2026-07-16.json`

### 3.3 REX taker 결과 불일치

| Metric | Current DB replay | Frozen result |
|---|---:|---:|
| Trades | 22 | 23 |
| Return | +3.2758% | +6.9811% |

현재 config hash와 현재 scalar/vector 계산은 일관되지만, frozen parity에서 사용한 REX event JSONL 및 source market cache가 현재 checkout에 없다. 그러므로 누락된 정확한 timestamp와 차이가 발생한 feature row는 현재 로컬 자료만으로 확정할 수 없다.

관련 근거:

- `results/rex_taker_rangepos_gate_strict_audit_2026-07-13.json`
- `results/portfolio_added_alpha_shadow_signal_parity_2026-07-16.json`
- `configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json`

### 3.4 Tail feature cache와 full compute 불일치

라이브 feature cache는 최근 8,640개 5분봉, 약 30일만 잘라 재계산한다.

- tail context: `execution/portfolio_live.py:87-89`
- tail refresh: `execution/portfolio_live.py:682-713`
- calendar HTF 계산: `preprocessing/market_features.py:158-255`

이 tail이 임의의 5분 timestamp에서 시작하기 때문에 3일·주간 resample 경계와 이전 shifted period history가 full frame과 달라질 수 있다.

20,000개 5분봉을 이용한 regression probe에서 최근 96개 행 모두 다음 피처 중 일부가 불일치했다.

| Feature | Maximum absolute difference |
|---|---:|
| `htf_3d_return_4` | 0.00028774 |
| `htf_3d_range_pos` | 0.19158 |
| `htf_1w_return_4` | 0.004741 |
| `htf_1w_range_pos` | 0.099995 |
| `weekly_range_pos` | 0.099995 |

기존 테스트 `tests/test_portfolio_live_selector.py:187-273`은 8,900개 행만 사용해 충분한 calendar history를 통과하지 않으므로 이 결함을 검출하지 못한다.

현재 5개 live sleeve는 문제 피처를 내부에서 다시 계산하거나 override하는 경로가 있어, 2026-07-19 이후의 실제 signal bit에서는 cached/cold-full/long-lookback 간 차이가 나오지 않았다. 따라서 이 결함은 확정된 미래 위험이지만 현재 장기 무거래의 직접 원인으로 확인되지는 않았다.

## 4. 정상으로 확인된 영역

### 4.1 현재 라이브 루프와 데이터 freshness

감사 시점의 live state에서 다음을 확인했다.

- expected decision bar와 latest bar 일치
- `decision_bar_complete=true`
- `source_freshness_missing=[]`
- open sleeve 없음
- frame build와 scoring cycle 정상 완료

따라서 프로세스 정지, 미완성 봉 global fail-close 또는 전체 데이터 freshness gate가 최근 무거래를 만든 증거는 없다.

### 4.2 현재 코드 내부 scalar/vector parity

현재 DB에서 2026-07-01 이후 알려진 nonzero vector signal 7개를 scalar live scorer로 재계산한 결과 timestamp와 방향이 모두 일치했다.

- REX taker short: 6개
- cand REX short: 1개
- mismatch: 0개

2026-07-19 이후 9,118개 bar에서도 cached, cold-full 및 더 긴 150,000-bar lookback의 최종 signal bit는 5개 sleeve 모두 0으로 일치했다.

이는 현재 live wrapper의 추가 gate가 신호를 임의로 제거하는 global bug는 없다는 증거다. 다만 입력 source 자체가 frozen 연구와 다르면 세 계산 경로가 동일하게 잘못된 0을 만들 수 있다.

### 4.3 cand REX control

Frozen base-event JSONL과 현재 DB reconstruction을 비교한 결과:

- frozen base candidates: 81개
- current base candidates: 81개
- timestamp set: 완전 일치
- maximum strength difference: 약 `4.98e-14`

따라서 BTC market 기반 REX base math 전체에 적용되는 전역 timestamp/stride 오류 가능성은 낮다. 주요 불확실성은 Rank7의 OI source와 REX taker의 누락된 frozen 입력에 집중된다.

## 5. 최근 장기 무거래 평가

현재 계산상 마지막 nonzero signal은 `2026-07-11 16:55 UTC`이고, 감사 기준 시점까지 약 38.95일간 신규 신호가 없다.

현재 DB 기준 2026년 과거 통계:

- accepted trades: 108회
- 관측 기간: 약 150.63일
- 평균 거래 빈도: 약 0.717회/일
- 과거 최대 union inter-trade gap: 약 13.08일

현재 gap은 과거 최대 gap의 약 3배다. 신호가 비정상적으로 군집될 수 있으므로 단순 Poisson 확률을 결론으로 사용해서는 안 되지만, 이 정도 gap은 별도 parity 검증 없이 정상이라고 분류하기에는 충분히 이례적이다.

## 6. 기존 parity 결과의 한계

`results/portfolio_added_alpha_shadow_signal_parity_2026-07-16.json`은 Fresh, Markov, REX taker 및 cand REX의 frozen source domain 내 decision parity를 통과했다.

그러나 다음은 검증 범위에 포함되지 않았다.

- Rank7
- 주문 제출 및 maker fill
- stale order 갱신과 cancel
- position netting 및 live state
- 실제 slippage와 exchange-side rejection
- 현재 PostgreSQL/live OI와 frozen 연구 OI의 동등성

따라서 해당 parity artifact만으로 현재 gross 8 포트폴리오 전체의 live/backtest parity를 주장할 수 없다.

## 7. Backtest 도구의 별도 결함

요청 구간의 끝이 portfolio promotion 시점보다 앞설 때도 `post_promotion` metric window를 무조건 계산하여 `ValueError("empty metric window")`가 발생한다.

- 관련 위치: `training/backtest_added_alpha_month.py:1001-1022`

본 감사에서는 임시 config에서 promotion `as_of`만 요청 구간 안으로 이동시켜 동일 DB 재생을 수행했다. 이 결함은 live signal 0의 원인은 아니지만 historical replay의 재현성과 자동 감사를 방해한다.

## 8. 우선 수정 권고

1. **Rank7 OI 입력 계약 고정**
   - research/live에서 동일한 timestamp, aggregation, forward-fill 및 precedence 규칙을 사용한다.
   - historical OI와 live snapshot의 전환 경계를 명시적으로 고정한다.

2. **슬리브별 schedule parity fixture 추가**
   - aggregate trade count가 아니라 timestamp, direction, strength, expiry와 source hash를 비교한다.
   - Rank7 expected raw 16개 및 accepted 12개를 regression fixture로 보존한다.

3. **REX taker frozen 입력 복구 또는 최소 fixture 보존**
   - 누락된 event JSONL과 market source를 복구한다.
   - 원본 복구가 불가능하면 expected schedule과 핵심 feature row를 저장해 이후 drift를 차단한다.

4. **Tail cache를 full compute와 동일하게 수정**
   - calendar boundary 이전까지 context를 정렬하거나 stateful HTF aggregation 결과를 캐시한다.
   - 최소 5주 이상의 데이터로 최신 행 full/tail equality regression을 추가한다.

5. **Backtest source provenance를 fail-closed로 검증**
   - source file/hash/schema/timezone가 manifest와 다르면 백테스트를 실패시킨다.
   - 결과에 슬리브별 schedule hash와 source hash를 기록한다.

6. **Historical window 도구 오류 수정**
   - promotion 이전 구간에서는 post-promotion metric을 생략하거나 빈 결과로 명시한다.

## 9. 최종 판정

| Claim | Verdict | Confidence |
|---|---|---|
| 현재 라이브 프로세스가 멈춰 있어 거래가 없다 | 기각 | 높음 |
| 현재 코드와 현재 DB로 최근 신호 0이 재현된다 | 확인 | 높음 |
| 현재 live signal이 frozen 연구 백테스트와 완전히 같다 | 기각 | 높음 |
| Rank7 activation schedule이 frozen 연구와 같다 | 기각 | 높음 |
| REX taker가 frozen 연구와 같다 | 확인 불가, 결과는 불일치 | 중간~높음 |
| tail cache와 full compute가 항상 같다 | 기각 | 높음 |
| 최근 약 39일 무거래가 정상이다 | 판정 보류 | 높음 |

안전한 운영 판단은 **"현재 live loop는 동작하지만 연구 parity는 깨져 있으며, Rank7 OI와 REX taker source를 복구·고정하기 전까지 장기 무거래를 전략의 정상 동작으로 인증할 수 없다"**이다.
