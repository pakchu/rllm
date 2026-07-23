# G8 라이브 전략 의미와 무거래 점검

점검 기준 시각: **2026-07-21 09:00 UTC**  
라이브 승격 시각: **2026-07-18 09:00 UTC**  
판정: **현재까지 거래가 없는 것은 전략 신호 기준으로 정상이며, 점검 중 발견된 집행 준비 결함 두 가지는 수정 및 라이브 재검증을 마쳤다.**

## 1. 현재 라이브 설정

- 포트폴리오: `configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json`
- 집행 설정: `configs/live/rex_rule_binance_mainnet_gross8.local.json`
- 네트워크: Binance USD-M mainnet
- 주문 모드: 실거래 (`dry_run=false`, `allow_live_orders=true`)
- 레버리지 예산: 8배
- 배분 방식: `research_gross`
- 총 연구 웨이트: 8.0
- LLM selector: 사용하지 않음 (`selector=none`)
- 동일 BTC의 롱·숏 및 알파 중첩 회계: `same_btc_low_high_v1`

| 알파 | 웨이트 | 최대 증거금 비율 | 방향 |
|---|---:|---:|---|
| `fresh_kimchi_fx` | 2.0 | 25% | 롱/숏 |
| `frozen_annual_rank7` | 2.0 | 25% | 롱 |
| `rex_taker_low_range_position` | 0.4 | 5% | 롱/숏 |
| `cand_rex_veto_7` | 1.6 | 20% | 롱/숏 |
| `markov_transition_long` | 2.0 | 25% | 롱 |

모든 알파가 동시에 켜질 때만 gross 8.0이 된다. 일부만 켜지면 해당 웨이트만 사용하므로 항상 8배 포지션을 유지하는 전략은 아니다.

## 2. 알파별 정성적 의미

### 2.1 `fresh_kimchi_fx`

국내외 가격·환율 괴리와 선물 수급이 극단적으로 어긋났다가 정상화될 가능성을 거래한다.

**롱 조건**

- 펀딩이 충분히 음수여서 숏 포지션이 붐빈 상태
- 최근 약 1시간의 매수 체결 흐름이 약 4시간 기준보다 빠르게 개선
- 김치 프리미엄 변화가 USD/KRW 변화보다 비정상적으로 약한 상태
- USD/KRW와 김치 데이터가 모두 유효

즉, 상위 시장 구조가 무너지지 않은 상태에서 국내 가격 괴리와 숏 쏠림이 되돌아오는 구간을 노린다.

**숏 조건**

- USD/KRW가 강하게 상승하고
- 직전 완성 일봉 BTC 수익률이 약 -3.4% 이하

즉, 원화 약세와 BTC 급락이 동시에 나타나는 명확한 위험회피 국면만 숏으로 거래한다.

**수명과 청산**

- 30분 간격으로 평가
- 다음 5분봉에 진입
- 최대 288개 5분봉, 약 24시간 보유
- 실제 체결가 기준 약 +4% 익절 / -2.5% 손절

### 2.2 `frozen_annual_rank7`

먼저 규칙으로 상승 가능성이 있는 불균형을 찾고, 동결된 ExtraTrees 앙상블이 예상 순수익과 불리한 변동을 평가해 질이 좋은 후보만 남긴다.

**기본 후보**

1. 음수 펀딩 + 양의 추세: 가격은 오르는데 숏이 붐빈 구간
2. 프리미엄 급락 + 강한 4일 상승: 상승 추세 중 선물 베이시스가 일시적으로 눌린 구간

**추가 필터**

- Kalman, BOCPD, semi-Markov 상태가 모두 유효해야 함
- 5개 ExtraTrees 모델이 예상 순수익 대비 예상 불리 변동이 충분히 좋다고 판단해야 함
- 펀딩 후보는 변동 범위나 일봉 내 눌림 위치 조건도 통과해야 함

시간 경계는 매시 정각이며, 입력 피처는 12개 5분봉만큼 지연해 미래 정보를 차단한다. 펀딩 원천 후보는 최대 약 48시간, 프리미엄 원천 후보는 최대 약 12시간 보유한다.

### 2.3 `rex_taker_low_range_position`

상위 시간대 추세와 반대 방향으로 단기 눌림이 나온 뒤 다시 원래 추세 방향으로 회복되는 구간을 거래한다.

- 상위 추세가 상승이면 눌림 후 롱, 하락이면 반등 후 숏
- 국소 추세 회복, 거래량, taker 흐름으로 후보 강도를 계산
- taker imbalance가 매도 우위여야 함
- 7일 범위 위치가 지나치게 높은 곳이 아니어야 함

롱에서는 상승 추세 안의 매도 압력성 눌림을 사는 의미이고, 숏에서는 하락 추세 안의 매도 압력 지속을 거래하는 의미다. 2시간마다 평가하고 최대 약 12시간 보유한다.

### 2.4 `cand_rex_veto_7`

REX 눌림·회복 후보를 더 낮은 강도 문턱에서 넓게 만든 뒤, 극단적 붕괴와 레버리지 과열을 제외한다.

- 4주 수익률이 약 -26.6%보다 나빠서는 안 됨
- OI z-score가 약 1.59 이하여야 함
- REX 기본 후보 자체가 롱 또는 숏으로 활성화되어야 함

따라서 단순히 신호 수를 늘리는 알파가 아니라, REX 후보 중 시장 붕괴 추격과 OI 과밀 진입을 피하는 위험-veto 버전이다. 현재 라이브에서는 로컬 LLM이 아니라 규칙과 수치 게이트만 사용한다.

### 2.5 `markov_transition_long`

Rank7과 비슷한 펀딩/프리미엄 불균형 롱 후보를 사용하지만, ML 수익 예측 대신 관측 가능한 시장 상태가 한 시간 이상 지속되는지를 확인한다.

**기본 후보 중 하나가 필요하다.**

1. 음수 펀딩 + 양의 8시간 추세
2. 프리미엄 급락 + 강한 4일 상승

그 후 추세·변동성·체결 흐름으로 만든 12개 상태 중 허용된 자기 전이만 통과한다. 즉, 한 시간짜리 잡음이 아니라 같은 시장 체제가 지속될 때만 롱으로 진입한다. 매시간 평가하고 최대 약 48시간 보유한다.

## 3. 포트폴리오의 전체 성격

- 주된 공통 베팅은 **상승 구조 안의 펀딩/프리미엄 불균형 또는 눌림 회복**이다.
- `frozen_annual_rank7`과 `markov_transition_long`은 기본 후보 가설을 공유하지만 각각 ML 품질 필터와 상태 지속성 필터로 분리된다.
- 두 REX 알파도 같은 눌림·회복 계열이지만 서로 다른 위험 게이트를 사용한다.
- 숏은 `fresh_kimchi_fx`의 명확한 위험회피 조건이나 REX의 상위 하락 추세에서만 나오므로 롱보다 드물다.
- 따라서 알파 이름은 다섯 개지만 완전히 독립적인 다섯 전략은 아니며, 상승 불균형 계열에 위험이 집중될 수 있다.

## 4. 라이브 무거래 확인

G8이 처음 로그에 나타난 `2026-07-18 09:00:30 UTC`부터 `2026-07-21 09:00:12 UTC`까지 확인했다.

| 항목 | 결과 |
|---|---:|
| 완료된 라이브 사이클 | 865 |
| `active`가 비어 있지 않은 사이클 | 0 |
| `opened`가 비어 있지 않은 사이클 | 0 |
| `closed`가 비어 있지 않은 사이클 | 0 |
| 열린 sleeve가 있던 사이클 | 0 |
| `dry_run=false` 확인 | 865 / 865 |
| 필수 ingest 누락(`miss`) | 0 / 865 |
| `trade_executions` 행 | 0 |
| `trade_execution_reservations` 행 | 0 |

즉, 주문이 발생했는데 DB에만 빠진 것이 아니라 **주문 이전의 활성 시그널 자체가 없었다.**

## 5. 동일 기간 라이브형 백테스트 재생

연구용 고정 이벤트 파일이 아니라 현재 PostgreSQL 데이터와 현재 라이브 스코어러를 사용했다.

- 결정봉 범위: `2026-07-18 08:55 UTC` ~ `2026-07-21 08:55 UTC`
- G8 첫 09:00 실행이 판단한 직전 완성봉부터 포함
- 원본 프레임: 18,900개 5분봉
- 각 시점은 미래 행을 자르고 최근 18,000봉만 사용
- 현재 포트폴리오·게이트·stride·동결 Rank7 bundle을 그대로 사용
- Rank7은 동일한 인과적 전체 피처 그래프를 한 번 만들고 각 정각 행만 평가
- 주문, 체결, 포지션 상태는 후보가 0건이므로 결과에 영향을 주지 않음

| 알파 | 평가 예정 시각 수 | 활성 후보 | 예상 신규 거래 |
|---|---:|---:|---:|
| `fresh_kimchi_fx` | 145 | 0 | 0 |
| `frozen_annual_rank7` | 72 | 0 | 0 |
| `rex_taker_low_range_position` | 37 | 0 | 0 |
| `cand_rex_veto_7` | 37 | 0 | 0 |
| `markov_transition_long` | 73 | 0 | 0 |
| **합계** | **364** | **0** | **0** |

### 주요 비활성 원인

- `fresh_kimchi_fx`
  - 145회 모두 음수 펀딩 롱 조건을 통과하지 못함
  - 김치/환율 충격 조건은 2회, 매수 흐름 가속은 57회 따로 통과했지만 동시에 성립하지 않음
  - 숏의 USD/KRW 모멘텀과 일봉 -3.4% 조건도 각각 0회
- `frozen_annual_rank7`
  - 72개 정각 모두 immutable base anchor가 없었고 funding/premium 원천 후보도 0회
- `rex_taker_low_range_position`
  - taker 조건은 20/37회, 범위 위치 조건은 29/37회 통과
  - 그러나 REX 기본 눌림·회복 후보가 37회 모두 비활성
- `cand_rex_veto_7`
  - 4주 붕괴 방지 조건은 37/37회, OI 과열 방지 조건은 26/37회 통과
  - 그러나 REX 기본 후보가 37회 모두 비활성
- `markov_transition_long`
  - 허용 Markov 상태 지속은 60/73회였음
  - 하지만 음수 펀딩/상승 추세 또는 프리미엄 급락/4일 강세 기본 후보가 0회

과거 연구 결과의 2026 YTD 거래 수는 108회로 대략 하루 0.7회 수준이었다. 독립 Poisson으로 단순 근사하면 3일 무거래도 약 12% 확률이므로 드문 편이지만 비정상이라고 볼 정도는 아니다. 실제 신호는 군집되므로 이 수치는 참고용이다.

## 6. 운영상 발견된 문제와 해결

두 문제 모두 이번 무거래의 원인은 아니었지만 다음 유효 신호를 막을 수 있어 수정했다.

### 6.1 Binance signed request 시간 오차

현재 state에는 포지션 reconciliation과 stale-order scan에서 아래 오류가 남아 있다.

```text
API Error 400: code=-1021, Timestamp for this request is outside of the recvWindow
```

점검 시 호스트와 Binance 공개 서버 시간 차이는 약 0.02초뿐이었다. 따라서 시스템 시계 자체보다는 프로세스 시작 때 계산한 client time offset이 장시간 유지된 것이 원인이었다.

`wave_trading/trading/binance_client.py`의 signed request가 Binance `-1021`을 받으면 공개 서버 시간을 다시 동기화하고, 새 timestamp와 signature로 정확히 한 번 재시도하도록 수정했다. 다른 API 오류는 재시도하지 않고 그대로 전파한다.

### 6.2 aggTrade 큐가 포지션이 없을 때 소비되지 않음

현재 barrier 보유 알파에는 다음 상태가 붙어 있다.

```text
aggtrade_stream=unhealthy:fail_closed
aggtrade_stream_pre_entry=unhealthy:fail_closed
```

공개 WebSocket endpoint 자체는 정상적으로 aggTrade를 전달했다. 코드상 원인은 열린 barrier 포지션이 없을 때 `_wait_with_barrier_monitor()`가 stream queue를 비우지 않는 구조다. 최대 200,000개 큐가 차면 `overflowed=true`가 되어 스트림이 영구적으로 unhealthy가 되고, barrier가 있는 `fresh_kimchi_fx`와 `frozen_annual_rank7` 신규 진입을 fail-closed한다.

이번 기간에는 두 알파의 기본 후보가 백테스트에서도 0건이어서 실제 누락을 만들지는 않았다. `_wait_with_barrier_monitor()`가 열린 barrier 포지션이 없어도 매 poll마다 stream queue를 drain하도록 수정했다. 따라서 다음 포지션이 열릴 때까지 과거 tick이 쌓이지 않는다.

### 6.3 오해하기 쉬운 진단 문자열

`fresh_kimchi_fx`가 비활성이고 방향이 아직 `AUTO`일 때 `barrier_contract=side_invalid:fail_closed`가 추가된다. 이것은 이번 무거래의 원인이 아니라, 앞선 방향성 XOR 조건이 실패한 뒤 barrier 계약 검증이 후행하면서 생기는 부가 진단이다.

## 7. 최종 판정

1. **전략 신호 관점:** 정상. 라이브와 동일 기간 재생이 모두 0건으로 일치한다.
2. **데이터 관점:** 정상. 865개 라이브 사이클 모두 필수 소스 누락이 0이었다.
3. **주문/DB 관점:** 정상적인 0건. 활성 신호, 예약, 주문 실행 레코드가 모두 0이다.
4. **향후 집행 준비 관점:** 수정 후 정상. Binance 시간 자동 재동기화와 idle aggTrade drain을 적용하고 라이브 프로세스를 재시작했다.

## 8. 수정 후 검증

- RLLM 관련 테스트: **92 passed, 5 subtests passed**
- wave-trading 전체 테스트: **6 passed**
- mainnet 읽기 전용 사전 확인:
  - Binance time offset: `-20ms`
  - 열린 포지션: 0
  - 열린 주문: 0
- 라이브 재시작 후 첫 사이클 `2026-07-21 09:30:31 UTC`:
  - `dry_run=false`
  - 필수 데이터 누락 0
  - 완성 의사결정봉 일치
  - stale-order scan 오류 없음
  - position recovery/reconcile 오류 없음
  - 5개 알파 모두 aggTrade unhealthy 사유 없음
  - 열린 포지션 및 주문 없음

## 9. 근거 파일

- `configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json`
- `configs/live/rex_rule_binance_mainnet_gross8.local.json`
- `configs/shadow/fresh_kimchi_fx_2026-07-16.json`
- `configs/shadow/frozen_annual_rank7_2026-07-16.json`
- `configs/shadow/rex_taker_low_range_position_2026-07-16.json`
- `configs/live/rex_veto_7_candidate.json`
- `configs/shadow/markov_transition_long_2026-07-16.json`
- `execution/portfolio_live.py`
- `execution/rank7_runtime.py`
- `execution/binance_aggtrade_stream.py`
- `logs/portfolio_live/rllm.log`
- `.omx/state/portfolio_live_state.json`
