# Portfolio live 병렬화 계획 및 알파 논리 감사 (2026-07-16)

## 목표와 안전 경계

- DB 커밋 대기는 포트폴리오 주기당 한 번만 수행한다.
- 공통 원천 데이터와 공통 feature snapshot은 한 번만 만든다.
- 알파별 특수 feature 변환과 score 계산은 서로 독립된 프로세스에서 병렬 수행한다.
- 프로세스는 주문 **의도(intent)** 만 반환한다. 거래소 클라이언트, 로컬 state, DB ledger는 중앙 조정기만 소유한다.
- 중앙 조정기는 서로 독립적인 알파 주문을 비동기로 제출하되, `(strategy, sleeve, signal_id, action)`을 DB에서 먼저 원자적으로 예약하여 중복 주문을 막는다.
- 실제 라이브 주문 없이 unit test와 dry-run으로 검증한다.

독립 프로세스가 거래소 주문과 JSON state를 직접 변경하도록 만들면 동일 신호 중복 주문, 같은 hedge-side 수량 귀속 충돌, 프로세스 종료 직후 state 유실이 발생할 수 있다. 따라서 계산은 multi-process, 주문 side effect는 중앙 coordinator라는 actor 경계를 사용한다.

## 구현 단계

1. 현재 직렬 scorer를 단일 알파 pure function으로 분리하고 직렬 결과를 회귀 테스트로 고정한다.
2. sleeve마다 전용 `ProcessPoolExecutor(max_workers=1)`를 두어 장기 실행 프로세스로 격리한다.
3. 한 cycle의 immutable `enriched/features` snapshot을 각 worker에 전달하고 결과 순서를 config 순서로 복원한다.
4. worker 예외·timeout은 해당 sleeve만 `active=false`로 fail-closed 처리한다.
5. 중앙 coordinator가 entry intent를 검증하고 DB에서 원자적으로 예약한 뒤 주문 coroutine을 동시 실행한다.
6. 주문 결과와 state 변경은 parent process가 결정적 순서로 반영한다.
7. live config hash는 변경하지 않고 테스트·dry-run·정적 검사를 수행한다.

## 현재 라이브 알파 감사

### `oi_upbit_ratio288_low`

- **확인된 결함과 수정:** 역사 연구는 `2020-01-01 02:55 UTC`에 해당하는 position 143에서 stride 6을 시작했지만 live config는 offset을 생략하여 epoch offset 0을 사용했다. `stride_offset_bars=5`와 `entry_delay_bars=1`을 명시했다.
- **미래참조:** 발견되지 않았다. 5분 완성봉, backward-asof OI/외부 데이터, 과거 rolling 값만 사용한다.
- **보수적 live 차이:** 연구는 USDKRW availability를 별도 gate로 강제하지 않았지만 live는 `upbit_volume_available`와 `usdkrw_available`을 모두 요구한다. 주말 FX 부재 시 fail-closed 하므로 연구보다 거래가 적어질 수 있다.
- **잔여 위험:** OI/Upbit/USDKRW의 실거래 lifecycle replay와 실제 maker miss 비용 검증이 아직 필요하다.

### `new_long_minimal_funding_premium`

- **확인된 결함과 수정:** 역사 연구의 position 143/stride 12 grid는 epoch offset 11인데 config가 offset 0을 사용했다. `stride_offset_bars=11`을 명시했다.
- **미래참조:** 발견되지 않았다. funding은 backward-asof, premium close는 causal close timestamp를 사용한다.
- **보수적 live 차이:** 연구 premium tolerance는 2시간, live는 10분이다. live가 더 엄격하여 오래된 premium으로 진입하지 않지만 연구 신호와 수가 달라질 수 있다.
- **잔여 위험:** 576 bars 보유 중 funding 지급/수취와 maker/taker 비용이 고정 6bp 연구 가정과 다르므로 lifecycle 비용 replay가 필요하다.

### `cand_rex_veto_7`

- **stride/진입/보유:** `stride=24`, `offset=11`, `entry_delay=1`, `hold=144`가 연구 계약과 맞는다.
- **미래참조:** 완성봉 HTF shift와 backward-asof OI를 사용하며 scorer 수준 미래참조는 발견되지 않았다.
- **신호 parity:** 기존 shadow parity audit에서 범위 내 mismatch 0으로 확인됐다.
- **중요한 연구 한계:** 후보군이 post-train 연구 과정에서 선택된 이력이 있어 pristine OOS 알파로 볼 수 없다. live 가능성과 통계적 일반화는 별개다.
- **잔여 위험:** 동일 side의 여러 sleeve가 Binance hedge position 하나로 합쳐지므로 로컬 state 유실 시 수량 귀속 복구가 모호하다.

## shadow 알파 감사

| 알파 | 현재 판정 | 핵심 이유 |
|---|---|---|
| `fresh_kimchi_fx` | shadow score only | signal parity는 있으나 연구 TP/SL barrier exit가 live runner에 연결되지 않았다. |
| `frozen_annual_rank7` | fail-closed | model bundle, 40-feature graph, causal warm-start와 source exit가 없어 live 재구성이 불가능하다. |
| `rex_taker_low_range_position` | shadow only | signal parity는 있으나 entry/fill/non-overlap/exit lifecycle parity가 미완료다. |
| `markov_transition_long` | shadow only | feature/transition/schedule parity는 있으나 주문 lifecycle replay가 미완료다. |

이 shadow 알파들은 이번 병렬 scorer에서 계산 실패가 전체 cycle을 중단하지 않도록 격리하지만, 위 blocker가 해소되기 전에는 live 주문 허가 대상으로 승격하지 않는다.

## 검증 완료 조건

- 직렬 scorer와 병렬 scorer의 score 결과가 동일하다(프로세스 메타데이터 제외).
- 각 sleeve가 서로 다른 worker PID에서 계산된다.
- 한 worker가 실패/timeout 나도 나머지 sleeve score와 주문 intent는 유지된다.
- 동일 `(sleeve, signal_id)`는 동시 coordinator/재시작에서도 한 번만 예약된다.
- 주문 task 하나의 실패가 다른 sleeve 결과를 취소하지 않는다.
- live anchor config SHA256이 기존 값과 동일하다.
- targeted pytest, 전체 관련 pytest, compile/static check가 통과한다.
