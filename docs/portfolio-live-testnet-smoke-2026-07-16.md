# Portfolio live Binance Futures testnet smoke (2026-07-16)

## 범위와 안전 경계

- Binance USD-M Futures **testnet만** 사용했다. Mainnet endpoint와 key는 사용하지 않았다.
- 테스트 시작 전 BTCUSDT 포지션과 미체결 주문이 모두 0인지 확인했다.
- 계정이 One-way mode였으므로 포지션/주문이 없는 상태에서 Hedge mode로 전환했다.
- 체결 가능성을 낮추기 위해 현재가에서 ±2% 떨어진 `GTX` post-only 주문을 사용했다.
- 테스트 종료 시 포지션과 미체결 주문이 모두 0인지 다시 확인했다.
- portfolio cycle에서 사용한 BTCUSDT leverage 6은 테스트 종료 후 config 기본값 1로 복원했다. Hedge mode는 포트폴리오 실행 필수 조건이므로 유지했다.

## 결과

### 1. 단일 주문 lifecycle

| 단계 | 결과 |
|---|---|
| SELL/SHORT 0.001 BTC post-only 제출 | `NEW` |
| 주문 조회 | `NEW` |
| 취소 | `CANCELED` |
| 최종 체결량 | `0.0000` |
| 잔여 주문/포지션 | 0 / 0 |

### 2. 실제 portfolio runner 한 cycle

실행 대상은 `portfolio_gross385_trainmdd40_2026-07-12.json`이며 testnet execution config와 고유 strategy/state 경로를 사용했다.

| 항목 | 결과 |
|---|---:|
| completed 5m bar | 2026-07-16 07:05 UTC |
| 공통 frame build | 5.96초 |
| 3-alpha process score | 0.77초 |
| 서로 다른 worker PID | 3개 |
| source freshness missing | 0 |
| active alpha | 0 |
| 신규/청산 주문 | 0 / 0 |
| 자동 network scope | `binance-testnet` |

DB reservation PK는 `(strategy_name, sub_strategy_name, signal_id, action, exchange, symbol)`로 생성된 것을 확인했다. 해당 cycle에는 active signal이 없어 reservation/execution row도 생성되지 않았다.

### 3. 동시 주문 I/O

LONG/BUY와 SHORT/SELL post-only 주문을 `asyncio.gather`로 동시에 제출했다.

| 항목 | 결과 |
|---|---|
| 동시 제출 수 | 2 |
| 제출 상태 | `NEW`, `NEW` |
| 조회 상태 | `NEW`, `NEW` |
| 취소 상태 | `CANCELED`, `CANCELED` |
| 최종 체결량 | `0.0000`, `0.0000` |
| cleanup market order | 0 |
| 종료 후 미체결/포지션 | 0 / 0 |

## 테스트 중 발견해 수정한 문제

- CLI help의 `%` 문자열이 argparse formatting error를 내던 문제를 수정했다.
- testnet config가 기본 `exchange=binance`를 사용해도 ledger/lease scope가 자동으로 `binance-testnet`이 되도록 분리했다.
- mainnet config와 testnet exchange scope를 섞으면 fail-closed 한다.
- executor 초기화가 Hedge mode 검사 등에서 실패할 때 exchange client를 닫도록 수정했다.

관련 커밋: `e278bec fix: harden testnet runtime identity and startup`

## 남은 검증 범위

이번 시점에는 세 active alpha가 모두 비활성이어서 **실제 alpha signal에 의한 체결→보유→청산**은 발생하지 않았다. 프로세스 병렬 계산, source freshness, testnet 연결, 단일/동시 주문 제출·조회·취소, 최종 계정 정리는 검증됐다. 실제 fill lifecycle은 다음 자연 발생 testnet signal이나 별도 제한된 synthetic sleeve canary에서 검증해야 한다.
