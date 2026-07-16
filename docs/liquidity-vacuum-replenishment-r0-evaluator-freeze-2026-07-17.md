# LVRT-R0 strict evaluator 동결

## 상태

`training/evaluate_liquidity_vacuum_replenishment.py`를 실제 가격 outcome을 열기 전에 커밋·해시 동결했다.

- evaluator commit: `d784101492828ae5e2a1dd7aa39674d47b53acb1`
- evaluator SHA256: `1d8d36e52eff79888c77da63428f19f55a24fbf48bf09d90feeedcb23501299c`
- freeze manifest SHA256: `9ea54fbe9d971b4943d7ac76da16fb0fa9fe870de039fe8cffae19edbded0102`
- `outcomes_opened=false`
- 열린 구간: 없음
- mutable parameter: 없음
- 가격/수익률 로드: 없음

freeze 스크립트는 evaluator 파일이 Git HEAD에 tracked·clean 상태이고, HEAD에서 꺼낸 바이트의 SHA256이 작업 트리 파일과 같은지 확인한 뒤에만 manifest를 생성했다.

## 동결된 실행 회계

- setup `t0`, confirmation `t1` 완료 후 다음 5분 시가 진입
- 12개 5분 봉 보유, 예정 시가 청산
- 0.5x gross
- 기본 비용 6bp/notional/side
- 스트레스 비용 8bp/notional/side
- 실제 funding 구간: `entry_time <= funding_time < exit_time`
- exit/re-entry가 같은 시각이면 funding settlement를 한 거래에만 배정
- CAGR: 미거래 현금 기간까지 포함한 전체 split 달력
- strict MDD:
  - global/pre-entry HWM 갱신
  - 진입 비용
  - 보유 봉의 유리한 극값이 먼저 HWM을 만든 뒤 불리한 극값 방문
  - funding credit은 HWM을 높이고, debit/credit 전체는 adverse equity에 반영
  - adverse 시점 가상 청산 비용
  - 실제 exit 비용
  - exit bar의 high/low는 보유 경로에서 제외

## 클록 변조 방지

지원도 단계에서는 primary 클록만 CSV로 저장했지만, evaluator freeze에서는 대조군까지 전부 재생하여 행 수와 SHA256을 동결했다.

| 정책 | 행 수 | SHA256 |
|---|---:|---|
| primary | 1,259 | `ed9dd6391df2118ac09d147a4e57c3cb3f6e105a13f6c0d973ee424cfedd54d2` |
| direction flip | 1,259 | `5293da15b12934d6b1724818ffe47178b0f0eb13de28bd51ddb1cb070bb32783` |
| reversal 확인 제거 | 2,756 | `aa9243be111675548f01b25775c96adae58c27f341bcfe4ac2d064294a882bdd` |
| 진입 1봉 추가 지연 | 1,259 | `9f3d259b9ac7bddfe285ec08474db5784e02cfb5d4ddf51985fd752f674ede94` |
| setup 1일 이동 | 1,235 | `7a49f6db1f4cdb202316a3b16022bca2d6fb2252e8b8cccf51d62bbe78fb3c1e` |
| confirmation 부호 순열 | 899 | `835968a030b83b85a175362d980fcb489fa3abf45417579ac700ed7203d2c3de` |

평가기는 가격을 읽기 전에 freeze manifest의 canonical hash, outcome-free 플래그, evaluator SHA, 모든 클록의 행 수·SHA를 재검증한다.

## 다음 한 번의 평가

다음 실행에서 처음으로 2020–2022 train과 2023 selection/H1/H2 OHLC 및 실현 funding을 연다. 2024·2025·2026 YTD는 계속 봉인한다. 사전 게이트를 하나라도 실패하거나 하루 이동/부호 순열 placebo가 전체 게이트를 통과하면 LVRT-R0를 수정하지 않고 기각한다.
