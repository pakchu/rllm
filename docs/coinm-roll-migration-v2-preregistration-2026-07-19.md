# COIN-M roll-migration v2 preregistration — 2026-07-19

v2는 새 알파 탐색이나 임계치 보정이 아니다. v1 evaluator가 단 한 거래의 보유
경로 결측으로 성과 artifact 생성 전에 중단된 뒤, checksum 검증 월별 archive로
일별 absent key를 보강한 소스에 **동일한 동결 규칙**을 재적용한다.

## 변경된 것

- source SHA: `d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b`
- manifest SHA: `29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af`
- 유효 봉: 366,177 → 368,180
- 월별 archive에서 일별 absent key 3,718개 추가
- 일별/월별 공식 revision 충돌 2행은 일별 값을 유지하고 conflict payload SHA로 기록

## 변경되지 않은 것

- v1 preregistration 코드 SHA `e887248...`
- 후보 2개와 방향
- 60분/30분 hold
- 모든 share/z/price-acceptance 임계치
- liquidity/DTE/delivery buffer
- support gate와 split/non-overlap 규칙
- 2024 이후 봉인

v2 wrapper는 support 단계에서 high/low와 진입 후 가격을 읽지 않는다. v2 support
artifact를 별도 커밋하고 새 evaluator 코드와 freeze artifact를 다시 고정한 뒤에만
성과를 계산한다.
