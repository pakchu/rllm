# COIN-M calendar-curve compression selection result — 2026-07-19

## 판정

**기각.** 결과 확인 전 봉인한 단일 12시간 후보는 pre-2024 fit과 완전 OOS
2023에서 모두 비용 차감 후 음수였다. 규칙이나 게이트를 결과에 맞춰 수리하지
않으며 2024 이후 구간은 열지 않는다.

| 구간 | 절대수익 | full-calendar CAGR | strict MDD | CAGR/MDD | 거래 | 평균 gross curve compression |
|---|---:|---:|---:|---:|---:|---:|
| fit 2020-07-15~2022 | -9.56% | -3.99% | 10.62% | -0.38 | 179 | 1.57bp |
| 2023 OOS | -3.17% | -3.17% | 4.01% | -0.79 | 60 | 2.55bp |
| 2023 H1 | -1.69% | -3.38% | 2.57% | -1.31 | 30 | 1.30bp |
| 2023 H2 | -1.50% | -2.96% | 2.70% | -1.10 | 30 | 3.81bp |

strict MDD는 진입 전/global HWM, 진입 비용, 보유 중 각 5분봉의 두 leg 독립
유리 극값 이후 독립 불리 극값, 가상 청산 비용과 실제 청산 비용을 모두 포함한다.

## 대조군과 원인

- 10bp/leg/side 비용 stress: fit -15.81%, 2023 -5.46%.
- 같은 clock 방향 반전: fit -10.82%, 2023 -3.91%.
- 1시간 지연: fit -9.95%, 2023 -3.61%.
- 24시간 지연: fit -10.17%, 2023 -3.52%.
- weekly cluster sign-flip p-value: fit 1.0, 2023 1.0.

원 방향과 반전 방향이 모두 손실이고, 관측된 gross curve compression은 사전등록
경제성 하한 12bp에 크게 못 미친다. 따라서 문제는 방향 선택이나 timing 수리가
아니라 **12시간 horizon에서 곡선 충격의 평균 회귀 크기가 거래 비용보다 작다**는
것이다. 이 계열은 추가 튜닝하지 않는다.

## 무결성

- evaluator freeze hash: `b2c0cde1f1542a4685d8491a2f148c75910eb8eb83642d8cb78e7d78bb78f7bd`
- result hash: `f8db6ba4e79be0a991a3cdfd13c81ea3fe760701651866308a34d5bdcd50684f`
- result artifact SHA-256: `88b355a11c630b8d52f23ef2a83fed6d3413ce102163def0ad6fdf5bd8e96bf6`
- candidates evaluated: 1
- candidates passing: 0
- 2024+ opened: false
