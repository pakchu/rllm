# POWR-12 strict evaluator 사전 동결

## 판정

**평가 코드와 모든 primary/control 클록을 outcome 전에 봉인했다.**

- evaluator commit: `cea52ae8c08c323fc151462ab677abb96e5abd07`
- evaluator SHA256:
  `d5733f28bd3c18d8bea821d2b9dfb1a34842c4ae6124c10d2e389d04058c173b`
- freeze artifact:
  `results/perp_only_wick_rejection_evaluator_freeze_2026-07-17.json`
- freeze artifact SHA256:
  `e2c74691c88c82d11550c40d030da32decf50a38e75343652e01fb222ee3fac2`
- freeze manifest hash:
  `435f45c0b6e8846150da3bd5f2ab18eecb1e2f0d5c204bcd5a2a2c5e786dec15`

Freeze 단계에서는 signal 생성에 이미 사용된 completed-bar OHLC로 클록만
재생했다. post-signal return을 계산하지 않았고 funding을 읽지 않았으며,
execution simulation도 실행하지 않았다. Public market/funding/simulation 경로는
이 freeze 검증에 성공하지 않으면 실행을 거부한다.

## 동결된 클록

| 정책 | 행 수 | SHA256 |
|---|---:|---|
| primary | 637 | `7ecd567bf182fd7f92a8a1583b8f82c409ea5530d2e0eef25174880d52502619` |
| direction flip | 637 | `917ebc28c8070c2fff3d9bf7e81694cd430931cff5bb4dce9ba48d9fc770a82e` |
| Spot-only wick | 8,609 | `4c538fbddae4b38d4a344c39feee784b5c0aba6655195a3e373124a7af548366` |
| common wick | 8,462 | `6325e21d3a632800c4c6cf0e9546740bd1e7b185f032319f6b22db3eba92b6bb` |
| basis-free Perp wick | 8,476 | `0df1ebfec774ac9aaffadbed0e26bb24e9d2f16b5775557a2f5f8f8660fb8b4b` |
| 진입 1봉 추가 지연 | 637 | `500dd066bdac7a6711fb801f16b16e0e17794bf8e8e82ca56f4789c2ff5ec05b` |
| stale Spot 1h | 9,124 | `411d7f3d5e07f0148a38b24ee519074694af63161c67f946eb24ea784b80a204` |
| stale Spot 1d | 8,810 | `7e5d48c1245208d000f2a000b45fe27c959d74d4459070e3ce98413fc0ac76ce` |

## 동결된 회계

- 0.5x
- 기본 6bp/notional/side, stress 8bp/notional/side
- 진입 `signal+3`, 지연 대조군 `signal+4`, 12개 5분 봉 보유
- realized funding rate, 반개구간 `[entry, exit)` 및 fixed-entry-notional 적용
- global/pre-entry HWM
- held high/low의 favorable-before-adverse 최악 순서
- entry, scheduled exit 및 adverse 극값의 가상 청산 비용
- 미거래 현금 구간을 포함한 split 전체 wall-clock CAGR
- 100,000회 weekly-cluster sign-flip, seed `20260717`

Funding 원본의 historical mark-price 열은 대부분 비어 있으므로 이를 미래 가격으로
보간하지 않는다. 사전등록된 realized rate를 fixed entry notional에 적용한다.

## 승격 경계

이번 one-shot은 2020–2022 train과 2023 selection/H1/H2만 연다. 성능을
통과해도 `selected_alpha`는 설정하지 않으며, 사전등록한 entry/position/PnL
직교성 및 marginal portfolio 개선을 별도 동결 단계에서 통과해야 최종 승격할
수 있다. 2024·2025·2026 YTD는 계속 봉인한다.
