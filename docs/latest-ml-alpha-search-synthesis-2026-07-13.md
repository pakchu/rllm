# Latest ML/TSFM alpha search synthesis (2026-07-13)

## Outcome

사용자가 허용한 **사전 고정 Top-10 중 하나의 일반화** 기준으로 평가했다. 모든
실험은 2023 selection manifest를 먼저 기록하고 2024/2025/2026을 나중에
계산했다. 목표 `CAGR / strict MDD >= 3 / 3 / 5`를 모두 만족한 후보는 없었다.

| Family | Representative candidate | 2024 ratio | 2025 ratio | 2026 ratio | Verdict |
|---|---|---:|---:|---:|---|
| Chronos-2 forecast | fit-oriented long | 2.42 | 0.48 | 1.35 | reject |
| Chronos-2 embedding | PCA32 Group-DRO long | 3.40 | -0.68 | 1.16 | reject |
| MOMENT frozen embedding | PCA32 MLP V-REx | 2.42 | -0.42 | 0.03 | reject |
| MOMENT continual probe | PCA16 slow MLP long | 1.45 | -0.11 | 2.55 | reject |
| MOMENT + Mamba2 | seq32 ERM 48h head | 3.69 | 0.25 | -0.13 | reject |
| Distributional path critic | ERM median utility long | 7.79 | 0.76 | 0.66 | reject |
| Delayed dense retrieval | mean8/k128 median utility long | 2.77 | 3.85 | 0.00 | research candidate only |
| Recency retrieval | mean8/k128 730d both | 3.70 | 0.01 | -1.36 | reject |

## What was learned

1. **Foundation representation에는 약한 정보가 있다.** MOMENT
   `current_mean8/k128` retrieval signed utility의 OOS Spearman은
   `0.0499 / 0.0260 / 0.0493`으로 모두 양수였다.
2. **하지만 경제적 alpha 강도가 부족하다.** 거래 비용과 strict intratrade MDD를
   포함하면 2024/2025/2026 목표를 동시에 넘지 못했다.
3. **Parametric model은 2025 relation flip을 반복했다.** ERM, V-REx,
   Group-DRO, continual replay, Mamba2 모두 같은 구조적 반전을 보였다.
4. **Risk-aware reward는 MDD는 개선하지만 방향 edge를 만들지는 못한다.** Path
   critic은 2024 ratio를 8.08까지 높였지만 2025 ratio는 0.16 이하로 붕괴했다.
5. **Retrieval은 부호 안정성은 개선했지만 밀도와 2026 수익이 부족하다.** Gate
   제거와 1y/2y memory도 문제를 해결하지 못했다.

## Leakage controls shared by the new experiments

- model/data/revision/PCA component hash 고정
- completed-hour input과 next-bar 5m execution
- fit-only normalization, label calibration, initial weights
- 2024 이전 feature/target/inference로 phase1 제한
- actual executable 2023 path hash로 Top-10 중복 제거
- manifest에 2024+ metric key가 없는지 검증
- phase2 완전 재실행 후 2023 path hash 일치 강제
- full-window CAGR, strict intratrade MDD, 편도 6bp

## Decision

현재 9개 입력 축(price/volume/imbalance/range/funding/premium/DXY/kimchi/USD-KRW)
위에서 모델 종류와 reward/gate를 더 바꾸는 것은 false-discovery risk만 키운다.
다음 alpha 연구는 다음과 같은 **독립 정보 축을 먼저 추가**해야 한다.

- futures open interest 변화와 OI-price divergence
- liquidation flow / aggressive order-flow imbalance
- spot-perpetual-futures basis term structure
- options IV/skew/term structure
- cross-asset breadth 및 crypto attention/liquidity regime

새 데이터는 먼저 availability/staleness/발표 지연을 명시하고, 2020-2022
feature admission → 2023 Top-10 freeze → 2024+ sealed evaluation 순서를 그대로
유지한다. 현재 retrieval `current_mean8/k128`은 신규 데이터가 추가될 때 비교할
**research baseline**이지 live alpha가 아니다.

## Official references used

- MOMENT model: <https://huggingface.co/AutonLab/MOMENT-1-small>
- MOMENT code: <https://github.com/moment-timeseries-foundation-model/moment>
- Mamba2 implementation: <https://huggingface.co/docs/transformers/model_doc/mamba2>
- MambaTS reference: <https://github.com/XiudingCai/MambaTS-pytorch>

