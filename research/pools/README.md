# Research Pools

누적 알파 리서치를 `alpha_feature_pool`, `beta_feature_pool`, `gamma_feature_pool`, `alpha_pool`, `portfolio_pool`로 관리한다. `feature_pool.json`은 세 피쳐 tier를 합친 호환용 통합 registry다.

## 목적

- 새 탐색이 기존 실패를 반복하지 않도록 `rejected/weak/candidate/promoted/live` 상태를 기록한다.
- train/test/eval/2026 오염을 명시한다.
- feature와 alpha, portfolio의 의존 관계를 추적한다.
- 결과 JSON/문서/스크립트 경로를 한 곳에서 찾는다.
- 피쳐 생성법(`generation_recipe`), 알파 사용법(`usage_recipe`), 포트폴리오 구성법(`construction_recipe`)을 복원 가능하게 남긴다.

## 파일

| file | 역할 |
|---|---|
| `feature_pool.json` | 알파/베타/감마 피쳐를 합친 통합 registry. 각 entry에 `feature_tier`가 있다. |
| `alpha_feature_pool.json` | 전략 edge의 핵심 원천이 된 피쳐 family. alpha/live sleeve의 직접 dependency. |
| `beta_feature_pool.json` | selector/gate/context/risk-state 피쳐. 단독 alpha가 아니라 기존 setup과 결합해 사용. |
| `gamma_feature_pool.json` | inventory/search set/실패/프로비넌스. 반복 실패 방지와 복원 목적. |
| `feature_inventory.json` | 알파 탐색 스크립트에서 추출한 전체 feature/source inventory |
| `history_inventory.json` / `history_inventory.md` | OMX history, docs, results, training scripts, live configs, research pools, git log를 전수 스캔한 알파/피쳐 히스토리 인덱스 |
| `alpha_pool.json` | 단독/게이트/동적 exit alpha 후보와 실패 기록 |
| `portfolio_pool.json` | sleeve 조합/라이브 후보/최적화 조건과 결과 |
| `schema.md` | 각 pool entry의 필드 규칙 |


## 피쳐 tier 정의

- `alpha_feature`: 전략 edge의 원천. alpha-pool/live sleeve의 직접 dependency이며 단순 흥미 피쳐가 아니다.
- `beta_feature`: **알파피쳐가 될 가능성이 남아 있는 pool**. selector/gate/context/risk-state로 실험한다. 아직 alpha가 아니다.
- `gamma_feature`: **노이즈/무효/반복 실패가 강하게 입증된 사용법 기록 pool**. 비슷한 재시도를 막기 위해 실패 원인, known failure, 재검토 조건을 남긴다. 기본 탐색 universe에서 제외한다.

승격/강등 흐름:

```text
gamma_feature --새 데이터/방법으로 실패 원인 제거--> beta_feature
beta_feature --전략 기준 통과--> alpha_feature
alpha_feature --전략 evidence 붕괴--> beta_feature 또는 gamma_feature
```

엄격 기준:

- beta -> alpha_feature: 해당 feature를 사용한 전략이 `test2024 CAGR/strict MDD >= 2.5`, 기본 `test2024 >= 20 trades`, `eval2025 >= 15 trades`, eval 양수, strict MDD/forced close/6bp 비용 적용.
- live-grade standalone: 단일 전략 `CAGR/strict MDD >= 3.0`.
- beta -> gamma_feature: 단순 미달이 아니라 “노이즈뿐”이라는 강한 증거가 필요하다. 예: broad+targeted 모두 robust positive 구조 없음, selector/portfolio 기여 없음, leakage/no-op/causal invalidation, 반복 regime inversion/OOS collapse, 또는 특정 standalone/additive 사용법이 noise로 판정됨.
- gamma -> beta: 새 데이터, 누수 수정, materially different usage mode 등으로 기존 실패 이유가 제거되어야 함.

단순히 live급이 아니거나 수치가 약하다는 이유만으로 gamma로 내리지 않는다. broad inventory/search map도 gamma가 아니다. 가능성이 남으면 beta에 둔다.

## 상태값

- `live`: 실제 운용/라이브 브릿지 기준 후보.
- `promoted`: live 가능으로 승격했으나 운용 여부는 별도 확인 필요.
- `candidate`: 추가 검증 가치가 있는 후보.
- `weak`: 신호는 보이나 목표 미달/분산 후보 수준.
- `rejected`: 반복 금지. 오염/누수/붕괴/노이즈.
- `archived`: 현재 방향과 다르거나 외부 asset 등으로 제외했지만 기록 보존.

## 기본 평가 프로토콜

- split: `train < 2024`, `test = 2024`, `eval = 2025`, `ytd2026 = 2026-01-01..data_end`.
- CAGR은 트레이드 없는 구간도 실제 사용 기간으로 계산한다.
- strict MDD는 포지션 홀딩 중 adverse excursion을 포함한다.
- 평가 구간 종료 시 포지션은 강제 청산/기간 내 exit만 인정한다.
- 기본 비용은 `6bp/side`로 관리한다. 과거 산출물이 `5bp/side`면 `cost_note`에 명시한다.
- `test/eval/2026`을 보고 고른 후보는 contamination note를 남긴다.

## CLI

```bash
python -m training.research_pool_registry summary
python -m training.research_pool_registry list alpha --status candidate
python -m training.research_pool_registry list alpha_feature
python -m training.research_pool_registry list beta_feature
python -m training.research_pool_registry list gamma_feature
python -m training.research_pool_registry list feature --tier beta_feature
python -m training.research_pool_registry show alpha btc_only_vwap_funding_asia
python -m training.research_pool_registry recipe feature btc_only_search_feature_set_20260709
python -m training.research_pool_registry recipe alpha btc_only_vwap_funding_asia
python -m training.research_pool_registry recipe portfolio gross580_dynamic_best
python -m training.build_research_feature_inventory
python -m training.build_research_history_inventory
python -m training.research_pool_registry history-summary
python -m training.research_pool_registry history-search rex --limit 20
python -m training.sync_feature_tier_pools
```


## 복원 절차

1. `alpha_feature_pool.json`, `beta_feature_pool.json`, `gamma_feature_pool.json` 또는 통합 `feature_pool.json`에서 feature id를 고르고 `recipe <tier> <id>`로 생성 원천/명령/출력을 확인한다.
2. `alpha_pool.json`에서 alpha id를 고르고 `recipe alpha <id>`로 사용 피쳐, entry/exit, 평가 프로토콜, 재현 명령을 확인한다.
3. `portfolio_pool.json`에서 portfolio id를 고르고 `recipe portfolio <id>`로 component alpha, weight, gross leverage, 제약, baseline 비교 대상을 확인한다.
4. 최신 feature 전체 목록은 `python -m training.build_research_feature_inventory`로 재생성하고 `feature_inventory.json`과 대조한다.
5. 전체 알파/피쳐 히스토리는 `python -m training.build_research_history_inventory`로 재생성하고, `python -m training.research_pool_registry history-summary` / `history-search <query>`로 누락 후보를 확인한다. 이 인덱스는 gamma 강등을 자동 수행하지 않으며, gamma는 노이즈-only/무효/반복 실패 증거가 확실할 때만 수동 분류한다.

## 승격 기준

- Alpha pool candidate: 다른 피쳐/gate/exit와 섞인 전략이 `test2024 CAGR/strict MDD >= 2.5`를 넘어야 한다.
- Live-grade standalone: 단일 전략으로 `CAGR/strict MDD >= 3.0`을 넘어야 한다.
- 이 기준 미만은 alpha가 아니라 feature/selector 후보 또는 weak evidence로 둔다.

## Tier 파일 동기화

통합 `feature_pool.json`을 수정한 뒤에는 아래 명령으로 세 tier 파일을 재생성한다.

```bash
python -m training.sync_feature_tier_pools
```
