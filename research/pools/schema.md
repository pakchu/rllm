# Research Pool Schema

모든 pool JSON은 다음 최상위 구조를 갖는다.

```json
{
  "schema_version": 1,
  "updated_at": "YYYY-MM-DD",
  "protocol": { ... },
  "entries": [ ... ]
}
```

## 공통 entry 필드

| field | required | 설명 |
|---|---:|---|
| `id` | yes | stable snake_case id. 레시피/문서/결과의 primary key로 쓴다. |
| `name` | yes | 사람이 읽는 이름 |
| `status` | yes | `live/promoted/candidate/weak/rejected/archived` |
| `family` | yes | alpha/feature family |
| `scope` | yes | `btc_only`, `btc_derivatives`, `btc_target_external_features`, `btc_target_cross_asset_features`, `portfolio`, `llm_selector`, etc. |
| `description` | yes | 구조 요약 |
| `evidence` | yes | 핵심 수치/판정 |
| `source_artifacts` | yes | 관련 script/result/doc/config 경로 |
| `contamination_risk` | yes | `low/medium/high` |
| `leakage_notes` | yes | train quantile, future close, OOS 선별 등 누수 메모 |
| `next_action` | yes | 다음 작업 또는 반복 금지 지시 |
| `feature_tier` | feature only | `alpha_feature/beta_feature/gamma_feature` |
| `tier_rationale` | feature only | 해당 tier로 분류한 근거 |


## Feature tier schema

`feature_pool.json`은 모든 피쳐를 담는 통합 registry이고, 아래 세 파일은 동일 entry를 tier별로 나눈 materialized view다.

| file | `feature_tier` | 의미 | 사용법 |
|---|---|---|---|
| `alpha_feature_pool.json` | `alpha_feature` | 실제 전략 edge의 원천 또는 alpha/live sleeve의 핵심 dependency | 우선 탐색/조합 대상. 단, 연결 전략 evidence가 바뀌면 재감사한다. |
| `beta_feature_pool.json` | `beta_feature` | 알파피쳐가 될 가능성이 남아 있는 selector/gate/context/risk-state 피쳐 | 기존 setup과 섞어 알파 승격 기준을 넘기는지 탐색한다. 아직 alpha가 아니다. |
| `gamma_feature_pool.json` | `gamma_feature` | 노이즈/무효/반복 실패가 강하게 입증된 사용법 기록. 비슷한 재시도를 막기 위한 pool | 기본 탐색 universe에서 제외한다. 기록된 disqualifying evidence를 뒤집는 새 데이터/방법이 있을 때만 재검토한다. |

### 엄격한 tier 전환 기준

#### beta -> alpha_feature

모두 필요하다.

1. 해당 feature를 사용한 재현 가능한 전략이 `test2024 CAGR/strict MDD >= 2.5`를 넘는다.
2. 비용은 기본 `6bp/side`, strict MDD는 포지션 중 adverse excursion 포함, split-end forced close를 적용한다.
3. 거래 수가 비자명해야 한다. 기본 기준은 `test2024 >= 20`, `eval2025 >= 15`; sparse 저빈도 예외는 명시적으로 표시한다.
4. `eval2025`가 명백히 깨지지 않아야 한다: 양수 CAGR, MDD 폭발 없음.
5. feature contribution이 baseline/no-feature variant 또는 portfolio marginal contribution으로 식별 가능해야 한다.
6. `generation_recipe`와 결과 artifact가 복원 가능해야 한다.

#### beta -> gamma_feature

감마 강등은 매우 보수적으로 한다. “아직 alpha가 아니다”가 아니라 “이 사용법은 노이즈뿐이라는 증거가 강하다”가 필요하다.

다음 중 강한 증거가 있어야 하며, `known_failures`/`next_action`에 반복 금지 조건을 남긴다.

- broad scan + targeted alphaization이 모두 robust positive test/eval 구조를 못 찾고, selector/portfolio 기여도 없다.
- 구조적 무효: leakage invalidation, causal data 부재, no-op threshold, 반복 regime inversion, 반복 OOS collapse.
- 사용자/검증자가 해당 additive/standalone 사용법을 noise로 판정했고, 대체 사용법은 별도 beta entry로 분리되어 있다.

주의:

- live-grade가 아니거나 현재 수치가 약하다는 이유만으로 gamma로 내리지 않는다.
- broad inventory/search map 자체는 noise 증거가 아니므로 gamma가 아니다.
- feature family 전체가 아니라 “실패한 사용법”을 gamma로 기록한다. 다른 사용법 가능성이 남으면 beta entry를 따로 유지한다.

#### gamma_feature -> beta

기록된 실패 원인을 제거하는 새 정보가 있을 때만 허용한다. 예: 새 데이터, 누수 수정, materially different usage mode. 최소한 `test2024 CAGR/strict MDD >= 1.5`와 양수 eval, 또는 강한 portfolio-selector 기여 증거가 필요하다.

#### alpha_feature -> beta/gamma

연결된 전략 evidence가 붕괴하거나 forced-close/strict-MDD/비용 재평가로 기준 미달이 확인되면 beta로 내린다. 누수/구조적 실패가 확인되면 gamma로 내린다.

주의: beta/gamma 피쳐는 alpha_pool 후보가 아니다. alpha_pool candidate는 전략 단위 기준(`test2024 CAGR/strict MDD >= 2.5`)을 만족해야 한다. live-grade standalone은 단일 전략 기준 `CAGR/strict MDD >= 3.0`을 요구한다.

## 복원 가능성 필드

모든 entry는 “나중에 같은 방식으로 다시 만들 수 있음”을 목표로 다음 recipe 중 하나를 가져야 한다.

| pool | recipe field | 목적 |
|---|---|---|
| feature / alpha_feature / beta_feature / gamma_feature | `generation_recipe` | 피쳐가 어떤 원천/스크립트/함수/명령으로 생성됐는지 복원 |
| alpha | `usage_recipe` | 어떤 피쳐를 어떤 entry/exit/평가 프로토콜로 사용했는지 복원 |
| portfolio | `construction_recipe` | 어떤 알파를 어떤 weight/제약/명령으로 조합했는지 복원 |

### `generation_recipe` 권장 필드

- `script`: 피쳐 생성 또는 inventory 생성 스크립트. 여러 개면 list로 기록한다.
- `functions`: 핵심 함수/클래스/계산 위치.
- `inputs`: 원천 데이터, DB 테이블, 캐시 파일, 외부 파일.
- `command`: 재생성 명령.
- `outputs`: 생성 산출물.
- `protocol`: split/target/cost/누수 방지 규칙.
- `notes`: 계산식, quantile fit 구간, known caveat.

### `usage_recipe` 권장 필드

- `feature_dependencies`: feature pool id 또는 구체 feature 목록.
- `entry_logic`: entry 조건. quantile이면 fit 구간과 방향을 명시한다.
- `exit_logic`: fixed hold/dynamic exit/TP/SL/period-end forced close.
- `side`: `long/short/both`.
- `script_or_config`: 평가/탐색 스크립트 또는 live config.
- `reproduce_command`: 동일 후보를 다시 평가하는 명령.
- `source_artifacts`: 결과 문서/JSON/스크립트.
- `evaluation_protocol`: split, cost, strict MDD, forced close, CAGR 계산 방식.
- `promotion_guardrails`: live 승격 전 재검증 조건.
- `live_use_notes`: live 사용/비사용 메모.

### `construction_recipe` 권장 필드

- `component_alphas`: 조합된 alpha ids.
- `weights`: sleeve별 weight. weight 단위/최소 비율을 명시한다.
- `gross`: 총 gross leverage/weight sum.
- `constraints`: MDD, cost, turnover, 동시 포지션, max leverage 등.
- `reproduce_command`: portfolio 조합/재평가 명령.
- `source_artifacts`: 결과 문서/스크립트.
- `evaluation_protocol`: split, cost, strict MDD, forced close.
- `contamination_warning`: test/eval/2026 선별 사용 여부.
- `baseline_required`: 비교해야 하는 기존 live/baseline id.


## Full history inventory schema

`history_inventory.json`은 pool entry가 아니라 전수 감사 인덱스다. 구조:

- `summary`: 스캔된 artifact/history/git hit 카운트.
- `artifacts`: `docs`, `training`, `results`, `configs/live`, `research/pools`, `.omx/exports`, notepad/wiki 등 파일 단위 분류.
- `history_hits`: `.omx/logs`, `.omx/runtime/**/history.jsonl`, `.codex/**/history.jsonl` 키워드 hit preview.
- `git_commits`: `git log --all`에서 alpha/feature 관련 subject hit.

주의: `suggested_review_tier=gamma_usage_or_failure_review`는 실패/노이즈 언어가 있다는 뜻일 뿐 자동 gamma demotion이 아니다. 감마 강등은 위의 엄격 기준을 계속 적용한다.

## Alpha promotion thresholds

- `feature_pool`: feature/gate/exit idea. It can be useful even before strategy-level acceptance.
- `alpha_pool candidate`: a combined strategy using the feature must clear `test2024 CAGR / strict MDD >= 2.5`.
- `live` / live-grade standalone: a single strategy must clear `CAGR / strict MDD >= 3.0` as a standalone strategy, with strict MDD including in-position adverse excursion and split-end forced close.
- Below-threshold discoveries stay as `weak` alpha evidence or feature/selector candidates, not promoted alpha candidates.

## feature entry 추가 필드

- `data_sources`: 원천 데이터 경로/테이블.
- `feature_examples`: 대표 feature 이름.
- `feature_names`: alpha search에서 실제 사용한 구체 feature 이름 전체 또는 wildcard 목록.
- `usable_for`: 추천 사용처.
- `known_failures`: 실패한 사용법.
- `generation_recipe`: 위 복원 recipe.

## alpha entry 추가 필드

- `side`: `long/short/both`.
- `entry_logic`: 사람이 읽는 조건.
- `exit_logic`: fixed hold/dynamic exit/TP/SL 등.
- `stats`: split별 핵심 지표. 가능하면 `return_pct`, `cagr_pct`, `strict_mdd_pct`, `cagr_mdd`, `trades`, `win_rate`, `sharpe_like`.
- `dependencies`: feature ids.
- `usage_recipe`: 위 복원 recipe.

## portfolio entry 추가 필드

- `gross`: gross leverage/weight sum.
- `weights`: sleeve weight map.
- `constraints`: 비용/MDD/weight 단위 등.
- `component_alphas`: alpha ids.
- `stats`: split별 portfolio metrics.
- `construction_recipe`: 위 복원 recipe.
