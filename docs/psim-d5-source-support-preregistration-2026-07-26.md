# PSIM-D5 source-support preregistration

## 상태

PSIM-D5의 source-only 계약을 사전 등록했다.

`official source execution is not authorized`

현재 허용된 다음 단계는 preregistration에 byte/hash로 고정된
source-support evaluator를 구현하고 독립 검토 후 execution seal을
만드는 것뿐이다. fresh official source one-shot, 시장 데이터, 모델,
reward, 거래, PnL, CAGR, strict MDD 및 outcome 접근은 아직 허가되지
않는다.

## 권위 체인

| 권위 | 값 |
|---|---|
| D5 선택 commit | `0e62ec05e6861b2619e6737dd594e7306ad7c93a` |
| D5 선택 문서 SHA-256 | `364302ddada267c7252c37cd211f088893597917ab6ea3bbe99f896c647beba1` |
| D4 preregistration commit | `7731f8322b1700550ff1aa46d8a6c6898c31eef0` |
| D4 preregistration manifest | `b37fe58cf7a043d2164f2e3b08856a75fefad87aef85c02083873e7f3cffb1c8` |
| D4 terminal commit | `e406778f76f252d3b0cddb33242edcb51e984c80` |
| D4 terminal result hash | `8563ef3ace444896295d7076cd0f839e8f62f89899e312d711f9768f5cbf84aa` |
| D5 semantics probe SHA-256 | `42265a1ed0899366047732e1fa5dad24d961bb4b0bd7fb7bb58479a77bc8894b` |
| D5 semantics probe result hash | `467f4272bc7276879c0087662a70d99c57d9cef421647f1a679e2fce65de4871` |

D4 terminal은 gate 4에서 source-only로 거절됐고, market/model/outcome
counter는 모두 0이었다. D5 probe는 synthetic fixture와 이미 동결된
D4 terminal/census artifact만 읽었다.

## D4 → D5 허용 delta

recursive contract delta는 정확히 52개 path다. 허용된 범주는 다음뿐이다.

1. candidate/decision/protocol namespace
2. fresh D5 source root, sealed ref, artifact path, failure action
3. D5 event semantics 계약
4. known-invalid metadata와 dependency `UNKNOWN` 표현
5. administrative migration의 source-audit 보존과 model-card 제외
6. model field `normalized_text_delta`와 승인 section 경계
7. source gate 4의 strict-parser-totality를 explicit-state-totality로 교체
8. 위 의미론을 반영하는 세 relation-control 설명

그 외 availability, split, boundary reset, bucket, feasibility exclusion,
forbidden access, official source roster, gate roster, control roster,
Git clone/hydration mechanics는 D4와 byte-equal이다.

| 계약 | hash |
|---|---|
| authorized D4→D5 delta | `81ec7b54b199801bf5f68f78c03de1c96583eb6c2a061124a2367966457c190d` |
| D5 event semantics | `d73f97f980009b199d918bce662876f29e3559ee618a38679ec5209aa8404dcf` |
| D5 batch hydration | `e2cff3df57a398ba65072b4243077c68f4ba71e44e5c11f182c7d884c4721381` |

## Model boundary

### Model-visible

- `normalized_text_delta`
- 승인된 본문 section:
  - `ABSTRACT`
  - `MOTIVATION`
  - `SPECIFICATION`
  - `RATIONALE`
  - `BACKWARD_COMPATIBILITY`
  - `SECURITY`
  - `TESTS`
  - `IMPLEMENTATION`
- non-administrative event의 old/new metadata state
- known-invalid 여부
- dependency state `UNKNOWN_INVALID_METADATA`와 null count

### Audit-only

- exact old/new source path
- path identity hash
- full normalized delta hash와 변경 줄 수
- raw header, `OTHER`, `COPYRIGHT`
- proposal number, status, author, date, URL, dependency ID
- administrative redirect event

legacy `intent_text` field는 허용하지 않는다. normalized algorithmic
delta는 작성자의 causal intent라고 주장하지 않는다.

## Metadata와 administrative migration

known-invalid state:

- `INVALID_DUPLICATE_IDENTICAL`
- `INVALID_DUPLICATE_CONFLICTING`
- `INVALID_MALFORMED_HEADER`
- `INVALID_SELF_DEPENDENCY`

이 state에서는 header/dependency를 first/last-wins, merge, dedupe,
rename 또는 self-edge drop으로 수리하지 않는다. dependency는
`UNKNOWN_INVALID_METADATA`, count는 `null`이다. `INVALID_UNKNOWN`은
model/outcome 전에 fail closed한다.

exact Ethereum ERC move stub은 target 번호가 path proposal 번호와
같을 때만 administrative quarantine이다. source event artifact에는
남지만 model card와 memorization challenge에서는 제외한다. invalid
metadata → redirect라면 invalid state도 audit에 보존한다.

공식 migration 근거:

- [EIPs ERC 분리 commit](https://github.com/ethereum/EIPs/commit/0f44e2b94df4e504bb7b912f56ebd712db2ad396)
- [path case 교정 commit](https://github.com/ethereum/EIPs/commit/47ce70257fae525a427780630bd8d1903cc96e75)
- [EIP-1](https://eips.ethereum.org/EIPS/eip-1)
- [YAML 1.2.2](https://yaml.org/spec/1.2.2/)

이 reference note는 selection evidence이며 model input이 아니다.

## Gate 4 totality

Ethereum 5,206 hydrated blob에 대해 preregistered class roster는:

| class | blob |
|---|---:|
| `D4_VALID` | 4,440 |
| lower-case ERC redirect | 365 |
| upper-case ERC redirect | 365 |
| identical duplicate header | 7 |
| malformed header | 20 |
| self dependency | 9 |

gate 4는 모든 blob이 strict D4 valid, exact administrative redirect,
또는 관측된 known-invalid state 중 하나로 설명될 때만 통과한다.
새로운 문법 오류, redirect target mismatch, 부가 text, reverse
administrative transition은 fail closed한다. source drop/repair는
허용하지 않는다.

## Fresh execution namespace

| 항목 | 값 |
|---|---|
| source root | `/tmp/psim-d5-source` |
| sealed ref | `refs/psim-d5/sealed-tip` |
| failure action | `REJECT_PSIM_D5_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES` |
| source-object reuse | D1/D2/D3/D4 모두 금지 |
| shared objects/cache/worktree | 금지 |
| checkout/index/status | 금지 |
| batch hydration | replica당 단일 complete OID manifest fetch |
| post-read | `GIT_NO_LAZY_FETCH=1` |

## Machine artifacts

| artifact | SHA-256 / hash |
|---|---|
| `results/protocol_specification_intent_maturity_d5_preregistration_2026-07-26.json` | `11465540d59181bc48ea28c5164579847cbd936bf005c69d874ec2c873c949b9` |
| preregistration manifest | `f08eeb300fceb906cdcde485b4bce184c48d4cb14a1cd9028046e0c21a287309` |
| `training/preregister_protocol_specification_intent_maturity_d5.py` | `cc47bd574db47ee8857fc07cd9ff9a168b996e1d9a07ae0251d2ae85c1fdc7c6` |
| `tests/test_preregister_protocol_specification_intent_maturity_d5.py` | `e6e8b014dec1e3c9634218a026353abe7c35da53b04ee662de2f6f6145cab186` |

다음 구현은 이 파일들의 commit/SHA와 preregistration manifest를
정적으로 결합해야 한다. 구현 또는 테스트가 바뀌면 기존 seal은
무효이며 official one-shot 전에 다시 독립 검토해야 한다.
