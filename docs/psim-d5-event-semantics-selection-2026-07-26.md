# PSIM-D5 source event semantics 선택

## 결정

PSIM-D5의 source representation을 다음으로 고정한다.

`EXACT_PATH_IDENTITY + NORMALIZED_ALGORITHMIC_TEXT_DELTA + EXACT_ERC_MIGRATION_QUARANTINE + EXPLICIT_INVALID_METADATA_STATE`

이 선택은 PSIM-D4의 terminal rejection과 5,206개 hydrated Ethereum
blob 전수조사에만 근거한다. synthetic probe 외 공식 proposal source,
시장 데이터, 모델, reward, 거래, PnL, CAGR, strict MDD 또는 outcome은
열지 않았다. 따라서 이것은 **D5 preregistration만 허가**하며 alpha나
수익성을 주장하지 않는다.

## 선택 이유

PSIM-D4 strict parser는 5,206개 blob 중 4,440개만 통과했다. 실패
766개 중 730개(`95.3003%`)는 2023-10-25 ERC 저장소 분리 과정에서
생긴 exact move stub이었다. 나머지 36개는 duplicate, malformed 또는
self-dependency metadata였다.

strict metadata parse를 text 접근의 선행조건으로 두면:

1. 관리성 저장소 이동을 protocol intent로 잘못 학습하거나,
2. invalid metadata가 있는 실제 문서의 관측 가능한 text를 모두
   버리거나,
3. source에 없는 first/last-wins, dedupe, rename, self-edge 제거를
   합성하게 된다.

D5는 세 문제를 분리한다. proposal identity는 protocol, proposal
number, old/new exact group path와 그 canonical hash로 보존한다.
normalized algorithmic text delta는 metadata parse와 독립적으로
기록한다. metadata는 해석 가능 여부를 categorical state로 전달한다.
LLM은 관측 가능한 text 변화와 invalidity 상태를 함께 볼 수 있지만,
parser가 source에 없는 값을 만들지는 않는다.

## 고정된 계약

### Identity

- proposal identity:
  `PROTOCOL + PROPOSAL_NUMBER + EXACT_OLD_NEW_GROUP_PATHS + CANONICAL_HASH`
- header의 `eip`/`bip` 값은 identity를 교정하거나 대체하지 않는다.
- side path와 blob 존재 형태가 다르거나 protocol/number/path가
  일치하지 않으면 event 생성 전에 fail closed한다.

### Normalized algorithmic text delta

- frozen D1 byte/Unicode/line normalization을 그대로 사용한다.
- `SequenceMatcher(autojunk=False)` opcode 순서를 따른다.
- 각 opcode에서 source order로 `REMOVE` 후 `ADD`를 기록한다.
- model-visible 형식은 `SECTION|DIRECTION|LINE`이다.
- model-facing field 이름은 `normalized_text_delta`이며 source bytes를
  작성자의 intent라고 단정하는 alias를 두지 않는다.
- header와 `OTHER`, body section을 모두 포함한다.
- non-administrative event의 invalid metadata가 알려진 상태라면 text
  delta는 model-visible이다.
- 이 delta는 source bytes 사이의 deterministic algorithmic proxy다.
  작성자의 인과 의도나 line move의 의미를 안다고 주장하지 않는다.
- repeated-line과 moved-line synthetic fixture로 동일 입력의
  deterministic output을 검증한다.

### Metadata와 dependency

명시 상태:

- `VALID`
- `INVALID_DUPLICATE_IDENTICAL`
- `INVALID_DUPLICATE_CONFLICTING`
- `INVALID_MALFORMED_HEADER`
- `INVALID_SELF_DEPENDENCY`
- `INVALID_UNKNOWN`
- `ADMINISTRATIVE_REDIRECT`

알려진 invalid metadata는 header나 dependency를 수리하지 않는다.
dependency delta는 `UNKNOWN_INVALID_METADATA`, count는 `null`이다.
`INVALID_UNKNOWN`은 raw audit state까지만 만들고 model 또는 outcome
전에 fail closed한다. Bitcoin preamble parser는 D4 strict parser와
동일하다.

### ERC migration quarantine

다음 조건을 모두 만족할 때만 administrative quarantine이다.

1. Ethereum proposal이다.
2. 문서 전체가 한 줄 move stub이다.
3. target은 공식 `ethereum/ercs` 저장소의 `ercs/erc-N.md` 또는
   `ERCS/erc-N.md` 형식이다.
4. target 번호 `N`이 proposal group path 번호와 같다.

quarantine event는 normalized delta hash와 실제 변경 줄 수를 audit에
보존하지만 model text는 비우고 model line count를 0으로 만든다.
non-admin side에 알려진 invalid metadata가 있으면 그 explicit state도
audit에 보존한다. 즉 administrative precedence가 invalidity 사실을
`False`로 덮어쓰지 않는다.
target mismatch, extra text, 다른 형식은 quarantine하지 않는다.
redirect에서 일반 문서로 돌아오는 reverse transition도 의미가
불명확하므로 fail closed한다.

공식 이력:

- ERC 분리 commit:
  [`0f44e2b`](https://github.com/ethereum/EIPs/commit/0f44e2b94df4e504bb7b912f56ebd712db2ad396)
- path case 교정 commit:
  [`47ce702`](https://github.com/ethereum/EIPs/commit/47ce70257fae525a427780630bd8d1903cc96e75)
- EIP-1:
  [canonical page](https://eips.ethereum.org/EIPS/eip-1)
- duplicate mapping key 기준:
  [YAML 1.2.2](https://yaml.org/spec/1.2.2/)

이 URL/claim은 D4 census 설계 전에 수행된 공식 자료 research에서
가져온 selection evidence다. D5 probe 자체는 external network를
열지 않았고, reference notes는 model-visible input이 아니다.

## Synthetic battery

| 검증 | 결과 |
|---|---:|
| known invalid metadata state | 4/4 model-visible |
| invalid dependency | 4/4 `UNKNOWN`, 무수리 |
| exact redirect path case | 2/2 식별 |
| administrative transition | 4/4 quarantine |
| invalid → redirect | invalid state audit 보존 |
| unknown grammar | 3/3 fail closed |
| reverse migration | fail closed |
| D4 Bitcoin parse output | 1/1 byte-semantic equal |
| metadata + body normalized delta | 보존 |
| repeated/moved-line proxy | 2/2 deterministic |
| quarantine audit diff | 보존 |

모든 fixture는 synthetic structure이며 역사적 specification blob을
복사하지 않았다.

## 권한과 artifact

| 항목 | 값 |
|---|---|
| D4 terminal commit | `e406778f76f252d3b0cddb33242edcb51e984c80` |
| D4 terminal result hash | `8563ef3ace444896295d7076cd0f839e8f62f89899e312d711f9768f5cbf84aa` |
| D4 census commit | `6be3e767d4320da0ce9aa34d2cfabbf4ac0fb3ef` |
| D4 census result hash | `85ff0c04a1fe06b34b7f214f5fd7b9a1191a4ef0dd990a7e7d002f72efe9428d` |
| D5 probe | `results/protocol_specification_intent_maturity_d5_event_semantics_probe_2026-07-26.json` |
| D5 probe SHA-256 | `f4496846b979ba1e832b4a7108ae9575f0c0e44101f006062efc4453aa6f8799` |
| D5 probe result hash | `b94321b815f4f32cc8c8b6d9b323b88d3b8f29ab1e7e410f4b8266b92e4c186b` |
| semantics version | `PSIM_PATH_IDENTITY_NORMALIZED_TEXT_DELTA_V1_EXACT_ERC_QUARANTINE` |

다음 허용 단계는 D5 preregistration이다. official source execution은
preregistration, evaluator implementation, independent review, execution
seal을 각각 별도 커밋으로 고정한 뒤에만 한 번 수행할 수 있다.
