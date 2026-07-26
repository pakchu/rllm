# PSIM-D7 source-support 사전등록

## 상태와 범위

PSIM-D7은 PSIM-D6의 source, transport, migration, split, control 계약을
그대로 상속하고 Bitcoin Gate 4에서 확인된 두 문법만 일반 규칙으로
추가한다.

1. frozen initial BIP parser가 정확히 `ValueError: PSIM malformed header
   line`으로 실패한 경우, 전체 문서에서 유일한 parseable later `<pre>`
   BIP header를 사용할 수 있다.
2. Bitcoin dependency token은 기존 bare decimal 외에 exact uppercase
   `BIP-<decimal>`을 허용한다.

이 사전등록은 **공식 source 실행을 허가하지 않는다
(`does not authorize official source execution`)**. 다음 허용 작업은
synthetic-only D7 evaluator를 구현·테스트·독립 리뷰하고, 그 구현
commit의 direct child로 preregistration과 코드를 묶는 execution seal을
만드는 것이다.

시장 데이터, 모델, reward, 거래, PnL, CAGR, strict MDD와 outcome은
열지 않았다. D6 official runner를 다시 실행하지 않았고,
`/tmp/psim-d6-source`와 신규 `/tmp/psim-d7-source`도 열거나 수정하지
않았다.

## 고정 authority

| authority | 값 |
|---|---|
| D6 preregistration commit | `a2ff036d03f01750da3527666e3be3d44737cbe2` |
| D6 preregistration SHA-256 | `9b6177ba02bf02783f7ddffe90cf4c5f1e385422ff658e17b28bf72d2f051d82` |
| D6 preregistration manifest | `0d6be5118ef7b34031af61bccc8a28944109db1a5411635ac2c822388e8895a6` |
| D6 implementation commit | `5c3f3f6d26046a8bc7b2f7ad09178d944d61e17b` |
| D6 execution-seal commit | `8185e14b2e98fef6a4f8545828dc48b7d98417f2` |
| D6 execution-seal SHA-256 | `cf9bdbea467a499c6075059ef9275f00699fb0431fa27643751539ffdea64e1d` |
| D6 terminal commit | `aef35e00f3ddcb91f6f4b6a37ff40d9d9f67a7a4` |
| D6 terminal SHA-256 | `f3e69893270be0d37299e78b651daa9208e1d05f07f24b39f6d1cf9a71c5d49f` |
| D6 terminal result hash | `052c8a0c5f3584a3c9a970f1fcfc434ebfd59a6aa25d5e087c6554aa3f2c31da` |
| D6 Bitcoin census commit | `bfa35dc12f1c5a2cf0cee0be4bbd85347f8ab7c3` |
| D6 Bitcoin census SHA-256 | `8bfe4a6c44a4c5381bb98caf2ffea57b42f2b3d77caec9e656895336b72d0217` |
| D6 Bitcoin census result hash | `7ef74a017f8c0c1eb416608dcf59c2ce74af6587f5a71203b53e846d31c039ed` |
| frozen D6 mechanism result hash | `dda4b4786b34064a104178580f6cd33e56d5616c282515f6579105231b5dab38` |
| D7 mechanism commit | `fe7f1d123eaf65af35d0d43e90f79faccbc53622` |
| D7 mechanism probe SHA-256 | `2a549e6acfac2127527272ffe69986177b5e36f68f66623c6921ababac35ee94` |
| D7 mechanism result hash | `832b1327d19b29f44f4fbd76dac312e001a7da19eb813ce41277d64a45492371` |

D7 builder는 위 canonical JSON, SHA-256, result/manifest hash와 각 census,
mechanism의 producer, test, decision document hash를 검증한다. D7
synthetic probe의 23-scenario battery도 committed artifact와
byte-equivalent여야 한다.

D6 terminal은 Gate 1~3 통과, Gate 4 실패, 11,280 blob/text row 완전
수집, market/model/outcome 접근 0, source attempt 1, repair/provider swap
없음으로 고정된다. D6 terminal commit은 D6 execution-seal commit의
direct child로 기록했다.

## D6 → D7 허용 델타

D6 contract core와 D7 successor core의 recursive value delta는 정확히
37개 path다.

| contract | hash |
|---|---|
| authorized delta | `b8295e977db265278d533d8fdc8f3dbf70e5905a04a95f7b936c191b0cd09440` |
| D7 grammar overlay | `271b6b0447d392c341420d50f174aa8f5017c0c2f0fed99e453a4dacef00a977` |
| D7-namespaced batch hydration | `98e9bb09e8d296d577020477c6e984c0333ab925e4aa68f22a13ab5d211cc492` |
| execution authorization | `21c7a2722fa19a20d937a232ca658d11e3bd7c35d67c05f16d5ac0814ea039bb` |

허용 변경은 다음뿐이다.

- candidate/decision/protocol D7 namespace
- `event_contract.d7_bitcoin_grammar` 단일 overlay
- D7 root, artifact와 sealed-ref namespace
- D7 failure-action namespace
- D6 source-object reuse 금지 추가
- D7 implementation/test/direct-child seal 요구
- model aggregation의 D7 미결정 표기

다음은 D6와 동일하다.

- Ethereum parser/event semantics
- exact 365 ERC migration restoration와 receipt authority
- lossless UTF-8 chunk transport, 8,192 bytes × 8 chunks
- truncation/summarization 금지
- D6 source mechanism object 전체
- repository remotes와 sealed-tip OID
- source/card interval
- archive schedules와 train/test/eval split
- gate roster와 relation-control roster
- official-source roster와 forbidden-access contract
- hydration transport mechanics

Hydration contract는 D7 failure namespace와
`D1, D2, D3, D4, D5, or D6 source-object reuse` 금지만 바뀐다. 이를 D6
namespace로 되돌리면 D6 hydration contract와 byte-equal이다.

## Bitcoin grammar overlay

### Later-header fallback

1. frozen initial BIP parser를 항상 먼저 실행한다.
2. fallback은 initial parser가 정확한 historical error를 낸 경우만
   가능하다.
3. exact `<pre>` fence는 balanced/non-nested여야 한다.
4. 전체 balanced block에서 parseable BIP header는 정확히 하나여야 한다.
5. header의 `BIP` 값은 path proposal과 같아야 한다.
6. 선택된 header 앞에는 다른 fenced block이 없어야 한다.
7. prefix, header, body와 full normalized text를 모두 보존한다.
8. malformed, multiple, unknown grammar는 typed error로 남기고 full roster
   수집 후 market/model/outcome 이전에 reject한다.

### Dependency token

- 허용: `[0-9]+`, exact uppercase `BIP-[0-9]+`
- SP/HTAB outer trimming: D6 규칙 상속
- `BIP-` 제거: integer edge validation에만 적용
- source text: 변경하지 않음
- positive/self/duplicate/count 규칙: D6와 동일
- lowercase, spaced prefix, range, unknown token: fail closed

proposal/BIP/OID별 allowlist는 없다. Census가 확인한 7개의 later-header
blob과 1개의 prefixed-dependency blob은 문법 범위 근거일 뿐 실행
예외 목록이 아니다.

## D6 transport와 model 경계

D6 causal rows의 직렬화와 canonical chunk partition은 완전히 고정된다.
재결합은 byte-for-byte여야 하고 9번째 chunk가 필요하면 typed error 후
complete roster를 수집하며 truncation과 summarization은 금지된다.

Chunk별 model output 결합은 source representation 범위가 아니므로
결정하지 않았다.

```text
UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION
```

별도 model-stage preregistration 전에는 model aggregation, inference,
reward, trade나 outcome을 열 수 없다.

## Fresh root와 실행 경계

- 공식 D7 source root: `/tmp/psim-d7-source`
- sealed ref: `refs/psim-d7/sealed-tip`
- D6 forensic/source root 재사용: 금지
- D6 official rerun: 금지
- source-object repair/provider swap 및 transport retry/fallback: 금지
- 이 preregistration의 official execution authorization: false
- D7 synthetic mechanism probe의 official execution authorization: false

공식 실행 전 별도로 필요한 authority는 다음 셋이다.

1. reviewed D7 implementation commit
2. reviewed D7 test commit
3. 위 동일 implementation commit의 direct child인 canonical D7 execution
   seal

## Machine artifact

| 항목 | 값 |
|---|---|
| preregistration JSON | `results/protocol_specification_intent_maturity_d7_preregistration_2026-07-26.json` |
| JSON SHA-256 | `e9402b984232a9c30a5bc427ee8b828b4e61b7f355746e36ee5fe986be3ae79d` |
| manifest hash | `7b6ac7c514bd3c0c8fad54a69707bb682a8a97bae020a603940c3410ddea378d` |
| producer | `training/preregister_protocol_specification_intent_maturity_d7.py` |
| producer SHA-256 | `669494125becd1e1ed82a3a3048eaad23a063ef49c24a1bf613ba828695203fa` |
| regression | `tests/test_preregister_protocol_specification_intent_maturity_d7.py` |
| regression SHA-256 | `720795099c82dc7dc04da41c5749527e4680a4c8ee16a4366e47e13b17d9270f` |

이 문서와 machine artifact는 D7 preregistration 준비 상태만 기록한다.
PSIM-D7 official source evaluator는 실행하지 않았다.
