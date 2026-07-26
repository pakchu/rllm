# PSIM-D6 source-support 사전등록

## 상태와 범위

PSIM-D6는 PSIM-D5 Gate 4의 두 실패 원인을 source representation
수준에서만 수정한다.

1. 고정된 365개 ERC migration episode의 exact receipt-bound restoration
2. D5 causal text rows의 lossless deterministic UTF-8 chunk transport

이 사전등록은 **공식 source 실행을 허가하지 않는다
(`does not authorize official source execution`)**. 다음 허용 작업은
synthetic-only D6 evaluator를 구현·테스트·리뷰한 뒤, preregistration과
코드를 함께 묶는 별도 execution seal을 만드는 것이다.

시장 데이터, 모델, reward, 거래, PnL, CAGR, strict MDD와 outcome은
열지 않았다. D5 official runner를 다시 실행하지 않았고,
`/tmp/psim-d5-source`도 열거나 수정하지 않았다.

## 고정 authority

| authority | 값 |
|---|---|
| D6 mechanism decision/probe commit | `f985acb9821913e10325ed9487bdcea8fc2d39d9` |
| D6 mechanism probe SHA-256 | `01b09218d71d83c6abc3c4225b708a1cae6fe9e426b9bbd98f4fe6e86579d60b` |
| D6 mechanism probe result hash | `dda4b4786b34064a104178580f6cd33e56d5616c282515f6579105231b5dab38` |
| D5 preregistration manifest hash | `f08eeb300fceb906cdcde485b4bce184c48d4cb14a1cd9028046e0c21a287309` |
| D5 terminal rejection commit | `0f69f7472d89474052186bbb2b13fa8d6bf5d77f` |
| D5 terminal rejection SHA-256 | `ffdebf2e5107f08345f16e21adc895d3bfc2f236d6b231322d03c372d4764ca1` |
| D5 terminal result hash | `0a23218e8784599f09e092d4f93942a48111c0af4f8e3ff85e2183eb84f56c56` |
| D5 post-terminal census commit | `4c4e3eb49962597eac7f63a0a3c1bf58f1fe73e4` |
| D5 census SHA-256 | `df2bcbef28b22d6daeb258d5c0f36b918d833b9c5fb0e5c9229a44edce4c2d59` |
| D5 census result hash | `0ca0a11f6693543dafbcf29052f2e963bf721c6e12f71f6fc9fbb1856e2dfe4a` |
| migration proposal roster hash | `c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd` |
| migration episode roster hash | `7065c33783f1ea54af1522da7e442ec05507c38355bb98ed90daf3f87e89b0bd` |
| 365-receipt manifest hash | `abf21a4691e4407158efc61a267cc6eaec8522751c25fa531aed6f782accdc07` |
| text-bound failure event roster hash | `0f299221248e66ca1eddc9cdd839cab504755537e47464c697c481544d169fd4` |

D6 builder는 위 artifact의 canonical JSON, SHA-256, result/manifest hash,
producer, test와 결정 문서를 모두 검증한다. D6 probe replay도 committed
artifact와 byte-equivalent여야 한다.

## D5 → D6 허용 델타

D5 contract core와 D6 successor core의 recursive value delta는 정확히
42개 path로 고정했다.

- authorized delta hash:
  `cb866ddfc173c294140725c391e0698cef779f6e3ee320dbdac6926f955bfbf0`
- source mechanism contract hash:
  `d03ea5ccacd415dab0f7d839b842d14e52492324c4fe432a723b6a2acb2c27b2`
- batch hydration contract hash:
  `880d4cdc7775ff34faa06ebe0a67b9f6b10656739860eae2173500697be14104`
- execution authorization contract hash:
  `4fbbe236be3844a65cc4d65f0fd2420f3c8f464b4c0e2c2fba248171a327d0b2`

변경 범위는 candidate/decision namespace, D6 event overlay, chunk model
field, Gate-4 totality, D6 artifact/root/ref namespace와 failure action뿐이다.
다음 항목은 D5와 동일하다.

- source gate roster와 relation-control roster
- availability, split, boundary-reset, bucket contract
- forbidden-access contract
- official source roster
- hydration transport mechanics

Hydration contract는 failure namespace와
`D1, D2, D3, D4, or D5 source-object reuse` 금지만 바뀐다. 이를 D5
namespace로 되돌리면 D5 hydration contract와 byte-equal이다.

## Exact migration restoration

일반적인 administrative-to-valid 예외는 없다. restoration은 다음 조건을
전부 만족할 때만 허용한다.

- D5 census가 고정한 proposal 365개 중 하나
- 전체 365개 authority roster와 receipt manifest가 정확히 일치
- exact three-step commit/day/blob-class sequence
- 동일 proposal path와 redirect target
- old/new blob OID 및 SHA-256 continuity
- episode canonical receipt가 해당 proposal의 고정 receipt와 일치

운영 경로는 caller가 receipt map이나 manifest를 주입할 수 없어야 한다.
고정 D5 census authority를 내부에서 로드하는 public authorizer만 사용할
수 있다. probe의 caller-supplied helper는 synthetic negative-control
전용이다.

승인된 restoration도 model-hidden administrative quarantine이다.
model payload는 다음 세 필드뿐이며 text chunks는 비어 있다.

```json
{
  "administrative_quarantined": true,
  "model_visibility": "ADMINISTRATIVE_QUARANTINE",
  "normalized_text_delta_chunks": []
}
```

proposal/path/commit/blob ID, receipt hash와 quarantine reason은 audit-only다.

## Lossless UTF-8 chunk transport

D5 causal rows를 다음 순서 그대로 직렬화한다.

```text
section|direction|line
```

- row 사이는 단일 LF, 시작/끝 추가 LF 없음
- 첫 번째와 두 번째 pipe만 section/direction 경계
- **두 번째 pipe 뒤의 line은 opaque text이며 다시 split/parse하지 않음**
- strict UTF-8
- offset 0부터 최대 8,192 bytes를 greedy하게 선택
- UTF-8 continuation byte에서는 code-point 경계까지 backtrack
- 최대 8 chunks, event 최대 65,536 bytes
- chunk 삭제·중복·재배열·변조·non-greedy repartition 금지
- 재결합 bytes가 full text와 byte-for-byte 일치해야 함
- 9번째 chunk 필요 시 typed event error 후 full roster를 계속 수집
- truncation과 summarization 금지

Model-visible chunk item의 필드는 정확히 다음 세 개다.

```text
normalized_text_delta_chunk
chunk_index
chunk_count
```

Full-text/chunk hash, byte count와 reconstruction receipt는 model-hidden
audit 자료다. Chunk는 event나 label이 아니라 transport fragment다.

Chunk별 추론 결과를 어떻게 결합할지는 source representation 문제가
아니므로 이 사전등록에서 결정하지 않았다. 고정 값은 다음과 같다.

```text
UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION
```

모델 단계를 열기 전에 별도 preregistration이 필요하다.

## Gate 4 totality

D5 runner는 첫 `ValueError`에서 Gate 4를 중단해 전체 failure roster를
terminal artifact에 남기지 못했다. D6 구현은 successful hydration 뒤의
event semantics 오류를 예외로 탈출시키지 않는다.

1. 네 개 fresh replica의 retained 2020-2023 proposal-group event를 전부
   평가한다.
2. event마다 정확히 하나의 typed audit outcome을 남긴다.
3. 양 replica의 outcome roster identity를 검증한다.
4. unknown grammar, receipt mismatch, noncanonical chunk, 9번째 chunk,
   strict UTF-8 오류가 있어도 roster 수집을 계속한다.
5. 전체 roster 완료 후 Gate 4를 판정한다.
6. unauthorized error outcome이 하나라도 있으면 market/model/outcome
   이전에 canonical rejection을 게시한다.
7. semantic event error 때문에 artifact 게시 전 return/raise할 수 없다.

Rejection report에는 raw/normalized text를 게시하지 않는다. 이 totality
계약은 authority/source I/O가 성공해 event semantics 평가가 시작된
범위에 적용된다.

## Fresh-root 및 실행 경계

- 공식 D6 source root: `/tmp/psim-d6-source`
- sealed ref: `refs/psim-d6/sealed-tip`
- D5 forensic/source root 재사용: 금지
- repair/provider swap/retry/fallback: 금지
- 이 preregistration의 official execution authorization: false
- D6 synthetic probe의 official execution authorization: false

공식 실행 전에 다음 세 authority가 별도로 필요하다.

1. reviewed D6 implementation commit
2. reviewed D6 test commit
3. preregistration과 위 코드를 묶는 canonical D6 execution seal

## Machine artifact

| 항목 | 값 |
|---|---|
| preregistration JSON | `results/protocol_specification_intent_maturity_d6_preregistration_2026-07-26.json` |
| JSON SHA-256 | `9b6177ba02bf02783f7ddffe90cf4c5f1e385422ff658e17b28bf72d2f051d82` |
| manifest hash | `0d6be5118ef7b34031af61bccc8a28944109db1a5411635ac2c822388e8895a6` |
| producer | `training/preregister_protocol_specification_intent_maturity_d6.py` |
| producer SHA-256 | `e81cda6e88f298c3682d605ae8ab1b9e05ea1ae6cd50085eb1df9dea851a20b1` |
| regression | `tests/test_preregister_protocol_specification_intent_maturity_d6.py` |
| regression SHA-256 | `ced15cc7b7fafa1e1b60d27f979334eba39dc09c5eedfab0f99c370e0a31192b` |

이 문서와 machine artifact는 D6 preregistration 준비 상태만 기록한다.
PSIM-D6 official source evaluator는 실행하지 않았다.
