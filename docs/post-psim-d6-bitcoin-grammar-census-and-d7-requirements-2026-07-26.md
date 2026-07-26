# PSIM-D6 사후 Bitcoin 문법 전수조사와 D7 요구사항

## 결론

PSIM-D6는 Ethereum의 D5 실패 원인 두 가지를 해결했다.

- exact ERC migration restoration: 통과
- 8,192-byte lossless UTF-8 chunk transport: 통과
- Ethereum event semantics: **4,985 / 4,985 통과**

그러나 공식 one-shot은 Bitcoin에서 **8개 blob / 7개 event**를
표현하지 못해 Gate 4에서 terminal reject되었다. 변경되지 않은
forensic replica의 Bitcoin proposal blob 434개와 event 371개를
시장·모델·성과 데이터 없이 전수조사한 결과, 원인은 정확히 두 문법
범주였다.

| 문법 범주 | blob | event |
|---|---:|---:|
| 정상 D4 parser grammar | 426 | 364 |
| 비헤더 서문 뒤의 exact `<pre>...</pre>` BIP header | 7 | 6 |
| Bitcoin dependency field의 exact `BIP-<decimal>` token | 1 | 1 |
| 합계 | 434 | 371 |

미지의 제3 문법은 없었다. 원문, normalized text, proposal/event/blob
identity roster는 artifact에 공개하지 않았다. 범주별 count와 canonical
roster hash만 고정했다.

## D6 terminal 경계

| 항목 | 값 |
|---|---|
| terminal commit | `aef35e00f3ddcb91f6f4b6a37ff40d9d9f67a7a4` |
| terminal JSON SHA-256 | `f3e69893270be0d37299e78b651daa9208e1d05f07f24b39f6d1cf9a71c5d49f` |
| terminal result hash | `052c8a0c5f3584a3c9a970f1fcfc434ebfd59a6aa25d5e087c6554aa3f2c31da` |
| runner commit | `5c3f3f6d26046a8bc7b2f7ad09178d944d61e17b` |
| seal commit | `8185e14b2e98fef6a4f8545828dc48b7d98417f2` |
| Bitcoin commit rows / hash | `1,482` / `7e60f24b78aa863a2b317a7dc3a32b2af8e367c3d25f4a97012f4ddfd28d89d2` |
| Bitcoin groups / hash | `371` / `3f7a8e10bb5f9ba57bb0231b5cd54a613fb81e67830c1ec1d9781fe0d22b6a8b` |
| hydrated blob count / manifest | `434` / `33b974cdc205d35aa6436ea38424b81a272735cae8042679422906d72affc332` |
| pristine object-store hash | `cbbcfe08eb5e20cb4fe67d28ef482a35c520dd820a30c1be0daa1dc2a5e1c756` |

D6 공식 evaluator는 다시 실행하지 않았다. terminal artifact의
시장·funding·future return·model·reward·trade·PnL·CAGR·strict MDD
접근 카운터는 모두 0이고 `outcomes_opened=false`다.

## Forensic replica 무결성 사고

사후 census 첫 preflight에서 `bitcoin-a.git`의 object-store가 terminal
snapshot과 달라진 것을 fail-closed로 검출했다.

| 항목 | terminal 기대값 | 사후 관측값 |
|---|---:|---:|
| object-store hash | `cbbcfe08...` | `95ab65ba...` |
| object count | 9,890 | 10,354 |

원인은 증거 없이 추정하지 않는다. 해당 root는 즉시 격리했고 복구,
삭제, fetch, 재사용하지 않았다. Census는 terminal hash
`cbbcfe08...`를 그대로 유지한 `bitcoin-b.git`만 열었다. 조사 전후
object-store hash와 9,890개 object roster가 동일했으며 network command는
0이었다.

따라서 이 결과는 두 현재 replica의 재현 비교가 아니라:

1. D6 terminal이 이미 고정한 원래 A/B 동일 receipt,
2. 현재도 terminal hash와 정확히 같은 B replica,
3. B에서 재생한 chain/group/blob/event roster

의 결합 증거다. 격리된 A는 어떤 D7 candidate에도 사용할 수 없다.

## 실패 범주 1: 비헤더 서문 뒤의 exact BIP header

기존 parser는 첫 nonblank block이 바로 BIP header라고 가정한다.
역사적 문서 7개는 비헤더 서문 뒤에 정상적인 exact
`<pre>...</pre>` metadata block을 갖는다.

전수 조건:

- baseline error: `ValueError: PSIM malformed header line`
- blob: 7
- 관련 event: 6, 모두 `UPDATE`
- 각 blob의 exact later `<pre>` candidate: 정확히 1개
- candidate `BIP` number와 path proposal number: 모두 일치
- candidate header field: 모두 10개
- candidate dependency edge: 모두 0개
- 서문 nonblank line: 6~7
- unknown grammar: 0

이는 서문을 삭제하거나 header를 합성하라는 뜻이 아니다. D7 후보는
전체 원문을 그대로 보존하면서 **metadata extraction anchor**만 exact
block으로 이동해야 한다.

## 실패 범주 2: prefixed Bitcoin dependency token

정상 BIP header 한 개가 dependency field에서 bare decimal 대신 exact
uppercase `BIP-<positive decimal>` token을 사용한다.

- baseline error:
  `ValueError: PSIM proposal number is not ASCII decimal`
- blob/event: 1 / 1
- event type: `CREATE`
- dependency field: 1개
- token: exact prefixed decimal 1개
- path proposal number 일치
- self/duplicate dependency 없음

Prefix 제거는 dependency edge의 정수 검증에만 적용해야 한다. 원문과
model-visible delta의 token은 절대 바꾸지 않는다.

## D7 synthetic mechanism의 필수 계약

아직 D7 candidate 또는 preregistration을 승인하지 않는다. 먼저
source-independent synthetic probe가 다음 규칙을 증명해야 한다.

### 1. Exact later-header selection

1. 기존 strict parser를 먼저 실행한다.
2. Bitcoin에서만 기존 header parse가 실패했을 때 exact `<pre>` /
   `</pre>` pair를 전수 탐색한다.
3. 기존 header parser를 수정 없이 통과하고 `BIP` number가 path
   proposal과 일치하는 candidate가 **정확히 하나**일 때만 metadata
   anchor로 선택한다.
4. 0개, 2개 이상, fence 불일치, number mismatch, duplicate field,
   malformed field는 모두 fail closed한다.
5. 서문과 원문 body는 삭제·요약·재배열하지 않는다.
6. proposal/event/blob/commit allowlist로 분기하지 않는다.

### 2. Exact dependency token grammar

Bitcoin의 기존 dependency field에 한해서 comma-separated token을:

- bare `[0-9]+`, 또는
- exact uppercase `BIP-[0-9]+`

로만 허용한다. `BIP-`는 integer edge 검증에만 제거한다. positive,
maximum count, duplicate, self-dependency 규칙은 그대로 유지한다.
lowercase, 공백 삽입, 다른 prefix, suffix, range, free text는 모두
거부한다.

### 3. D6 불변 부분

- exact ERC migration restoration은 변경하지 않는다.
- D6 `section|direction|line` row와 8,192 bytes × 최대 8개 chunk
  transport를 변경하지 않는다.
- raw/normalized text reconstruction은 byte-for-byte여야 한다.
- 모든 event는 정확히 하나의 typed outcome을 가져야 한다.
- unknown grammar roster가 하나라도 있으면 전체 roster를 수집한 뒤
  market/model/outcome 접근 전에 terminal reject한다.
- model aggregation 규칙은 여전히 별도 preregistration 없이는
  승인되지 않는다.

## 기각한 대안

- 관측된 proposal, event, blob OID allowlist
- `ERROR_BLOB_DECODE_UNAVAILABLE`를 pass class로 이름만 변경
- 첫 번째 later `<pre>` block을 무조건 선택
- 여러 candidate 중 first/last wins
- dependency field 전체를 자유형 문자열로 허용
- header 값이나 dependency 원문 수정
- 실패 event 삭제 또는 model-hidden 일괄 quarantine
- 격리된 `bitcoin-a.git` 복구·재사용
- D6 공식 evaluator 재실행

## 재현 artifact

- census:
  `results/protocol_specification_intent_maturity_d6_bitcoin_grammar_census_2026-07-26.json`
- census SHA-256:
  `8bfe4a6c44a4c5381bb98caf2ffea57b42f2b3d77caec9e656895336b72d0217`
- census result hash:
  `7ef74a017f8c0c1eb416608dcf59c2ce74af6587f5a71203b53e846d31c039ed`
- producer:
  `training/audit_protocol_specification_intent_maturity_d6_bitcoin_grammar_census.py`
- regression:
  `tests/test_audit_protocol_specification_intent_maturity_d6_bitcoin_grammar_census.py`
