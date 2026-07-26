# PSIM-D7 Bitcoin 문법 mechanism 선택

## 결정

PSIM-D7의 source representation 후보로 다음 합성 mechanism을 선택한다.

`PSIM_D7_UNIQUE_LATER_BIP_HEADER_PLUS_PREFIXED_DEPENDENCY_V1`

이 결정은 **D7 preregistration을 작성할 권한만** 부여한다. 공식 source
실행, model aggregation, LLM 호출, 시장 데이터 접근, 성능 평가 또는
candidate 승인을 허용하지 않는다.

## 근거

PSIM-D6 terminal 이후 변경되지 않은 Bitcoin forensic replica에서
434개 blob과 371개 event를 전수조사했다.

- 기존 parser 통과: 426 blob
- 비헤더 서문 뒤 unique exact `<pre>` header: 7 blob / 6 event
- exact uppercase `BIP-<decimal>` dependency token: 1 blob / 1 event
- unknown grammar: 0

이 결과는
`results/protocol_specification_intent_maturity_d6_bitcoin_grammar_census_2026-07-26.json`
에 고정됐다.

개별 proposal, event, blob 또는 commit identity는 mechanism 조건에
사용하지 않는다. 관측 roster는 hash로만 결박된다.

## Mechanism 1: unique later BIP header

### 정상 경로

기존 frozen `parse_bip_preamble(raw)`를 항상 먼저 실행한다. 성공하면
기존과 동일하게 header `BIP` number와 path proposal number가 같아야
한다. 이때 anchor는 `INITIAL_FROZEN_BIP_PARSER`다.

### 제한된 fallback

Fallback은 초기 parser가 정확히
`ValueError: PSIM malformed header line`을 낸 Bitcoin 문서에만 열린다.

1. 기존 strict UTF-8/NUL/blob/line normalization을 먼저 통과한다.
2. normalized 전체 문서에서 exact `<pre>` / `</pre>` pair를 찾는다.
3. Fallback 문서의 exact fence 전체가 balanced·non-nested여야 한다.
   unmatched opening/closing과 nested/overlapping fence는 즉시 거부한다.
4. 각 balanced block을 수정 없이 기존 frozen BIP parser에 넣는다.
5. parseable BIP header block이 전체 문서에서 **정확히 하나**여야 한다.
6. 그 block의 `BIP` number가 path proposal number와 같아야 한다.
7. 선택된 header block 앞에는 다른 fenced block이 하나도 없어야 한다.
   따라서 malformed/duplicate header block을 무시하고 다음 block을
   선택할 수 없다.
8. block 앞에는 최소 한 줄의 nonblank prefix가 있어야 한다.
9. prefix, header, body 전체 normalized text를 그대로 보존한다.

0개 또는 2개 이상, matching+nonmatching 두 header, path mismatch,
다른 초기 error, malformed fence는 모두 fail closed한다. first/last
wins는 없다.

## Mechanism 2: prefixed Bitcoin dependency

Bitcoin의 기존 dependency fields에 한해 comma-separated token을:

- bare `[0-9]+`
- exact uppercase `BIP-[0-9]+`

두 형태로만 받는다. 기존 D6 token 규칙과 같이 각 comma-separated
token의 바깥 SP/HTAB은 먼저 제거하지만, prefix 내부 공백은 허용하지
않는다.

`BIP-` 제거는 integer edge 검증에만 적용한다. source header와
normalized/model-visible text는 바꾸지 않는다. 기존 positive decimal,
maximum dependency count, sorting, duplicate, self-dependency 규칙을
그대로 적용한다.

다음은 모두 거부한다.

- lowercase prefix
- prefix 내부 공백
- range/suffix/free text
- zero/negative/nondecimal
- bare/prefixed cross-style duplicate
- self dependency
- count overflow
- multiline value

## D6에서 바꾸지 않는 부분

D7 grammar mechanism은 다음 D6 계약을 수정하지 않는다.

- 365개 exact ERC migration causal receipt
- migration model-hidden quarantine
- `section|direction|line` causal model rows
- chunk당 최대 8,192 UTF-8 bytes
- event당 최대 8 chunks
- byte-for-byte reconstruction
- 9번째 chunk fail closed
- full typed outcome roster
- unknown grammar가 있으면 market/model/outcome 전에 terminal reject

D6 mechanism binding:

- commit:
  `f985acb9821913e10325ed9487bdcea8fc2d39d9`
- artifact SHA-256:
  `01b09218d71d83c6abc3c4225b708a1cae6fe9e426b9bbd98f4fe6e86579d60b`
- result hash:
  `dda4b4786b34064a104178580f6cd33e56d5616c282515f6579105231b5dab38`

## Synthetic battery

23개 source-independent scenario를 전부 통과했다.

### 통과

- normal initial header
- generic prefix 뒤 unique later header
- exact prefixed dependency
- bare/prefixed mixed dependencies
- maximum dependency count

### 예상대로 fail closed

- later header path mismatch
- parseable later header 2개
- matching + mismatching header 동시 존재
- unmatched opening 전/후
- stray closing fence
- nested fence
- malformed header block 뒤 valid header
- duplicate header field block 뒤 valid header
- 허가되지 않은 initial parser error
- NUL
- invalid UTF-8
- lowercase/spaced/range dependency token
- self dependency
- cross-style duplicate
- dependency count overflow

Synthetic fixture의 원문은 artifact에 넣지 않았다. Scenario ID,
expected/observed typed outcome, receipt hash와 roster hash만 게시했다.

## 접근 경계

Probe는 다음만 읽었다.

- committed D6 Bitcoin grammar census artifact
- committed D6 mechanism artifact
- 각 producer/test/document hash

다음은 열지 않았다.

- `/tmp/psim-d6-source`
- historical proposal text
- network
- market/funding/future return
- model/LLM
- reward/trade/PnL
- CAGR/strict MDD
- outcome

PSIM-D6 evaluator를 재실행하지 않았다.

## 재현 artifact

- mechanism probe:
  `results/protocol_specification_intent_maturity_d7_mechanism_probe_2026-07-26.json`
- SHA-256:
  `2a549e6acfac2127527272ffe69986177b5e36f68f66623c6921ababac35ee94`
- result hash:
  `832b1327d19b29f44f4fbd76dac312e001a7da19eb813ce41277d64a45492371`
- synthetic scenario roster hash:
  `96c44d9f3c1cc2b84ce69fd3787195b095a2a3da3427ef4b59eb319383c9aff0`
- producer:
  `training/probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_mechanism.py`
- regression:
  `tests/test_probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_mechanism.py`

## 다음 단계

다음 커밋은 이 mechanism과 D6 불변 계약을 exact hash로 결박한
PSIM-D7 source-only preregistration이다. Preregistration 이후에도
reviewed implementation과 direct-child execution seal 없이는 공식 source
평가를 실행할 수 없다.
