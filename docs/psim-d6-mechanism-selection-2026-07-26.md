# PSIM-D6 source mechanism 선택

## 결정

PSIM-D5의 terminal Gate 4 실패를 해결할 D6 source mechanism으로
다음 두 가지를 synthetic-only probe에서 선택했다.

1. **Receipt-bound exact migration restoration quarantine**
2. **Lossless deterministic UTF-8 model-text chunks**

이 선택은 D6 preregistration만 허가한다. 공식 source 실행, forensic
root 재사용, 시장 데이터, 모델, reward, 거래, PnL, CAGR, strict MDD
또는 outcome 접근은 허가하지 않는다.

## Authority

| 항목 | 값 |
|---|---|
| D5 census commit | `4c4e3eb49962597eac7f63a0a3c1bf58f1fe73e4` |
| D5 census JSON SHA-256 | `df2bcbef28b22d6daeb258d5c0f36b918d833b9c5fb0e5c9229a44edce4c2d59` |
| D5 census result hash | `0ca0a11f6693543dafbcf29052f2e963bf721c6e12f71f6fc9fbb1856e2dfe4a` |
| migration proposal roster hash | `c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd` |
| migration episode roster hash | `7065c33783f1ea54af1522da7e442ec05507c38355bb98ed90daf3f87e89b0bd` |
| per-proposal receipt manifest hash | `abf21a4691e4407158efc61a267cc6eaec8522751c25fa531aed6f782accdc07` |
| text-bound failure event roster hash | `0f299221248e66ca1eddc9cdd839cab504755537e47464c697c481544d169fd4` |

Probe는 D5 census JSON뿐 아니라 그 producer, test와 설계 문서의
SHA-256도 검증한다. D5 census는 5,206개 blob과 4,985개 event를
전수 조사했으며, D6 probe는 해당 artifact만 읽고
`/tmp/psim-d5-source` 또는 proposal 원문을 열지 않는다.

## Mechanism 1: exact migration restoration

D6는 일반적인 `ADMINISTRATIVE_REDIRECT → VALID` 예외를 만들지 않는다.
복원 event는 다음 조건이 모두 맞을 때만 model-hidden administrative
quarantine으로 처리할 수 있다.

- proposal이 고정된 365개 receipt authority에 존재
- 동일한 `EIPS/eip-N.md` path
- 세 event 모두 `UPDATE`
- exact commit sequence:
  1. `0f44e2b94df4e504bb7b912f56ebd712db2ad396`
  2. `47ce70257fae525a427780630bd8d1903cc96e75`
  3. `25cdf1d059778236e28bf22d752ca48a35af91f6`
- exact day sequence:
  `2023-10-25`, `2023-10-25`, `2023-10-26`
- exact class sequence:
  `VALID → lower redirect → upper redirect → VALID`
- old/new blob OID와 SHA-256 continuity
- lower/upper redirect target이 path proposal `N`과 일치
- 전체 episode canonical hash가 해당 proposal의 D5 census receipt와
  정확히 일치
- 전체 receipt map의 canonical manifest hash도 고정 authority와 일치

마지막 조건은 개별 receipt가 맞더라도 authority roster를 삭제하거나
새 proposal을 추가하는 것을 막는다.
운영용 authorizer는 caller가 receipt map이나 manifest hash를 주입받지
않는다. D5 census의 고정 365개 map과 manifest hash를 내부에서
검증·로드한다. 한 건짜리 synthetic authority 주입 경로는 private
probe helper로만 남긴다.

### Synthetic 결과

- exact three-step episode: 승인
- restoration model text chunks: 0
- generic reverse transition: 승인하지 않음
- mutation/authority negative controls: 12개 모두 거부

Negative controls에는 path, target, commit, day, class, blob continuity,
step order, one-step generic reverse, extra forbidden field, altered receipt,
receipt roster 삭제와 확장이 포함된다.

## Mechanism 2: UTF-8 chunk transport

### Row serialization

D5의 causal model row 순서를 그대로 유지한다.

```text
section|direction|line
```

- row 사이는 단일 LF byte
- 시작/끝의 추가 LF 없음
- section은 기존 model-visible section만 허용
- direction은 `ADD` 또는 `REMOVE`
- line 내부 CR/LF와 추가 field는 거부

### Split algorithm

- full text를 strict UTF-8 bytes로 인코딩
- offset 0부터 최대 8,192 bytes를 greedy하게 선택
- 경계가 UTF-8 continuation byte면 code-point 시작 전까지 backtrack
- row나 LF 경계를 별도 의미 단위로 취급하지 않음
- 빈 text는 chunk 0개
- nonempty chunk는 1~8,192 bytes
- 최대 8 chunks, 즉 9번째 chunk가 필요하면 fail closed
- chunk text를 index 순서로 UTF-8 재인코딩해 연결한 결과가 full text와
  byte-for-byte 같아야 함
- truncation과 summarization 없음

Model-visible chunk payload field는 다음 세 개뿐이다.

```text
normalized_text_delta_chunk
chunk_index
chunk_count
```

Full-text SHA-256, chunk SHA-256, byte count와 reconstruction receipt는
audit-only다. event/proposal/commit/path/blob ID와 hash는 model chunk에
넣지 않는다.

Migration restoration의 exact commit/date/mechanism을 나타내는
`quarantine_reason`과 receipt hash도 audit-only다. Model-facing
restoration payload는 generic `administrative_quarantined`,
`model_visibility`, 빈 `normalized_text_delta_chunks`만 가진다.

### Synthetic 결과

| case | bytes | chunks | 결과 |
|---|---:|---:|---|
| empty | 0 | 0 | pass |
| exact bound | 8,192 | 1 | pass |
| bound + 1 | 8,193 | 2 | pass |
| UTF-8 boundary | 8,195 | 2 | pass |
| LF boundary | 8,193 | 2 | pass |
| single oversized row | 20,000 | 3 | pass |
| D5 historical maximum control | 58,416 | 8 | pass |
| exact eight chunks | 65,536 | 8 | pass |
| ninth chunk | 65,537 | - | fail closed |

삭제, 중복, 순서 교환, byte 변경, index/count 위조, extra model field,
동일 text의 non-greedy repartition, 다른 full text의 9개 tamper
control도 모두 거부했다.

## Access boundary

Probe artifact는 다음을 고정한다.

- D5 census artifact read: true
- D5 forensic root access: false
- D5 official run invocation: false
- historical proposal text access: false
- external network access: false
- market/model/outcome access: false
- official raw text publication: false
- synthetic only: true

Probe 결과에는 synthetic full text나 chunk text를 게시하지 않고
case별 byte/chunk count와 hash receipt만 기록한다.

## 기각한 방식

- 모든 administrative→valid transition 일반 허용
- proposal 하나만 맞으면 receipt roster 검증 생략
- semantic/문장/row-aware 임의 chunking
- 8 KiB 이후 절단
- LLM summarization으로 원문 delta 대체
- 9번째 chunk를 조용히 삭제
- chunk-level model 결과 aggregation을 지금 결정
- D6 runner 또는 official source 실행을 probe와 동시에 구현

특히 chunk별 model output aggregation은 source representation과 다른
설계 문제다. 모델 단계를 열기 전 별도 preregistration에서 결정해야
한다.

## 재현 artifact

- probe:
  `results/protocol_specification_intent_maturity_d6_mechanism_probe_2026-07-26.json`
- probe SHA-256:
  `01b09218d71d83c6abc3c4225b708a1cae6fe9e426b9bbd98f4fe6e86579d60b`
- probe result hash:
  `dda4b4786b34064a104178580f6cd33e56d5616c282515f6579105231b5dab38`
- producer:
  `training/probe_protocol_specification_intent_maturity_d6_mechanism.py`
- regression:
  `tests/test_probe_protocol_specification_intent_maturity_d6_mechanism.py`

다음 단위는 이 mechanism과 authority를 변경 불가능하게 고정하는 D6
preregistration 준비성 리뷰다. 아직 official source 실행으로
넘어가지 않는다.
