# PSIM-D5 사후 이벤트 의미론 전수조사와 D6 요구사항

## 결론

PSIM-D5 Gate 4 실패는 blob parser 문제가 아니었다. 공식 실행에서
이미 수화된 Ethereum proposal blob 5,206개와 proposal group
4,985개를 시장·모델·outcome 접근 없이 전수 검사한 결과:

- blob decode 성공: **5,206 / 5,206**
- event semantics 성공: **4,430 / 4,985**
- event semantics 실패: **555 / 4,985**
- 미등록 blob grammar: **0**

실패 원인은 정확히 두 종류였다.

1. model-visible normalized text delta가 고정된 event당 8,192 bytes를
   초과: **190건**
2. exact ERC migration redirect에서 정상 문서로 복원되는 전이를
   `reverse administrative migration`으로 거부: **365건**

따라서 D5의 path identity와 explicit invalid-metadata decoder는
역사 자료 전체를 디코딩하는 데 성공했다. 실패한 부분은 그 위의
**event representation contract가 실제 source history 전체를
표현하지 못한 것**이다.

이 조사는 source semantics만 다룬다. 후보 선택, 시장 데이터, 모델,
reward, 거래, PnL, CAGR, strict MDD 또는 outcome은 열지 않았다.
PSIM-D5를 다시 실행하지 않았고 `/tmp/psim-d5-source`를 수정하거나
candidate로 승인하지 않았다.

## 고정된 조사 경계

| 항목 | 값 |
|---|---|
| D5 terminal commit | `0f69f7472d89474052186bbb2b13fa8d6bf5d77f` |
| D5 terminal JSON SHA-256 | `ffdebf2e5107f08345f16e21adc895d3bfc2f236d6b231322d03c372d4764ca1` |
| D5 terminal result hash | `0a23218e8784599f09e092d4f93942a48111c0af4f8e3ff85e2183eb84f56c56` |
| D5 runner commit / SHA-256 | `90e7740edcd68a3b4c3acf8e9fe9a14f9e4eb8e1` / `744959177c1f18d62cb920f5bd9c1068eb5415c07d4f7d5719af5b37542e0dba` |
| D5 semantics commit / SHA-256 | `0e62ec05e6861b2619e6737dd594e7306ad7c93a` / `d1aaf55effec3df8f38854992b4c60bd39d612e4bd6cd00fe705f60b5cac9d85` |
| forensic object-store hash, before/after | `d449f80fd3a6d2f1993e01c6418294d5385381084a9c2b893179bca368bae34a` |
| commit-chain rows / hash | `6,958` / `c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9` |
| proposal groups / hash | `4,985` / `a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c` |
| hydrated blob OID count / manifest SHA-256 | `5,206` / `8aa47dbe594df92a42ce87f6872f2bb3548f5370371f7668b26c80a47c53c944` |
| census network commands | `0` |
| object-store mutation | 없음 |
| D5 evaluator invocation | 없음 |

조사는 `/tmp/psim-d5-source/ethereum-a.git`의 이미 수화된 object만
읽었다. Git wrapper는 `cat-file`, `diff-tree`, `for-each-ref`,
`ls-tree`, `rev-list`, `rev-parse`, `symbolic-ref`, `verify-pack`만
허용했다. `fetch`, `clone`, `ls-remote`, `remote`, `pull`, `push`,
`submodule`과 알 수 없는 global option은 argv 단계에서 fail
closed한다. 조사 전후 object-store snapshot은 동일했다.
감사 artifact 출력은 검사를 시작하기 전에 repo-local flat
`results/*.json` 상대 경로로 제한한다. forensic root, 모든 absolute path,
nested/non-JSON result와 symlink output은 모두 거부한다.

## 전체 결과

### Blob decoder

| 분류 | blob |
|---|---:|
| `D4_VALID` | 4,440 |
| `ERC_MIGRATION_REDIRECT_LOWER_PATH` | 365 |
| `ERC_MIGRATION_REDIRECT_UPPER_PATH` | 365 |
| `D4_MALFORMED_HEADER_LINE` | 20 |
| `D4_SELF_DEPENDENCY` | 9 |
| `D4_DUPLICATE_IDENTICAL_HEADER` | 7 |

D4 사후 grammar census와 정확히 일치한다. D5 decoder는 exact
administrative redirect와 기존 invalid metadata를 합성 수리하지
않고 모두 명시적 상태로 표현했다.

### Event semantics

| outcome | group |
|---|---:|
| `PASS_MODEL_VISIBLE` | 3,700 |
| `PASS_ADMINISTRATIVE_QUARANTINE` | 730 |
| `ERROR_MODEL_TEXT_BOUND` | 190 |
| `ERROR_REVERSE_ADMINISTRATIVE_MIGRATION` | 365 |
| 합계 | 4,985 |

전체 event ID와 outcome pair의 canonical roster hash는
`0d89d91530566087e50ac55fbad585b5eafc0ddad01382179a03571d4314c3ad`
이다. 따라서 위 합계는 첫 예외까지의 부분 실행이 아니라 전체 group
전수 결과다.

## 실패 1: whole-event text bound

`normalized_text_delta` 전체가 event당 8,192 bytes를 넘으면 D5는
event를 거부했다.

- 실패 event: **190**
- unique proposal: **136**
- `CREATE`: 103
- `UPDATE`: 87
- 최초 effective day: 2020-02-12
- 최종 effective day: 2023-10-17
- 실패 text 최소: 8,339 bytes
- 실패 text 최대: 58,416 bytes
- 실패 text 합계: 2,678,077 bytes
- 실패 model row 최소/최대: 18 / 943
- 실패 model row 합계: 30,580
- 통과 event 최대: 8,153 bytes
- 실패 event roster hash:
  `0f299221248e66ca1eddc9cdd839cab504755537e47464c697c481544d169fd4`
- proposal roster hash:
  `a6765fbf7ae8cf6fe7b1341a03fe3385841cef9d7f28199c8bdcbf1f4e6ee9d6`

190건은 모두 absent→`D4_VALID` CREATE 또는
`D4_VALID`→`D4_VALID` UPDATE다. invalid metadata나 migration
예외가 아니다. 즉, 8 KiB 경계는 잘못된 source를 차단한 것이 아니라
정상적인 큰 specification change를 표현하지 못했다.

원문은 결과 artifact에 넣지 않았다. 감사 결과에는 byte/line count
집계, byte+line event-size roster hash, event roster hash, blob
SHA-256과 대표 metadata만 기록했다.

## 실패 2: administrative restoration

2023-10-25의 ERC repository migration은 D5가 730개 전이를 exact
administrative quarantine으로 처리했다. 그러나 다음 날인
2023-10-26, 같은 365 proposal이 upper-case redirect stub에서
정상 EIP 문서로 복원되는 전이는 모두 거부됐다.

- 실패 event: **365**
- unique proposal: **365**
- effective day: 모두 2023-10-26
- event type: 모두 `UPDATE`
- old state: 모두 `ADMINISTRATIVE_REDIRECT`
- old class: 모두 `ERC_MIGRATION_REDIRECT_UPPER_PATH`
- new state: 모두 `VALID`
- new class: 모두 `D4_VALID`
- proposal roster hash:
  `c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd`
- failure event roster hash:
  `2bc4fec07ac144245a48d44e0d24f5425705eeb81d41b4268bfe4367eee54a1a`
- 3-step episode roster hash:
  `7065c33783f1ea54af1522da7e442ec05507c38355bb98ed90daf3f87e89b0bd`
- proposal별 episode receipt: **365개**

각 proposal에 대해 직전 두 event와 실패 event를 결박해 다음 sequence
전체를 확인했다.

| step | commit | day | class transition | D5 outcome |
|---:|---|---|---|---|
| 1 | `0f44e2b94df4e504bb7b912f56ebd712db2ad396` | 2023-10-25 | `D4_VALID` → lower redirect | quarantine pass |
| 2 | `47ce70257fae525a427780630bd8d1903cc96e75` | 2023-10-25 | lower redirect → upper redirect | quarantine pass |
| 3 | `25cdf1d059778236e28bf22d752ca48a35af91f6` | 2023-10-26 | upper redirect → `D4_VALID` | reverse error |

365개 episode 모두 `UPDATE`이며 모든 old/new path가 각
`EIPS/eip-N.md`로 유지됐다. lower/upper redirect target도 모두 path의
proposal `N`과 일치했다. 730개 administrative-quarantine pass event와
365개 reverse error event가 이 episode receipt에 각각 정확히 한 번
포함된다.

proposal roster hash는 D4 census에서 고정한 365개 migration proposal
roster와 동일하다. 따라서 이는 일반적인 administrative→valid
transition을 임의 허용해야 한다는 뜻이 아니다. proposal별 path,
redirect target, blob hash, commit/day 순서로 증명된 **exact migration
episode의 복원 단계**를 D5 contract가 누락했다는 뜻이다.

## D6의 outcome-blind 필수 조건

아직 D6 candidate나 성능 후보를 승인하지 않는다. 다음 단계는
synthetic-only mechanism probe이며 아래 조건을 먼저 증명해야 한다.

### 1. Exact migration episode

- 단일 blob의 class만 보고 일반 예외를 허용하지 않는다.
- proposal path identity, old/new blob class, commit/effective-day 순서,
  고정된 migration proposal roster를 causal episode로 결박한다.
- valid→lower redirect→upper redirect→valid의 exact episode 전체를
  audit에는 보존하고 model-visible intent에서는 격리한다.
- target mismatch, 부가 text, 다른 날짜·순서·path·proposal roster는
  격리하지 않고 fail closed한다.

### 2. Deterministic bounded chunks

다음 synthetic probe가 구현할 splitter 후보는 아래처럼 완전히
결정적이어야 한다. 이 문서만으로 D6 candidate를 승인하지는 않는다.

1. D5와 동일한 causal model row 순서를 사용한다. 각 row를 UTF-8
   `section|direction|line`으로 직렬화하고 row 사이에 단일 LF byte
   (`0x0a`)를 넣는다. 첫/마지막에 별도 LF는 붙이지 않는다.
2. 이 전체 byte string의 SHA-256, byte count, row count를 audit-only
   full-diff receipt에 기록한다.
3. offset 0부터 최대 8,192 bytes의 연속 구간을 greedy하게 자른다.
   경계가 UTF-8 continuation byte를 가리키면 유효한 code-point
   boundary가 될 때까지 end offset을 뒤로 이동한다. 따라서 separator
   byte와 단일 row가 경계를 넘는 경우도 별도 예외 없이 같은 규칙을
   따른다.
4. 빈 full text는 chunk 0개다. 빈 값이 아닌 각 chunk는 1~8,192
   bytes이고, chunk UTF-8 bytes를 index 순서로 이어 붙이면 full text와
   byte-for-byte 같아야 한다.
5. 이 source interval의 관측 최대 58,416 bytes를 포함하도록
   `MAX_CHUNKS_PER_EVENT=8`을 고정한다. 65,536 bytes를 넘겨 9번째
   chunk가 필요한 event는 truncation하지 않고 fail closed한다.
6. audit-only manifest는 full-text hash, ordered chunk SHA-256/byte
   counts, chunk count와 reconstruction verdict를 가진다. model-visible
   chunk payload는 기존 categorical event payload와
   `normalized_text_delta_chunk`, 0-based `chunk_index`,
   `chunk_count`만 가진다. event/proposal/commit/path/blob ID와 hash는
   추가하지 않는다.
7. 각 model invocation은 chunk 하나만 받고 `chunk_index` 오름차순으로
   실행된다. chunk별 model 결과를 합치는 규칙은 source probe 범위가
   아니며, 시장·모델 단계를 열기 전에 별도 preregistration으로
   고정해야 한다.

Synthetic battery는 최소한 empty, 정확히 8,192 bytes, 8,193 bytes,
UTF-8 multi-byte boundary, LF boundary, 단일 oversized row, 정확히
65,536 bytes, 65,537 bytes를 포함한다. 또한 chunk 삭제·중복·순서
교환·byte 변경·index/count 위조와 full hash 불일치를 모두 검출해야
한다. truncation, 임의 summarization, token-budget 기반 내용 삭제는
금지한다.

### 3. Gate 4 전수성

- 모든 blob decode receipt를 event loop 전에 확정한다.
- 모든 group의 outcome을 집계하고 합계가 고정 group count와 일치해야
  한다.
- 첫 exception에서 중단하지 말고 unknown class/error roster를
  terminal rejection artifact에 남긴 뒤 reject한다.
- 이 synthetic probe와 별도 설계 리뷰를 통과한 뒤에만 D6
  preregistration을 만들 수 있다.

현재 D5 사후 census는 이미 종료된 forensic root의 정확한 재현을 위해
관측 count를 상수로 결박하고, root가 달라지면 artifact를 쓰지 않고
실패한다. D6 official runner는 이 post-terminal seal 패턴을 그대로
복사하면 안 된다. D6에서는 unknown roster를 끝까지 수집하고 canonical
rejection artifact를 먼저 게시하는 별도 terminal path가 필요하다.

## 기각한 대안

- D5 공식 evaluator 반복 실행
- 관측된 첫 365/190건만 proposal ID allowlist로 통과
- 모든 administrative→valid transition 일반 허용
- 8 KiB 이후 text 절단
- LLM summary로 원문 delta 대체
- 큰 event 삭제 또는 sampling
- `/tmp/psim-d5-source`를 수리해 candidate로 재사용

앞의 방식들은 source 의미를 임의 변경하거나 사후 관측에 맞춘 예외를
만든다. D6는 outcome과 무관하게 source history 전체를 총체적으로
표현하는 mechanism이어야 한다.

## 재현 artifact

- census:
  `results/protocol_specification_intent_maturity_d5_event_semantics_census_2026-07-26.json`
- census SHA-256:
  `df2bcbef28b22d6daeb258d5c0f36b918d833b9c5fb0e5c9229a44edce4c2d59`
- census result hash:
  `0ca0a11f6693543dafbcf29052f2e963bf721c6e12f71f6fc9fbb1856e2dfe4a`
- evaluator:
  `training/audit_protocol_specification_intent_maturity_d5_event_semantics_census.py`
- regression:
  `tests/test_audit_protocol_specification_intent_maturity_d5_event_semantics_census.py`

다음 작업 단위는 위 두 mechanism을 synthetic battery로 먼저
검증하는 D6 설계·probe다. 아직 시장·모델·outcome 단계로 넘어가지
않는다.
