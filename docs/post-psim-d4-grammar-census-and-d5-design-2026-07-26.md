# PSIM-D4 사후 문법 전수조사와 D5 설계 방향

## 결론

PSIM-D4의 실패는 단일 역사적 예외가 아니었다. 공식 실행에서 이미
수화된 Ethereum proposal blob 5,206개를 결과·시장·모델 접근 없이
전수 분류한 결과:

- D4 strict parser 통과: 4,440개 (`85.2862%`)
- 2023-10-25 ERC 저장소 이동용 exact redirect: 730개
- 실제 비정상 metadata: 36개

D4 실패 766개 중 730개(`95.3003%`)는 specification intent가 아니라
저장소 분리로 발생한 관리성 이동 문서였다. 따라서 다음 후보는
예외를 하나씩 parser에 추가해서는 안 된다. D5는 proposal **경로
identity와 causal text diff를 기본 의미**로 사용하고, exact migration
redirect를 격리하며, 유효하지 않은 metadata를 명시적 상태로
보존해야 한다.

이 문서는 source grammar 설계만 다룬다. candidate 선택, 시장 데이터,
모델, reward, 거래, PnL, CAGR, strict MDD 또는 outcome은 열지 않았다.

## 고정된 조사 경계

| 항목 | 값 |
|---|---|
| D4 terminal commit | `e406778f76f252d3b0cddb33242edcb51e984c80` |
| D4 terminal JSON SHA-256 | `4d947075c0f54c5cd09c732710da0502c87d89fa52029fe81367dd3f27ab2aaf` |
| D4 terminal result hash | `8563ef3ace444896295d7076cd0f839e8f62f89899e312d711f9768f5cbf84aa` |
| forensic object-store hash, before/after | `fd0ac6636ab7a954e46deb82188f9963f135b1b92152785c6f50205995766a2a` |
| commit-chain rows/hash | `6,958` / `c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9` |
| proposal groups/hash | `4,985` / `a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c` |
| hydrated blob OID count/manifest SHA-256 | `5,206` / `8aa47dbe594df92a42ce87f6872f2bb3548f5370371f7668b26c80a47c53c944` |
| census network commands | `0` |
| object-store mutation | 없음 |
| D4 evaluator invocation | 없음 |

조사는 `/tmp/psim-d4-source/ethereum-a.git`의 이미 수화된 object만
`git cat-file --batch`로 읽었다. D4 terminal artifact와 object-store
snapshot이 사전 고정값과 다르면 즉시 실패하며, 조사 전후 snapshot
동일성도 강제한다. Git wrapper는 `network=True` 표시에 의존하지 않고
`cat-file`, `diff-tree`, `for-each-ref`, `ls-tree`, `rev-list`,
`rev-parse`, `symbolic-ref`, `verify-pack`만 허용한다. `fetch`, `clone`,
`ls-remote`, `remote`, `pull`, `push`, `submodule` 및 알 수 없는 global
option은 argv 단계에서 fail closed한다.

## 전수조사 결과

| 분류 | blob | proposal | 해석 |
|---|---:|---:|---|
| `D4_VALID` | 4,440 | 701 | D4 strict parser 통과 |
| `ERC_MIGRATION_REDIRECT_LOWER_PATH` | 365 | 365 | `ercs/erc-N.md` 이동 stub |
| `ERC_MIGRATION_REDIRECT_UPPER_PATH` | 365 | 365 | `ERCS/erc-N.md` 이동 stub |
| `D4_DUPLICATE_IDENTICAL_HEADER` | 7 | 2 | 동일한 `status: Draft` 중복 |
| `D4_MALFORMED_HEADER_LINE` | 20 | 4 | 역사적 pseudo-field 표기 |
| `D4_SELF_DEPENDENCY` | 9 | 1 | EIP-3779의 self-reference 포함 |

proposal 수는 class별 unique count이며 class 사이에서 배타적인 전체
proposal roster를 의미하지 않는다.

### 관리성 migration redirect

두 redirect class의 730개 blob은 모두 `2023-10-25`에 발생했고, 각
stub의 target 번호는 proposal path 번호와 일치했다. lower-case path
365개 뒤 같은 날 upper-case path 365개로 교정되었다. 365 proposal
roster의 canonical hash는
`c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd`
이다.

이는 공식 이력과 일치한다.

- Ethereum EIPs의 ERC 분리 commit
  [`0f44e2b`](https://github.com/ethereum/EIPs/commit/0f44e2b94df4e504bb7b912f56ebd712db2ad396)
  은 2023-10-25에 기존 문서를 이동 stub으로 바꿨다.
- 같은 날 후속 commit
  [`47ce702`](https://github.com/ethereum/EIPs/commit/47ce70257fae525a427780630bd8d1903cc96e75)
  은 target path의 대소문자를 교정했다.
- 분리된 ERC 저장소의 대응 commit은
  [`8dd085d`](https://github.com/ethereum/ERCs/commit/8dd085d159cb123f545c272c0d871a5339550e79)
  이다.
- 공식 migration 추적은 Ethereum ERCs
  [issue #1](https://github.com/ethereum/ERCs/issues/1)과
  [issue #8](https://github.com/ethereum/ERCs/issues/8)에 남아 있다.

따라서 이 730개 text change를 protocol intent로 모델에 노출하는
것은 잘못이다.

### 비정상 metadata 36개

중복 header:

- proposal: `2544`, `3102`
- 7개 blob 모두 동일한 `status: Draft`가 두 번 등장
- 최초 관측: EIP-2544, commit
  `bd912a490d97da82a73313facf4458bbaa0dab2b`

malformed header:

- proposal: `2515`, `2615`, `2711`, `2718`
- `requires (*optional): 155`: 8개
- `requires (*optional): 165 721`: 1개
- `requires (*optional): 2718`: 10개
- 빈 `requires (*optional):`와 `replaces (*optional):`: 1개

self dependency:

- proposal: `3779`
- `requires: 2315, 3540, 3670, 3779, 4200`: 9개

현재 EIP-1은 ordered front matter와 field 형식을 명시한다
([EIP-1 source](https://raw.githubusercontent.com/ethereum/EIPs/master/EIPS/eip-1.md),
[rendered EIP-1](https://eips.ethereum.org/EIPS/eip-1)).
YAML 1.2.2는 mapping key uniqueness를 요구하고
([YAML 1.2.2](https://yaml.org/spec/1.2.2/)), Jekyll도 front matter가
valid YAML이어야 한다고 명시한다
([Jekyll front matter](https://jekyllrb.com/docs/front-matter/)).

공식 Ethereum 문서에서 duplicate key나 malformed historical
metadata를 `first wins`, `last wins`, merge 또는 자동 교정하는 규칙은
찾지 못했다. 그러므로 임의 resolution은 source fact가 아니라 구현
편의에 의한 합성 정보가 된다.

버전 주의:

- migration 근거는 2023-10-25의 공식 issue/commit 이력이다.
- 조사 시 확인한 EIP-1 source의 최신 변경 근거는 2026-02-17 commit
  [`ac85644`](https://github.com/ethereum/EIPs/commit/ac856441732539da4554e1ddb4e445f3f02be65c)이다.
- YAML 기준은 2021-10-01의 1.2.2 specification이다.
- Jekyll 문서는 조사 시점 사이트 표기 `v4.4.1` 기준이다.

## D5에 요구되는 의미론

아래는 outcome-blind source contract 요구사항이다. 아직 candidate
선택이나 성과 주장이 아니다.

1. **Path identity가 proposal identity의 기준이다.**
   역사적 invalid header의 `eip` 값을 고쳐 쓰거나 path와 합성하지
   않는다.
2. **Causal text diff는 metadata parse 성공과 독립적이다.**
   invalid metadata가 있어도 해당 시점까지 이용 가능한 raw text
   변화 자체는 보존한다.
3. **Exact administrative migration은 격리한다.**
   한 줄 전체가 고정된 Ethereum ERC 이동 형식과 일치하고 target
   proposal이 path proposal과 같을 때만 administrative redirect로
   분류한다. 부가 text, target mismatch, 다른 host/path는 격리하지
   않고 fail closed한다.
4. **Metadata validity를 명시적 categorical state로 보존한다.**
   최소 상태는 `VALID`, `DUPLICATE_IDENTICAL`,
   `DUPLICATE_CONFLICTING`, `MALFORMED`, `SELF_DEPENDENCY`,
   `ADMINISTRATIVE_REDIRECT`, `UNKNOWN_INVALID`이다.
5. **유효하지 않은 dependency를 추론하거나 수리하지 않는다.**
   dependency feature는 `UNKNOWN/UNAVAILABLE`로 기록한다. 중복 제거,
   self-edge 삭제, 숫자 사이 comma 삽입, pseudo-field rename은 금지다.
6. **미분류 문법은 fail closed한다.**
   전수조사에서 관측하지 못한 parser failure를 자동 통과시키지 않는다.
7. **관리성 사건은 model-visible intent에서 제외한다.**
   단, audit trail에는 원문 hash, commit, path, effective day,
   quarantine reason을 남긴다.

## 기각한 대안

- 첫 오류만 고친 뒤 공식 실행을 반복하는 방식
- duplicate field의 first/last value 선택
- identical duplicate를 자동 deduplicate
- self dependency 자동 제거
- malformed pseudo-field를 정상 field로 rename
- migration stub을 protocol specification text로 학습
- invalid metadata가 있는 blob/event 전체 삭제

앞의 다섯 방식은 source에 없는 의미를 만든다. migration 학습은
관리 작업을 intent로 오인한다. 전체 삭제는 실제 causal text 변화와
invalidity 상태를 동시에 잃는다.

## 재현 artifact

- census:
  `results/protocol_specification_intent_maturity_d4_grammar_census_2026-07-26.json`
- census SHA-256:
  `eaa5946844bb218b1ae211c84d509c49482111af6ee165bbb54fbad26ff3b77f`
- census result hash:
  `85ff0c04a1fe06b34b7f214f5fd7b9a1191a4ef0dd990a7e7d002f72efe9428d`
- evaluator:
  `training/audit_protocol_specification_intent_maturity_d4_grammar_census.py`
- regression:
  `tests/test_audit_protocol_specification_intent_maturity_d4_grammar_census.py`

다음 단위는 이 요구사항을 synthetic-only probe로 구체화한다. probe가
exact redirect quarantine, path identity, raw text preservation,
invalid metadata state, dependency unknown, fail-closed behavior를 모두
증명한 뒤에만 D5 preregistration으로 넘어간다.
