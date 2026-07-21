# BPAX-120 합성 게이트 폐기 판정

## 결론

`BPAX-120`은 **알파 후보로 승격하지 않는다**. Gemma 4 E2B가 메모리 상한은
통과했지만, 사전등록된 24개 의미 사례를 17개만 정확히 판정했고 model output
22개 중 17개만 strict parse/quote 검증을 통과했다.

사전등록의 no-repair 규칙에 따라 label 정규화, prompt 변경, few-shot 추가,
모델 교체 또는 사례별 예외 규칙을 적용하지 않는다. SEC 본문과 BTC 성과를 열지
않고 exact singleton을 폐기한다.

## 실행 결과

| 항목 | 결과 | gate |
|---|---:|---:|
| 전체 합성 사례 | 24 | 24 |
| 실제 모델 호출 | 22 | 22 |
| guard 무호출 | 2/2 | 2/2 |
| 정확 class | 17/24 | 24/24 |
| model parse | 17/22 | 22/22 |
| model quote 검증 | 17/22 | 22/22 |
| swap invariance | 통과 | 통과 |
| peak allocated | 6,856,160,256 bytes (약 6.39 GiB) | <= 7 GiB |
| peak reserved | 6,893,338,624 bytes (약 6.42 GiB) | <= 7.25 GiB |

- GPU: NVIDIA GeForce RTX 5090, visible device 1개
- model load: 10.31초
- 22회 inference 합계: 47.76초
- 모델/revision: `google/gemma-4-E2B-it` /
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- runner는 모델 호출 전 commit `98629e5`로 고정됨

메모리만 보면 동일한 batch/context의 8GB GPU에서 가능성이 있지만, 이는 실제
RTX 3060 Ti smoke를 대신하지 않는다. 더 중요한 의미 gate가 실패했으므로 해당
모델의 BPAX live deployment는 금지 상태다.

## 실패 유형

### 1. retraction label 대소문자 위반 4건

다음 네 사례에서 의미 자체는 retraction을 골랐으나
`BTC_ACCESS_retraction`을 출력해 strict enum을 위반했다.

- `suspended_customer_trading`
- `terminated_client_custody`
- `delisted_retail_bitcoin`
- `regulatory_access_halt`

이것을 대문자 정규화하면 점수는 좋아지지만, 사전등록 이후 parser를 느슨하게
바꾸는 사후 repair이므로 허용하지 않는다.

### 2. planned pilot 오분류 1건

`mou_and_pilot_only`를 expansion으로 골랐고 evidence quote는 비워 두었다. class와
quote 계약을 동시에 위반해 parse failure로 처리됐다.

### 3. 의미 경계 실패 2건

- `third_party_access`: issuer가 아니라 unrelated exchange의 고객 접근을 expansion으로
  오분류했다.
- `mixed_access_direction`: custody expansion과 trading suspension이 함께 있는 문장에서
  expansion 한 조각만 골랐다. mixed evidence를 fail-closed하지 못했다.

이 두 건은 단순 문자열 casing 문제가 아니라 issuer attribution과 contradictory
evidence aggregation이라는 핵심 의미 경계 실패다. 따라서 parser 정규화만으로도
후보를 살릴 수 없다.

## 누수 경계

이번 실행에서 열린 것은 사전등록 artifact, 고정 모델 파일과 합성 문장뿐이다.

- SEC filing body: 0
- historical semantic row: 0
- BTC market row: 0
- funding row: 0
- future return/PnL field: 0
- 2024년 이후 SEC source row: 0

따라서 이 결과는 수익성 실패가 아니라 **모델 의미 추출 gate 실패**다. 경제적
성과나 실제 alpha 여부는 평가하지 않았다.

## 고정 artifact

- 결과:
  `results/sec_edgar_bitcoin_product_access_synthetic_gate_2026-07-22.json`
- 결과 SHA-256:
  `036af95ce032bdf9de2b10a742f457cdc09e6096b60616f5d5f5da5c4001e2c4`
- manifest hash:
  `f9c720f07da82b46d21f63310c3caceeb931efc2f17987f760d6e732709185e6`
- runner SHA-256:
  `111efe9dcd3a30520d97d530ab63835d9fc9fe9b559465ac2492a7a554962b18`
- 사전등록 SHA-256:
  `ab975eea454fbe1a784adaee979c5ad6162be9b18363c7fe3aa47959e075b883`

모델 사양과 사용법은 [공식 Gemma 4 E2B 모델 카드](https://huggingface.co/google/gemma-4-E2B-it)와
[Google AI Edge Gemma 4 문서](https://developers.google.com/edge/litert-lm/models/gemma-4)를
기준으로 고정했다.
