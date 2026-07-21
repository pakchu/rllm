# BPAX-120 사전등록: SEC Bitcoin Product-Access Breadth

## 현재 판정

`BPAX-120`은 아직 알파가 아니다. 이 커밋은 **SEC 본문, BTC 가격, funding,
수익률, PnL 및 2024년 이후 자료를 한 행도 열지 않은 상태**에서 의미 분류와
집행 규칙을 고정한 사전등록이다.

비교 clock 13개와 mechanism control 3개는 파일 무결성 고정을 위해 raw byte
SHA-256만 계산했다. 2024년 이후를 포함할 수 있는 파일도 행/컬럼을 파싱하거나
시각·side·성과를 읽지 않았으며, artifact에 hash-only read와 parsed row `0`을
별도로 기록했다.

- 허용된 다음 단계: 24개 합성 문장에 대한 Gemma 4 E2B 의미·VRAM 게이트
- 아직 금지: SEC 본문 수집, 과거 의미 라벨 생성, novelty 계산, 백테스트,
  2024년 이후 eval
- 실패 시: 프롬프트·모델·threshold·hold를 고치지 않고 이 singleton을 폐기

## 가설

개별 기업의 Bitcoin 보유량이 아니라, 여러 발행사가 고객/기관/merchant에게
제공하는 **실사용 Bitcoin 접근 채널**이 동시에 열리거나 닫히는 사건을 센다.

- `BTC_ACCESS_EXPANSION`: live/completed 상태로 고객이 Bitcoin을 매매·보관·
  입출금·이체·결제·settlement할 수 있게 됨
- `BTC_ACCESS_RETRACTION`: 해당 접근이 정지·종료·상장폐지·규제 중단·운영
  장애로 실제 사라짐
- 계획, MOU, pilot, 제3자 서비스, 혼합 방향, 내부 도구, 채굴 장비 및 기업
  treasury 매수/매도/담보/보유는 모두 `UNSUPPORTED`

LLM은 `{class, quote}` 두 필드의 사실 추출만 수행한다. 방향, 기간, 노출,
threshold 및 수익 판단은 LLM에 맡기지 않는다.

## EBCT와 다른 점과 중복 위험

이전 `EBCT-72`는 기업 자기자본의 Bitcoin liquidity draw/buffer 간 **상태
전환**이었다. BPAX는 고객의 상품 접근이라는 다른 의미 객체를 사용하고,
이전 상태가 필요 없는 filing event를 14일 rolling breadth로 합산한다.

다만 같은 SEC 8-K/6-K source와 cross-issuer breadth를 쓰므로 family 중복 위험은
높다. 이를 숨기지 않고 다음을 강제했다.

1. EBCT의 모든 balance-sheet role은 BPAX에서 `UNSUPPORTED`여야 한다.
2. IRH-36, refined-product 계열을 해시 고정한 mechanism negative control로 둔다.
3. 가장 가까운 UCBR-12를 포함해 PSR, PCBR, OPDR, CLD, SQFD, SDDR 및 기존
   semantic/network/FX/live clocks 총 13개와 진입 Jaccard, ±24시간 coverage,
   signed 5분 exposure correlation을 비교한다.
4. Jaccard `<=0.10`, coverage `<=0.35`, 절대 correlation `<=0.35`를 하나라도
   넘으면 수익률을 열지 않고 폐기한다.

## 고정 event와 집행 규칙

1. 같은 accession의 여러 window가 오직 한 directional class일 때만 채택한다.
2. co-filer는 가장 작은 numeric CIK 하나로 귀속한다.
3. 한 issuer는 마지막 채택 filing부터 30 calendar day cooldown을 둔다.
4. historical ready time은 `acceptanceDateTime + 60분`이다. live에서는 이 값과
   durable receipt + parsing + redaction + 완료된 inference 중 늦은 시각을 쓴다.
5. 최근 14 calendar day distinct issuer score를
   `expansion - retraction`으로 계산한다.
6. 최소 4개 issuer가 있을 때 eligible state가 중립에서 `>=+3` 또는 `<=-3`으로
   바뀐 새 filing에서만 각각 long/short 신호를 낸다. expiry만으로는 신호를
   만들지 않는다.
7. ready + 5분 뒤 첫 BTCUSDT perpetual 5분봉 open에 진입하고 120시간 보유,
   0.5x, stop/TP 없음, global non-overlap을 적용한다.
8. base cost 6 bp/side, stress 10 bp/side, exact funding 및 held 5분 OHLC 전체를
   포함한 strict MDD를 사용한다. 절대수익률을 CAGR/MDD와 항상 함께 표시한다.

## Gemma 4 E2B 고정

- 모델: `google/gemma-4-E2B-it`
- revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- 원본 weight SHA-256:
  `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`
- runtime: Transformers `5.7.0.dev0` at
  `5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb`, bitsandbytes `0.49.2`,
  Accelerate `1.12.0`, Torch `2.9.0`
- inference: NF4 4-bit double quant, FP16 compute, one GPU, batch 1,
  `enable_thinking=false`, greedy decode

Google의 현재 모델 카드상 E2B는 2.3B effective/5.1B including embeddings,
35 layers, 128K context의 dense 모델이다. 합성 전체 실행에서 peak allocated는
7 GiB 이하, peak reserved는 7.25 GiB 이하여야 한다. 연구 GPU 통과는 3060 Ti
8GB 통과를 의미하지 않으며, live 전 실제 target smoke가 별도로 필요하다.

- [Gemma 4 E2B official model card](https://huggingface.co/google/gemma-4-E2B-it)
- [Google AI Edge Gemma 4](https://developers.google.com/edge/litert-lm/models/gemma-4)
- [Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

## 합성 및 이후 gate

합성 gate는 24/24 class 일치, 모든 model output parse/quote 일치, 양방향 prompt
injection guard 2/2, entity/product/date/amount swap invariance 및 위 VRAM 상한을
모두 요구한다.

그 이후에도 train 2021–2022와 selection 2023에서 source support, issuer/month
concentration, 양방향 거래 수, novelty가 먼저 통과해야 경제 평가를 열 수 있다.
경제 gate는 각 split의 절대수익 양수, `CAGR/strict MDD >= 3`, strict MDD
`<=15%`, 10 bp stress 수익 양수, 2023 양 반기 양수와 finite controls 대비
margin을 요구한다. 2024년 이후는 그때까지 계속 봉인한다.

## 고정 artifact

- 사전등록:
  `results/sec_edgar_bitcoin_product_access_preregistration_2026-07-22.json`
- artifact SHA-256:
  `ab975eea454fbe1a784adaee979c5ad6162be9b18363c7fe3aa47959e075b883`
- contract hash:
  `124d37318a0e2bd60f81e81f76484fd3f3e356168bee71b137603b65f0ddb7ff`
- manifest hash:
  `67cac2d57942c96bc243ebc45be5b6720f13e6385dd731af94f85f32997ca770`
- synthetic cases hash:
  `d258922f785479758c52827b8837053ac259e881ff13f134294e65742aeaa6a0`

공개 filing이 Gemma 사전학습에 포함됐을 수 있으므로 clean-room 또는 zero
memorization을 주장하지 않는다. identity/date/number redaction과 quote grounding은
위험 완화일 뿐 제거가 아니다.
