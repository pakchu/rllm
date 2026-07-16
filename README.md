# RLLM Trading Research

RLLM은 BTCUSDT 선물 트레이딩에서 **LLM/RL의 장점을 실제 수익성 검증과 라이브 실행까지 연결**하기 위한 연구/실험 레포입니다. 초기 목표는 이미지 기반 VLM+RL 트레이딩 봇이었지만, 여러 차례의 실패와 검증을 거치며 현재는 **텍스트/피쳐 기반 REX 후보 생성 + LLM/RLLM 게이트 + 엄격한 백테스트 + 안전한 실행 브릿지** 구조로 재정렬되었습니다.

> 이 레포는 투자 조언이 아닙니다. 모든 결과는 과거 데이터 기반 연구 결과이며, 실거래 전 별도 검증과 리스크 관리가 필요합니다.

---

## 현재 결론 요약

### 방향 전환
- **이미지 기반 VLM/RL 단독 접근은 현재 폐기/보류**했습니다.
  - 차트 이미지만으로 안정적인 수익성과 일반화를 확보하기 어려웠습니다.
  - 숫자/시장 구조 정보가 이미지에 압축되면서 중요한 미세 구조와 멀티타임프레임 정보가 손실됐습니다.
- 현재 중심은 **price action / rolling extrema / macro-premium context / derivatives context**를 텍스트 및 수치 피쳐로 구조화한 뒤, LLM이 후보를 해석하거나 gate하는 방식입니다.
- analyzer/trader 2-모델 구조도 과도하게 크고 불안정해 **단일 REX+RLLM gate 구조**로 축소했습니다.

### 현재 가장 쓸 만한 후보
- 후보군: `rex_htf_pullback_reclaim`
- 기본 정책: REX가 후보 방향을 만들고, LLM/RLLM gate가 `TRADE` 또는 `ABSTAIN`을 결정합니다.
- 현재 live pilot은 **short-only / bearish manual regime 전제**로 구성되어 있습니다.
- 긴 regime 판단은 자동화하지 않고, 사용자가 수동으로 `BEAR` regime을 열어주는 구조입니다.

### 최근 검증 요약
대표 결과 파일:
- `results/rex_dual_regime_tp4_eval_2025_2026h1_lev1.25_2026-07-03.json`
- `results/rex_dual_regime_tp4_eval_2025_2026h1_lev1.5_2026-07-03.json`
- `results/rex_dual_regime_tp4_all_2021_2026h1_lev1.0_2026-07-03.json`

최근 eval 구간 `2025-01-29 ~ 2026-05-20`:

| 레버리지 | CAGR | strict MDD | CAGR / strict MDD | 거래 수 |
|---:|---:|---:|---:|---:|
| 1.25x | 48.4% | 8.17% | 5.92 | 57 |
| 1.5x | 50.9% | 10.48% | 4.85 | 57 |

장기 `2021-01-02 ~ 2026-05-20`, 1.0x:

| CAGR | strict MDD | CAGR / strict MDD | 거래 수 |
|---:|---:|---:|---:|
| 26.9% | 25.19% | 1.07 | 482 |

해석:
- 최근 하락/약세 regime에서는 유의미한 short-biased edge가 보였습니다.
- 전 기간 universal edge는 아직 부족합니다.
- 따라서 현재 전략은 “항상 켜는 범용 봇”이 아니라 **수동 bearish regime pilot**으로 취급해야 합니다.

### 2026-07-16 신규 알파 포트폴리오 배분 (forward shadow)

현재 live gross 3.85 포트폴리오는 그대로 유지합니다. 새 알파를 포함한 배분은
train+2024에서만 순위를 정하고, 이미 연구 노출된 2025/2026은 고정 1위의 veto에만
사용했습니다. 39,040개 배분을 정확한 5분봉 same-BTC low/high strict MDD로 평가한
best-found shadow 후보는 다음과 같습니다.

- `fresh_kimchi_fx`: 2.00
- `frozen_annual_rank7`: 2.00
- `rex_taker_low_range_position`: 0.40
- `cand_rex_veto_7`: 1.60
- `markov_transition_long`: 2.00
- 총 gross: **8.00**

| 구간 | 절대수익 | CAGR | strict MDD | CAGR / MDD | 거래 수 |
|---|---:|---:|---:|---:|---:|
| Train (2020-09~2023) | 2,274.53% | 158.73% | 36.58% | 4.34 | 861 |
| Test 2024 | 180.81% | 180.22% | 16.05% | 11.23 | 203 |
| Eval 2025 (report/veto) | 148.35% | 148.51% | 12.35% | 12.03 | 133 |
| 2026 YTD (report/veto) | 69.24% | 251.14% | 15.00% | 16.74 | 108 |

이 수치는 강하지만 **pristine OOS가 아니며 live 승격 근거가 아닙니다**. 후보 universe와
미래 구간에 과거 연구 노출이 있으므로 정확한 동일 배분을 forward shadow로만 운영해
새 데이터에서 실행 parity와 MDD를 확인해야 합니다. 상세 규약과 상위 배분은
[`docs/portfolio-added-alpha-update-2026-07-16.md`](docs/portfolio-added-alpha-update-2026-07-16.md)에 있습니다.

현재 no-order DB one-shot에서는 완료봉 freshness, 90,000분/18,000봉 history contract,
4/5 sleeve signal scoring이 통과했습니다. scoreable 4개 sleeve는 동결 source와
전체 포트폴리오 평가 시작 구간에서 candidate-side decision hash mismatch 0을
확인했습니다. REX-taker는 `active_from=2021-01-01`을 실행 계약으로 고정해 이전
후보를 차단했습니다. 이는 체결/exit/PnL 패리티를 뜻하지 않습니다.
`frozen_annual_rank7`은 정확한
40-feature/state/threshold/exit bundle이 없어 fail-close하고 Fresh Kimchi TP/SL도 아직
portfolio lifecycle에 연결되지 않았습니다. 따라서 기존 live 설정은 그대로 유지합니다.
상세 readiness는
[`docs/portfolio-added-alpha-shadow-readiness-2026-07-16.md`](docs/portfolio-added-alpha-shadow-readiness-2026-07-16.md),
신호 감사는
[`docs/portfolio-added-alpha-shadow-signal-parity-2026-07-16.md`](docs/portfolio-added-alpha-shadow-signal-parity-2026-07-16.md)에 있습니다.

### 거래 빈도
최근 eval 후보 기준:

| 구간 | 거래 수 | 일평균 거래수 | 평균 간격 |
|---|---:|---:|---:|
| 2025-01-29 ~ 2026-05-20 | 57 | 0.120/day | 8.3일에 1회 |
| 최근 6개월 | 39 | 0.216/day | 4.6일에 1회 |
| 최근 3개월 | 12 | 0.135/day | 7.4일에 1회 |
| 2021 ~ 2026H1 | 482 | 0.246/day | 4.1일에 1회 |

저빈도 전략이며 매일 거래하는 봇이 아닙니다.

---

## 현재 아키텍처

```text
Live DB / historical CSV
        |
        v
preprocessing/live_db_features.py
- BTCUSDT 1m -> completed 5m bars
- Upbit KRW-BTC
- USDKRW
- synthetic DXY FX components
- Binance premium index
- Binance funding rate
        |
        v
preprocessing/market_features.py
- trend / range / RSI / MFI / volatility
- higher timeframe features
- rolling extrema(REX) features
- macro / kimchi / USDKRW features
- funding / premium features
        |
        v
training.event_candidate_pool_probe._feature_candidates
- REX candidate families
- candidate side LONG/SHORT
- candidate strength
        |
        v
execution/rex_llm_live.py
- live REX candidate scoring
- frozen RLLM gate
- data quality checks
- stale signal age calculation
        |
        v
execution/wave_execution.py
- dry-run/testnet/live safety gates
- manual regime gate
- short-only pilot gate
- flat position requirement
- no open orders requirement
        |
        v
../workspace/wave_trading
- Binance futures client
- maker order execution
```

---

## 주요 피쳐

### Price action / REX
Rolling extrema 기반 위치 정보가 현재 가장 중요한 축입니다.

예시:
- `rex_36_range_pos`
- `rex_144_range_pos`
- `rex_576_range_pos`
- `rex_2016_range_pos`
- `rex_8640_range_pos`
- `rex_*_range_width_pct`
- `rex_*_cur_to_min_pct`
- `rex_*_max_to_cur_pct`

핵심 가설:
- 여러 기간의 range 위치와 현재가의 max/min 상대 위치는 LLM이 해석하기 좋은 구조화된 price action 표현입니다.
- 단일 피쳐는 약하지만, 여러 약한 알파가 결합될 때 유효할 수 있습니다.

### REX에서 LLM selector가 필요한 이유

REX는 최종 매매 모델이라기보다 **후보 생성기(candidate generator)** 로 봐야 합니다. `rex_htf_pullback_resume` /
`rex_htf_pullback_reclaim` 같은 family는 rolling extrema 기반으로 “이 위치에서 반등/재개 후보가 있다”를 만들지만, 후보가
많아질수록 validation spike와 regime mismatch가 섞입니다. 현재 레포의 취지도 원래의 자유형 analyzer/trader가 아니라,
**deterministic 후보 생성 + LLM/RLLM gate/selector + 엄격한 OOS 검증 + 안전 실행**으로 수렴했습니다.

운영 원칙:

- LLM selector는 **새 방향을 창조하지 않습니다.** REX가 만든 `side`와 `hold_bars`를 받아 `TRADE/ABSTAIN`
  또는 `TAKE/SKIP`만 결정합니다.
- live에서는 raw REX 후보를 그대로 주문으로 보내지 않습니다. 최소한 frozen symbolic gate가 필요하고,
  scale-up 전에는 frozen LLM selector를 shadow/live-dry-run으로 검증해야 합니다.
- `family` label은 보조 설명값입니다. 최근 재생성 overlap 검증에서 `resume`/`reclaim` label만 다르고
  `gate/side/hold/signal_pos`가 같은 row가 있었으므로, 실행 판단은 `gate`, `side`, `hold_bars`,
  `signal_pos` 중심으로 고정합니다.
- selector 학습/선택은 오염 방지를 위해 chronology를 고정합니다. 예: threshold/train은 과거 구간,
  validation/test에서 margin 또는 adapter를 선택하고, eval/2026은 선택 후 보고 전용으로 둡니다.
- REX dual-regime/live gate는 kimchi/USDKRW/DXY 같은 외부 피쳐를 쓰므로 live DB의 외부 데이터 freshness가
  깨지면 거래를 차단하거나 historical neutral-fill 규칙과 동일하게 처리해야 합니다.

#### 생성 방법: compact regime-thesis selector

현재 live pilot에 가장 가까운 형태입니다. 검증된 symbolic thesis를 label-first SFT 데이터로 바꿔 작은 모델이
`TRADE`/`ABSTAIN`을 빠르게 logprob scoring 하도록 만듭니다. 현재 표준 파일은 이미 생성되어 있습니다.

| split | 파일 |
|---|---|
| train | `data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl` |
| test | `data/rex_regime_thesis_range_kimchi_label_test_2025.jsonl` |
| eval | `data/rex_regime_thesis_range_kimchi_label_eval_2026h1.jsonl` |

새로 만들 때는 `training.build_rex_regime_thesis_sft`를 사용하되, `train-jsonl`, `test-jsonl`, `eval-jsonl`은
이미 시간순으로 분리된 REX candidate/action record를 넣어야 합니다. 같은 파일을 test/eval에 중복 지정하지 않습니다.

기본 gate prior:

```text
range_vol >= 0.023959233645008706
AND kimchi_premium_change <= 0.0
```

학습 전 dry-run:

```bash
python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl \
  --output-dir checkpoints/dryrun_rex_regime_thesis \
  --max-samples 512 \
  --sample-mode balanced \
  --max-steps 1 \
  --dry-run
```

짧은 LoRA sanity run:

```bash
python -m training.train_text_sft \
  --model-name gemma4-e4b \
  --train-jsonl data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl \
  --output-dir checkpoints/rex_regime_thesis_gemma4_lora_sanity \
  --max-samples 512 \
  --sample-mode balanced \
  --max-seq-length 1024 \
  --max-steps 32 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32
```

평가는 greedy generation보다 `TRADE` vs `ABSTAIN` candidate logprob로 합니다. label-first 포맷을 쓰는 이유는
첫 토큰/짧은 label logprob가 바로 trading gate score가 되기 때문입니다.

```bash
python -m training.eval_text_label \
  --eval-jsonl data/rex_regime_thesis_range_kimchi_label_eval_2026h1.jsonl \
  --output results/rex_regime_thesis_label_eval.json \
  --key decision \
  --model-name gemma4-e4b \
  --adapter-dir checkpoints/rex_regime_thesis_gemma4_lora_sanity \
  --prediction-mode candidate_logprob \
  --score-normalization sum \
  --predictions-output results/rex_regime_thesis_label_predictions.jsonl
```

#### 생성 방법: REX candidate TAKE/SKIP ranker

더 RLLM다운 형태는 symbolic gate를 teacher로 복제하는 대신, REX 후보마다 signal-time feature prompt를 만들고
미래 path reward는 **학습 label 생성에만** 사용해 `TAKE/SKIP` selector를 학습하는 것입니다.

```bash
python -m training.build_rex_candidate_ranker_records \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz \
  --train-output data/rex_candidate_ranker_train.jsonl \
  --test-output data/rex_candidate_ranker_test_2024.jsonl \
  --eval-output data/rex_candidate_ranker_eval_2025.jsonl \
  --summary-output data/rex_candidate_ranker_summary.json \
  --combo rex_htf_pullback_resume:0.85,rex_htf_pullback_reclaim:0.85 \
  --threshold-start 2020-01-01 \
  --threshold-end 2024-01-01 \
  --train-start 2020-01-01 \
  --train-end 2024-01-01 \
  --test-start 2024-01-01 \
  --test-end 2025-01-01 \
  --eval-start 2025-01-01 \
  --eval-end 2026-01-01 \
  --hold-bars 144 \
  --stride-bars 24
```

절차:

1. 먼저 ridge/logistic 같은 cheap baseline으로 `TAKE/SKIP`가 학습 가능한지 확인합니다.
2. 그 다음 `training.train_text_sft`로 짧은 LoRA sanity run을 돌립니다.
3. `training.eval_rex_candidate_ranker_adapter`로 validation에서 margin만 고르고 eval은 그대로 보고합니다.
4. long run은 adapter가 ridge sanity floor를 이긴 뒤에만 진행합니다.

과거 sanity 결과상 20-step Gemma4 candidate-ranker는 2025 validation은 좋아졌지만 2026에서 깨졌습니다. 따라서
현재 live에 바로 쓸 수 있는 것은 **candidate-ranker 자체가 아니라 frozen symbolic/compact regime-thesis gate**이고,
candidate-ranker는 shadow selector 후보로 취급합니다.

### Macro / Premium
- synthetic DXY
- USDKRW
- kimchi premium
- kimchi premium change
- Binance premium index
- funding rate

주말 FX 결측 처리:
- 학습/eval에서는 짧은 backward-asof tolerance 후 결측이면 availability=0, numeric feature=0으로 neutral-fill했습니다.
- live도 동일하게 **FX 휴장/주말에는 external missing을 hard block하지 않도록** 맞췄습니다.
- 평일 장중 external missing은 데이터 장애로 보고 차단합니다.

### Position / execution context
현재 live runner는 신규 진입 전 다음을 확인합니다.
- stale signal age
- manual regime
- allowed side
- existing position 없음
- open orders 없음

---

## Live Trading 상태

### ExtraTrees rank-7 운영 라이프사이클

2026-07-15 연구에서 동결된 ExtraTrees rank-7은 OOS/stability 기준을 통과한
신규 research champion 후보지만, 아직 현재 `execution/rex_llm_live.py`에 연결된
live policy는 아닙니다. 학습 cutoff, 교체 주기, UTC 시간별 추론, 데이터 freshness,
shadow/testnet/canary, 롤백 및 live enablement blocker는 다음 문서에 고정했습니다.

- [`ExtraTrees rank-7 production lifecycle full battery`](docs/expanding-extratrees-rank7-production-lifecycle-full-battery-2026-07-16.md)

운영 핵심은 **시간별 추론, 월별 challenger rehearsal, 연 1회 champion refit/교체**입니다.
동일한 frozen rank-7 계약에서 직접 비교한 결과, 월별 refit은 부분 연도/source
재가중으로 edge가 희석되어 2023-2026H1 전체 CAGR/MDD가 `2.15`에 그쳤고 연별은
`3.13`을 유지했습니다. 따라서 현재 월별 job은 shadow diagnostic 전용이며 live
자동 교체는 금지합니다.

- [Pre-2025 cadence selection](docs/expanding-extratrees-rank7-refit-cadence-pre2025-2026-07-16.md)
- [Frozen cadence OOS comparison](docs/expanding-extratrees-rank7-refit-cadence-oos-2026-07-16.md)

### 구현됨
- `execution/wave_execution.py`
  - RLLM policy record를 wave_trading Binance futures executor로 변환합니다.
  - 기본값은 안전하게 `dry_run=true`, `allow_live_orders=false`, `manual_regime=UNKNOWN`입니다.
- `execution/rex_llm_live.py`
  - DB에서 최신 completed bar를 읽고 REX+RLLM policy record를 만든 뒤 실행 브릿지에 전달합니다.
- 설정:
  - `configs/live/rex_llm_binance_testnet_bear_pilot.json`

### 안전장치
- live order는 명시적으로 `--live --allow-live-orders --manual-regime BEAR`가 있어야 합니다.
- 현재 pilot config는 `allowed_signals=["SHORT"]`입니다.
- `require_flat_position=true`: 기존 포지션이 있으면 진입 차단.
- `require_no_open_orders=true`: 미체결 주문이 있으면 진입 차단.
- `max_probability_age_sec=600`: 최신 completed bar가 너무 오래됐으면 차단.
- HOLD/ABSTAIN은 exchange 초기화도 하지 않는 순수 NOOP입니다.

### Testnet 검증 완료
수행한 검증:
1. 기존 testnet BTCUSDT LONG 포지션 청산.
2. REX live policy run 확인.
3. 강제 SHORT smoke 주문으로 실제 주문 경로 확인.
4. smoke 포지션 즉시 청산.
5. 최종 상태 flat/open_orders=0 확인.

결과:
- `RLLM -> wave_execution -> wave_trading -> Binance testnet futures` 주문 경로 정상.
- 현재 정책 실행은 신호가 없으면 `ABSTAIN/NOOP`.

### Live mainnet 상태
- mainnet private API preflight에서 Binance API key 권한/IP 제한 오류를 확인했습니다.
- 오류:
  - `API Error 401: Invalid API-key, IP, or permissions`
- 따라서 mainnet 실주문은 아직 불가합니다.
- 원인 후보:
  - futures 권한 없음
  - IP whitelist 미등록
  - live key/secret 불일치

---

## 데이터 소스

현재 live DB에서 필요한 외부 수집 데이터:
- Binance `BTCUSDT` 1m futures bars
- Upbit `KRW-BTC` 1m bars
- Polygon/FX `USDKRW` 1m bars
- FX components for synthetic DXY:
  - `EURUSD`, `USDJPY`, `GBPUSD`, `USDCAD`, `USDSEK`, `USDCHF`
- Binance futures premium index kline
- Binance funding rate

관련 문서:
- `docs/live-db-binance-premium-funding-data-request.md`
- `docs/live-rex-llm-binance-bridge-2026-07-03.md`

---

## 중요 파일

| 파일 | 역할 |
|---|---|
| `preprocessing/live_db_features.py` | live DB -> completed bar -> feature snapshot |
| `preprocessing/market_features.py` | market/HTF/REX/macro/aux feature 생성 |
| `preprocessing/external_features.py` | DXY, kimchi, USDKRW join 및 availability 처리 |
| `preprocessing/binance_aux_features.py` | premium index, funding rate join |
| `training/event_candidate_pool_probe.py` | REX 및 후보 family strength/side 생성 |
| `training/build_rex_regime_thesis_sft.py` | frozen gate를 LLM SFT target으로 변환 |
| `execution/rex_llm_live.py` | live REX+RLLM policy runner |
| `execution/wave_execution.py` | wave_trading 실행 브릿지 및 안전 게이트 |
| `configs/live/rex_llm_binance_testnet_bear_pilot.json` | testnet/live pilot 설정 |
| `docs/expanding-extratrees-rank7-production-lifecycle-full-battery-2026-07-16.md` | rank-7 학습·교체·시간별 추론·승격/롤백 운영 계약 |
| `docs/expanding-extratrees-rank7-refit-cadence-oos-2026-07-16.md` | rank-7 월별/연별 expanding refit 누수 격리 비교 |
| `.omx/plans/long-strategy-2025-eval-plan.md` | long-side 전략 탐색 계획 |

---

## 실행 예시

### Testnet dry run

```bash
../workspace/wave_trading/.venv/bin/python -m execution.rex_llm_live \
  --config configs/live/rex_llm_binance_testnet_bear_pilot.json \
  --env .env \
  --dry-run \
  --lookback-minutes 45000
```

### Testnet live 1회 실행

```bash
../workspace/wave_trading/.venv/bin/python -m execution.rex_llm_live \
  --config configs/live/rex_llm_binance_testnet_bear_pilot.json \
  --env .env \
  --live \
  --allow-live-orders \
  --manual-regime BEAR \
  --lookback-minutes 45000
```

주의:
- config가 `testnet=true`인지 확인해야 합니다.
- mainnet은 API 권한/IP 문제가 해결되기 전까지 사용하지 않습니다.
- `.env`는 절대 커밋하지 않습니다.

### Portfolio live post-only refresh 주의

`execution.portfolio_live`의 portfolio runner는 entry/exit 모두 maker/post-only 주문을 일정 주기로 취소하고 새 maker 가격으로 다시 거는 방식으로 추격합니다. 이때 반드시 **미체결 수량이 아니라 남은 수량**만 재주문해야 합니다.

운영 규칙:
- refresh 직전 `get_order.executedQty`를 먼저 반영합니다.
- cancel 응답의 `executedQty`도 다시 반영합니다. cancel 중 체결되는 race가 있을 수 있기 때문입니다.
- 새 post-only 주문 수량은 `requested_quantity - reconciled_filled_quantity`입니다.
- 남은 수량이 Binance 최소 주문 수량보다 작으면 더 이상 추격 주문을 내지 않습니다.
- timeout cancel도 동일하게 최종 체결량을 reconcile한 뒤 `FILLED`, `PARTIAL_CANCELLED`, `TIMEOUT_CANCELLED`를 판단합니다.

회귀 테스트:

```bash
uv run python -m unittest tests/test_portfolio_live.py
```

특히 `test_post_only_refresh_reorders_only_uncancelled_remainder`는 첫 주문이 부분체결되고 cancel 중 추가 체결된 상황에서 다음 추격 주문이 최초 수량이 아니라 잔여 수량만 나가는지 검증합니다.

---

## 하드웨어 기준

### Live trading
`16GB RAM + RTX 3060 Ti 8GB`로 충분히 가능권입니다.

Live runner는 대부분 다음 작업입니다.
- DB query
- pandas feature 생성
- REX candidate 계산
- 짧은 LLM/gate inference 또는 deterministic gate
- exchange API 호출

GPU 병목보다 DB/API/프로세스 안정성이 더 중요합니다.

### Fine-tuning / 연구
- Gemma 4 E4B 계열을 full precision으로 fine-tune하는 것은 3060 Ti 8GB에서 어렵습니다.
- 가능권 설정:
  - 4-bit QLoRA
  - batch size 1
  - 짧은 context 768~1280
  - gradient accumulation
  - gradient checkpointing
- 안정적 연구/반복 학습은 24GB+ VRAM 또는 현재 작업 머신급 GPU가 더 적합합니다.

---

## 지금까지의 주요 교훈

1. **이미지 기반만으로는 부족했습니다.**
   - 수치/구조 정보 손실이 크고, 일반화가 약했습니다.
2. **LLM은 숫자 자체보다 구조화된 맥락/규칙/후보 해석에 더 적합합니다.**
   - 후보 생성은 deterministic feature/candidate engine이 담당하고, LLM은 gate/해석/선택 쪽이 더 안정적입니다.
3. **과최적화와 누수 방지가 핵심입니다.**
   - train/test/eval split을 명확히 해야 합니다.
   - 2026 이후 데이터를 격리하는 long 전략 탐색 계획을 새로 세웠습니다.
4. **성과가 좋아 보여도 live 전 안전장치가 필수입니다.**
   - stale signal, 기존 포지션, open order, manual regime, side whitelist를 모두 확인해야 합니다.
5. **현재 후보는 regime-dependent입니다.**
   - 최근 약세장/하락장 short pilot으로는 쓸만하지만, 장기 전 구간 범용 전략으로는 아직 부족합니다.

---

## 다음 단계

### 1. Long-side 전략 탐색
계획 파일:
- `.omx/plans/long-strategy-2025-eval-plan.md`

방침:
- 2026 이후 데이터 완전 제외.
- Train: 2020~2023
- Validation/selection: 2024
- Final eval: 2025
- long-only 후보를 탐색하고, 2025에서 평가합니다.

### 2. Live readiness 개선
- mainnet API 권한/IP 문제 해결.
- live process watchdog 추가.
- 포지션 ownership/reconciliation 설계.
- 실거래 전 작은 notional test와 자동 청산/비상 정지 루틴 검증.

### 3. RLLM 구조 개선
- REX 후보 + LLM reasoning prompt 개선.
- Gemma 4 E4B 또는 소형 모델 기반 deterministic/LLM hybrid gate 비교.
- feature text representation 개선.

---

## Git / 운영 메모

- 작업 단위별 커밋을 유지합니다.
- `.env`와 credential은 커밋하지 않습니다.
- 큰 데이터/체크포인트는 디스크 사용량을 관리해야 하며, WSL 사용량은 300GB를 넘기지 않도록 정리합니다.
- 현재 remote:
  - `https://github.com/pakchu/rllm.git`
  - branch: `main`
