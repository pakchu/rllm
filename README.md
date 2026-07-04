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
