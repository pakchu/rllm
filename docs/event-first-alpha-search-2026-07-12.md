# Event-first alpha search (2026-07-12)

## 목적
기존 feature-threshold/gate 최적화가 과적합되기 쉬워서, 먼저 구조적 이벤트 가족을 정의하고 train/test/eval로 후보 이벤트 자체를 검증했다.

## 누수 방지
- 입력: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz`
- 이벤트 strength threshold: train 분포에서만 산출
- 후보/ensemble 선택: train+test만 사용
- eval: 선택 후 단 1회 holdout 확인
- 진입: signal 이후 1 bar delay
- 피쳐: 현재/과거 row 기반 rolling OHLCV, taker, OI, funding/premium placeholders, DXY/USDKRW/kimchi 계열

## Scan A: broad event family
Command artifact: `results/event_family_params_broad_2026-07-12.json`

Split:
- train: 2020-01-01..2024-01-01
- test: 2024-01-01..2025-01-01
- eval: 2025-01-01..2026-06-01

Selected by test ensemble:
- `rex_htf_long_pullback_resume@h144@q0.85`
- `rex_htf_pullback_resume@h144@q0.92`
- `rex_htf_deep_pullback_resume@h144@q0.90`

Full-window stats:

| split | abs return | full-window CAGR | strict MDD | CAGR/MDD | trades | p approx | sides |
|---|---:|---:|---:|---:|---:|---:|---|
| train 2020-2023 | 31.97% | 7.18% | 15.64% | 0.46 | 265 | 0.243 | L180/S85 |
| test 2024 | 12.80% | 12.77% | 5.55% | 2.30 | 46 | 0.157 | L44/S2 |
| eval 2025-2026H1 | 8.10% | 5.67% | 4.35% | 1.30 | 27 | 0.117 | L7/S20 |

Verdict: positive but not strong enough. It does not clear target CAGR/MDD>=3 under full-period CAGR.

## Scan B: recent REX event family
Command artifact: `results/event_family_params_recent_rex_2026-07-12.json`

Split:
- train: 2020-01-01..2025-01-01
- test: 2025-01-01..2026-01-01
- eval: 2026-01-01..2026-06-01

Selected by test:
- `rex_htf_pullback_reclaim@h144@q0.75`
- Entry concept: multiscale REX higher-timeframe pullback where local trend reclaims in the larger trend direction.
- Exit: fixed hold 144 bars (12h), next-open/1-bar-delay, strict non-overlap simulator.

Full-window stats:

| split | abs return | full-window CAGR | strict MDD | CAGR/MDD | trades | p approx | sides |
|---|---:|---:|---:|---:|---:|---:|---|
| train 2020-2024 | 35.32% | 6.23% | 22.30% | 0.28 | 768 | 0.311 | L458/S310 |
| test 2025 | 15.32% | 15.33% | 4.83% | 3.17 | 74 | 0.037 | L31/S43 |
| eval 2026H1 | 5.34% | 13.40% | 4.40% | 3.04 | 43 | 0.328 | L15/S28 |

Verdict:
- 최근 regime에서는 test/eval 모두 CAGR/MDD>=3을 간신히 넘는다.
- 그러나 장기 train의 MDD가 22.3%이고 CAGR/MDD 0.28이라 standalone/live-grade로 승격하면 안 된다.
- 해석: 최근 하락/전환 regime에서만 작동하는 REX-context 후보. RLLM에는 “regime-conditioned event context”로 투입 가능하지만, 단독 전략으로는 부족하다.

## 결론
- 새로운 방식(event-first)은 raw gate 탐색보다 나은 후보를 찾았다.
- 핵심 edge는 여전히 REX/HTF pullback 계열에 집중되어 있다.
- 완전한 새 독립 알파라기보다는 기존 REX thesis의 event-context 확장이다.
- 다음 단계는 이 후보를 기존 portfolio/RLLM meta-controller의 입력으로 넣고, train에서 MDD가 커지는 구간을 설명하는 regime split을 먼저 찾아야 한다.
