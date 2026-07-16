# Added-alpha shadow signal parity audit — 2026-07-16

## 결론

주문·체결·exit를 제외한 **후보 신호(source-contract) 패리티는 4/4 통과**했다.
Fresh Kimchi, Markov, REX-taker, REX-veto 모두 각자 동결된 역사 소스 구간에서
feature contract, stride, 후보 방향, frozen rule gate의 decision hash가 일치한다.

이 결과는 live 승격 근거가 아니다. `frozen_annual_rank7`은 아직 score할 수 없고,
Fresh Kimchi TP/SL 및 전체 포트폴리오의 non-overlap·체결·exit·strict MDD 패리티도
검증되지 않았다. 또한 이 포트폴리오는 연구 오염 후보이므로 forward shadow 전용이다.

- 기계 판독 결과:
  [`portfolio_added_alpha_shadow_signal_parity_2026-07-16.json`](../results/portfolio_added_alpha_shadow_signal_parity_2026-07-16.json)
- 실행 도구: `training.audit_portfolio_shadow_signal_parity`
- strict audit: 27.66초, max RSS 6,600,028 KiB
- 주문 경로: 사용하지 않음

## 패리티 결과

| Sleeve | 비교 범위의 gated decision | LONG | SHORT | mismatch | SHA-256 |
|---|---:|---:|---:|---:|---|
| `fresh_kimchi_fx` | 1,081 | 163 | 918 | 0 | `2c8f177f...ff52` |
| `markov_transition_long` | 1,534 | 1,534 | 0 | 0 | `224a0e7e...dc40` |
| `rex_taker_low_range_position` | 615 | 438 | 177 | 0 | `0c1df0e5...9ff7` |
| `cand_rex_veto_7` | 949 | 558 | 391 | 0 | `841c7d0e...8f43` |

추가 확인:

- Fresh long/short raw mask mismatch: 각각 0.
- Fresh gate feature 8개 mismatch: 모두 0.
- Markov base gate, transition key, schedule mismatch: 모두 0.
- REX-taker base 후보 1,689개, strength/side/gate mismatch: 모두 0.
- REX-veto 평가 범위 base 후보 1,270개, gated 후보 949개 mismatch: 모두 0.
- timestamp modulo stride는 역사 positional grid와 eligibility 내부에서 정확히 일치했다.
  범위 밖 warm-up/end slot은 signal 비교에서 명시적으로 제외했다.

## 수정한 실제 원인

### Markov feature drift

역사 정책은 `build_market_feature_frame(window_size=144, zscore_window=48,
volume_window=48)`을 사용했지만 shadow scorer는 generic 288/96/96 frame을 사용했다.
특히 `trend_96`의 실질 shift가 달라져 후보가 크게 어긋났다. Markov 전용 frozen
feature adapter와 config contract를 추가해 역사 계산과 동일하게 만들었다.

### REX-taker contract drift

동결 JSONL은 2026-07-03 ranker의 144/48/48 feature graph와
`event_candidate_pool_probe._feature_candidates`로 생성됐다. 하지만 shadow config는
2026-07-12 event-reasoning contract를 가리켰다. config를
`rex_candidate_ranker_20260703`으로 수정하고 exact adapter를 추가했다.

### 너무 짧았던 live/shadow history

기존 45,000분은 9,000개의 5분봉뿐이라 completed HTF feature contract를 만족하지
못했다. `market_features_v1`의 3일봉 guard는 17,280 source rows를 요구하므로 최소
86,400분이 필요하다. 신규 shadow 후보는 90,000분을 요구하고, runner는 요청 이력과
실제 반환 이력이 부족하면 DB scoring 전에 fail-close한다.

90,000분 DB smoke 결과:

```text
orders_enabled=false
completed_bar_fresh=true
feature_history_rows=18000
required_feature_history_rows=18000
signal_scoring_ready_count=4
runtime_blocked_sleeves=[frozen_annual_rank7]
elapsed=13.24s
max_rss=708996 KiB
```

`cand_rex_veto_7`의 실제 DB `htf_1w_return_4`도 0으로 붕괴하지 않고
`0.00594889`로 계산됐다.

## 역사 활성 구간을 실행 계약으로 고정

### REX-taker의 2020 pre-source gap

REX-taker의 동결 역사 source contract는 2021-01-01부터다. 포트폴리오 train은
2020-09-01부터이므로 2020년 9~12월에는 이 sleeve가 역사적으로 inactive였다.
처음 진단한 generic scorer는 이 빈 구간에서 46개 후보를 만들 수 있었다. 이를
숨기거나 역사 수익률에 뒤늦게 추가하지 않고 config에 `active_from=2021-01-01`을
명시해 scorer도 같은 구간에서 반드시 inactive가 되게 했다.

수정 후 결과:

- **source-contract signal parity:** 4/4 통과
- **전체 포트폴리오 평가 시작일부터 candidate signal parity:** 4/4 통과
- pre-source 35,136개 5분봉에서 runtime gated signal: 0

### REX-veto의 pre-history OI fail-close

역사 numeric-only gate는 2020-09-01 이전 OI 부재 행 113개를 통과시켰지만 live는
`open_interest_available=0`이면 차단한다. 차이는 모두 포트폴리오 평가 시작 전이며,
평가 범위 안 mismatch는 0이다. 이는 stale/missing OI를 거래하지 않는 의도된
안전 강화다.

## 입력 provenance와 누수 경계

감사는 다른 checkout의 파일을 암묵적으로 읽지 않는다. checkout에 없는 연구
artifact는 `--artifact-root`로 명시해야 하며,
`configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json`에 동결된
market/funding/premium/OI/REX JSONL SHA-256 및 JSONL row count가 하나라도 다르면
feature 계산 전에 종료한다. 결과 JSON에도 실제 검증한 모든 source hash를 기록한다.

- feature builder는 현재 완료봉까지의 rolling/shift/backward-asof만 사용한다.
- Fresh/Markov/REX adapter와 Markov transition에 suffix를 붙여도 prefix가 변하지 않는
  회귀 테스트를 추가했다.
- frozen strength threshold를 사용하며 live expanding quantile로 재선택하지 않는다.
- 이 감사는 미래 label, 수익률 target, 주문 결과를 입력으로 사용하지 않는다.

그러나 이는 **알파 일반화나 pristine OOS를 증명하지 않는다.** 후보 universe와
2025/2026은 이미 연구에 노출됐다. 검증의 의미는 오직 “동결 연구 신호를 no-order
scorer가 같은 입력에서 재구성한다”는 것이다.

## 재현

```bash
PYTHONPATH=. python -m training.audit_portfolio_shadow_signal_parity \
  --artifact-root /home/pakchu/rllm

PYTHONPATH=. <wave-venv-python> -m execution.portfolio_shadow \
  --env <env-path> \
  --lookback-minutes 90000 \
  --output results/portfolio_added_alpha_shadow_db_smoke_2026-07-16.json
```

감사는 mismatch가 있으면 결과 JSON을 쓴 뒤 non-zero로 종료한다. 진단 목적으로만
`--allow-mismatch`를 사용할 수 있다.

## 남은 승격 blocker

1. Fresh Kimchi TP/SL barrier exit와 exact non-overlap replay.
2. Rank7 model/40-feature/state/threshold/source별 exit bundle.
3. 5개 sleeve의 체결·수수료·slippage·position netting·strict MDD end-to-end parity.
4. frozen weight/parameter를 유지한 충분한 forward 거래 표본.
5. 신규 후보를 live runner로 옮길 때도 최소 90,000분 이력을 강제하는 별도 승격 작업.
