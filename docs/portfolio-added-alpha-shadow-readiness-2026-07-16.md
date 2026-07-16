# Added-alpha portfolio forward-shadow readiness — 2026-07-16

## 결론

Gross 8 후보는 **주문 없는 DB one-shot shadow scoring**까지 연결됐다. 현재 DB의
완료 5분봉은 fresh였고 외부 데이터 availability도 통과했으며, 주문 경로는 실행되지
않았다. 하지만 전체 5개 중 **4개만 signal scoring 가능**하고, `frozen_annual_rank7`은
필수 runtime bundle이 없어 명시적으로 fail-close한다. 따라서 현재 상태는
`forward_shadow_candidate_not_live`이며 live 승격은 금지된다.

- 현재 live 설정 `portfolio_gross385_trainmdd40_2026-07-12.json`은 변경하지 않았다.
- DB smoke: `orders_enabled=false`, `completed_bar_fresh=true`, 8.24초,
  max RSS 449,924 KiB.
- 신호 scoring: 4/5 가능.
- 완전한 trade lifecycle historical/live parity: 아직 0/5 인증되지 않음.
- 증거: [`portfolio_added_alpha_shadow_db_smoke_2026-07-16.json`](../results/portfolio_added_alpha_shadow_db_smoke_2026-07-16.json)
- 기계 판독 readiness: [`portfolio_added_alpha_shadow_readiness_2026-07-16.json`](../results/portfolio_added_alpha_shadow_readiness_2026-07-16.json)

## Sleeve별 상태

| Sleeve | Signal scoring | Execution 상태 | 남은 blocker |
|---|---|---|---|
| `fresh_kimchi_fx` | 가능 | 불완전 | 연구의 TP/SL barrier exit 미연결, schedule parity 미검증 |
| `frozen_annual_rank7` | fail-close | 불가 | 모델/임계값/40-feature/state warm-start/source별 exit/parity bundle 없음 |
| `rex_taker_low_range_position` | 가능 | fixed-hold 경로 존재 | historical/live schedule parity 미검증 |
| `cand_rex_veto_7` | 가능 | fixed-hold 경로 존재 | historical/live schedule parity 미검증 |
| `markov_transition_long` | 가능 | fixed-hold 경로 존재 | historical/live schedule parity 미검증 |

`가능`은 현재 완료봉에서 causal feature를 계산하고 신호를 score할 수 있다는 뜻이다.
수익률 backtest와 동일한 체결·exit·포지션 lifecycle이 증명됐다는 뜻은 아니다.

## Rank7 감사

Rank7의 동결 사양은 depth 2, leaf 32, max-features 0.8인 300-tree ExtraTrees
5개이며, source별 net/adverse 예측 평균에 score/risk/interaction quantile을 적용한다.
필요 runtime contract는 다음과 같다.

1. 2026 annual cutoff에서 학습된 5개 모델.
2. funding/premium score threshold와 risk cap, funding width/pullback threshold.
3. 12×5분 지연과 현재 source identity 예외를 포함한 정확한 40-column feature graph.
4. Kalman, BOCPD, semi-Markov, nested-barrier, market-braid의 causal warm-start.
5. 144-bar immutable anchor와 funding/premium source별 exit.
6. frozen prefix feature/activation/schedule hash와 동일한 historical/live replay.

모델 용량은 blocker가 아니다. 동결 learner로 synthetic 460×40 입력에 같은 구조의
5개 모델을 fit한 크기 probe는 joblib 무압축 약 1.19 MiB, compress=3 약 0.29 MiB였다.
이는 production 모델 파일의 정확한 크기가 아닌 구조적 근사치다. 실제 blocker는
feature 및 execution parity다.

현재 shadow tail 45,000분은 가장 큰 명시적 core rolling window인 8,640×5분
(43,200분)을 넘는다. 그러나 누적 state와 weak feature warm-start까지 동일하다는
end-to-end 증거는 없으므로 “45,000분이면 Rank7 전체가 재현된다”고 간주하지 않는다.

## 실행 및 검증

주문 없이 현재 DB를 한 번 score하는 명령:

```bash
env PYTHONPATH=<worktree> <wave-venv-python> \
  -m execution.portfolio_shadow \
  --env <env-path> \
  --output results/portfolio_added_alpha_shadow_db_smoke_2026-07-16.json
```

검증 결과:

```text
40 passed in 5.11s
mode=forward_shadow_score_only
orders_enabled=false
completed_bar_fresh=true
runtime_blocked_sleeves=[frozen_annual_rank7]
signal_scoring_ready_count=4
```

## 다음 승격 게이트

1. 4개 scoreable sleeve의 frozen historical interval에서 decision hash를 연구 신호와
   동일하게 만든다.
2. Fresh Kimchi TP/SL와 Rank7 source별 barrier exits를 portfolio lifecycle에 구현한다.
3. Rank7 bundle exporter와 immutable checksum registry를 만들고 40-feature replay를
   byte/hash 수준으로 검증한다.
4. 모든 sleeve가 통과한 뒤에만 지속 forward shadow를 시작한다.
5. forward 기간에는 파라미터·weight를 변경하지 않고, 충분한 거래 수와 실제 체결
   slippage/MDD를 수집한다. 그 전에는 testnet/live 주문을 허용하지 않는다.
