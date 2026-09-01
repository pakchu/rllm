# PPOSM SFT+RLVR architecture search — 2026-09-02

## 결론

현재 frozen PPOSM 데이터와 causal feature만으로는 **always-TP4를 위험조정 기준까지 이기는 SFT+RLVR 모델 구조를 찾지 못했다**. 수익 자체는 개선한 conditional model이 있었지만, preregistered CAGR/MDD와 stress-MDD gate를 동시에 통과하지 못했다. 따라서 배포 승인은 없으며 historical OOS 추가 탐색도 중단한다.

정형 결과는 `results/pposm_sft_rlvr_architecture_search_negative_result_2026-09-02.json`에 고정했다.

## 가장 가까운 결과

첫 pairwise residual router는 constant TP4가 아니었다.

- OOS routes: `TP4 99 / TP12 44 / SKIP 14`
- base return: `53.0705% → 54.3761%`
- stress return: `50.0383% → 51.5604%`
- CAGR/MDD: `3.3682 → 3.3312` (실패)
- stress MDD: `5.9430% → 6.0946%`, `+0.1516pp` (허용 `+0.01pp`, 실패)

근거: `results/pposm_conditional_residual_sft_rlvr_critic_2026-09-02.json`.

## 검증한 구조

1. **Independent-trade pairwise residual**
   - Qwen2.5-1.5B SFT128 + residual-utility RLVR64.
   - conditional/absolute-profit improvement는 확인했지만 위험조정 gate 실패.
2. **Shadow-TP4 lifecycle-anchor residual**
   - pre-2024 always-TP4가 실제 진입한 102 anchors만 사용.
   - 단일 action 교체가 전체 순차 portfolio에 미치는 log-equity/CAGR-MDD/stress-MDD marginal을 reward로 사용.
   - 37개 train threshold 모두 실패; OOS 미개방.
3. **SKIP/TP12 독립 specialist adapters**
   - 독립 SFT+RLVR adapter와 독립 threshold pair 550개 검증.
   - 통과 0개; OOS 미개방.
4. **Exact-target + lifecycle-utility dual-verifier RLVR**
   - format, exact KEEP/SWITCH target, lifecycle utility를 함께 검증.
   - 23개 train threshold 모두 실패; OOS 미개방.
5. **Rank-budget diagnostic**
   - shared/specialist score 모두 quota 10,609쌍에서 통과 0개.

## 실제 원인

학습 runtime, cached model file, 또는 constant-policy만의 문제가 아니다. 핵심은 **causal feature가 sparse lifecycle marginal label을 안정적으로 분리하지 못한다는 것**이다.

- 102 anchors / SKIP positives 25 / TP12 positives 28.
- richer temporal+shadow expanding-year CV:
  - SKIP best pooled AUC `0.5355`.
  - TP12 temporal AUC `0.5625`, 근사 95% 구간 `0.420–0.705`.
- 15-feature OI+temporal CV:
  - SKIP AUC `0.542` (`0.374–0.700`).
  - TP12 AUC `0.502` (`0.357–0.642`).

즉 더 큰 text model이나 더 많은 RLVR step은 일반화 신호를 만들기보다 102개 train anchors를 암기할 가능성이 높다.

## OI 및 cache 확인

후속 검사에서 중요한 정정이 있었다. OI-enriched cache에는 OI/OI-value만 있었지만, 원본 Binance UM metrics archive에는 실제 positioning ratio 4종이 보존되어 있었다.

- `count_toptrader_long_short_ratio`
- `sum_toptrader_long_short_ratio`
- `count_long_short_ratio`
- `sum_taker_long_short_vol_ratio`

strict prior-completed-5m join으로 다시 계산한 train-only expanding-year CV:

- SKIP AUC `0.6344`, 95% 구간 `0.5086–0.7521`, balanced accuracy `0.5549`.
- TP12 AUC `0.5305`, 95% 구간 `0.3778–0.6772`, balanced accuracy `0.5455`.
- global account ratio null `2.94%`, top-trader ratios null `35.29%`, taker ratio null `15.69%`.

따라서 SKIP에는 약한 신호가 있으나 TP12와 전체 materiality를 지지할 수준은 아니다. 즉 새 full SFT+RLVR run은 아직 보류한다. 정형 근거는 `results/pposm_true_ratio_source_continuation_2026-09-02.json`에 있다.

표본을 462개 counterfactual signals로 늘린 추가 train-only 검사도 최종 gate를 넘지 못했다.

- 최고 후보: ratio-only TP12.
- pooled AUC `0.5867` (`0.5275–0.6409`), 요구 `>=0.60` 미달.
- balanced accuracy `0.6156`.
- 연도별 AUC `0.195 / 0.696 / 0.235`로 regime 방향이 반전.
- train 내부 경제 proxy는 일부 통과했지만 classification gate 미통과 score를 후행 calibration한 값이므로 OOS 근거로 사용하지 않았다.

근거: `results/pposm_ratio_counterfactual_diagnostic_2026-09-02.json`.

Cache builder는 `training/build_oi_enriched_cache.py`에서 ratio 4종을 선택적으로 보존하도록 수정했다. 현재 행보다 정확히 한 개 완료된 5분 source row만 사용하고 ratio forward-fill/interpolation은 하지 않는다.

Live/history 경로에는 여전히 별도 운영 위험이 있다.

- live runner는 `open_interest_binance_live` snapshot을 적재한다.
- historical 5m `open_interest_binance`는 명시적 import/backfill이 필요하다.
- live config는 `1m` default가 될 수 있지만 archive는 `5m`이므로 consumer period mismatch 시 OI availability가 비어 장기간 gate가 닫힐 수 있다.
- historical OI/ratio feature는 최소 한 개 완결 5m bar shift와 gap/availability 처리가 필요하다.

관련 코드: `/home/pakchu/upbit-usdt/app/services/ingest.py`, `/home/pakchu/upbit-usdt/app/storage/bars.py`.

## 재개 조건

다음 조건 전에는 historical OOS를 더 열거나 gate를 낮추지 않는다.

- 새 hash-bound causal source가 expanding-fold pooled AUC `>=0.60`.
- AUC lower bound가 `0.50` 초과.
- balanced accuracy `>=0.55`와 연도별 방향 일관성.
- 후보 source: 실제 Binance long-short ratio history, 실제 taker long-short ratio history, 또는 fresh forward/live-parity lifecycle labels.

현재 상태는 **WATCH / verified negative / no deployment**다.

로컬 feature/source 탐색은 `results/pposm_sft_rlvr_goal_blocker_2026-09-02.json`에서 종료 조건을 고정했다. 새 forward labels 또는 새 hash-bound source가 들어오기 전에는 같은 historical OOS를 이용한 threshold/model repair를 하지 않는다.
