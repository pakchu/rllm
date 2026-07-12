# Invariant ensemble rolling-gate validation (2026-07-13)

## Verdict

불변 tail classifier의 absolute 2023 score threshold를 폐기하고, 현재 score를
제외한 과거 180/365일 rolling percentile로 교체했다. 개별 모델과 고정
objective ensemble을 포함한 792개 policy specification 중 2023 양수 후보
359개, distinct Top-10을 사전 고정했다.

공식 `alpha_pool`/`live_grade` 승격은 **0개**였다. 그러나 사전 6위
`ensemble_invariant_stable24`가 세 OOS 모두 양수이고 목표에 근접했다.

## Near candidate: pre-evaluation rank 6

Configuration:

- stream: `ensemble_invariant_stable24`
- members: stable24의 V-REx/Group-DRO linear+MLP 6개
- side: long-only
- calibration: prior 1460 anchors (~365d), q95
- hold: 48h

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 select | +34.58% | 34.61% | 5.98% | 5.79 | 46 |
| 2024 Test | +30.20% | 30.13% | 7.94% | 3.79 | 51 |
| 2025 Eval | +15.10% | 15.11% | 5.16% | 2.93 | 45 |
| 2026 YTD | +8.39% | 21.35% | 5.03% | 4.25 | 23 |

2025가 3.0보다 0.07 낮고 2026이 5.0보다 0.75 낮으므로 현재 기준에서는
승격하지 않는다. OOS를 본 뒤 q/window를 미세 조정하는 것도 금지한다.

## Other Top-10 evidence

- Rank 8 MLP V-REx10 stable24: 2024 ratio 3.00, 2025 0.41, 2026 -0.05
- Rank 9 MLP V-REx1 stable24: 2024 2.89, 2025 1.13, 2026 1.44
- 나머지 상위 linear 후보는 2025/2026에서 대부분 붕괴했다.

## Score stability

고정 ensemble score는 여전히 세 OOS에서 양수 Spearman을 보였다.

| Ensemble | 2024 | 2025 | 2026 |
|---|---:|---:|---:|
| invariant stable8 | +0.0866 | +0.0409 | +0.1279 |
| all stable8 | +0.0836 | +0.0424 | +0.1246 |
| invariant stable16 | +0.0763 | +0.0438 | +0.1180 |
| invariant stable24 | +0.0592 | +0.0098 | +0.1238 |

rolling percentile은 scale drift를 줄였지만, 일부 ensemble member가 서로
반대 판단을 내리는 구간까지 평균 score tail에 포함되는 문제가 남았다.

## Next branch

OOS threshold 미세조정 대신 train-only model ensemble의 동시점 disagreement를
사용한다.

1. ensemble mean과 member standard deviation을 함께 계산한다.
2. mean의 절댓값을 disagreement만큼 shrink한 confidence score를 만든다.
3. 현재 score를 제외한 causal rolling percentile을 동일하게 적용한다.
4. shrink family를 2023에서 다시 Top-10 고정하고 2024+를 report-only로 둔다.

## Artifacts

- Evaluator: `training/evaluate_invariant_groupdro_rolling_gate.py`
- Test: `tests/test_invariant_groupdro_rolling_gate.py`
- Frozen manifest: `results/invariant_groupdro_rolling_top10_manifest_2026-07-13.json`
- Result: `results/invariant_groupdro_rolling_alpha_scan_2026-07-13.json`
