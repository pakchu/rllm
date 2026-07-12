# Invariant ensemble uncertainty validation (2026-07-13)

## Verdict

V-REx/Group-DRO 6-member ensemble에 disagreement-aware shrink, SNR, sign
agreement gate를 적용했다. 2023에서 고정한 Top-10의 `alpha_pool` 및
`live_grade`는 **0개**였다.

불확실성 gate는 일부 연도의 MDD를 줄였지만 2024/2025/2026을 동시에
개선하지 못했다. 이 branch에서 gate 변형을 더 늘리는 것은 false discovery만
키우므로 중단한다.

## Protocol

- train-only stable8/16/24 feature sets 재현
- member models: linear/MLP V-REx1, V-REx10, Group-DRO (6개)
- transforms:
  - member mean
  - signed shrink `|mean| - 0.5*std`, `|mean| - 1.0*std`
  - SNR `mean / (std + 0.05)`
  - 4/6, 5/6, 6/6 sign agreement
- zero-confidence는 explicit abstain
- prior 180/365d rolling percentile, 현재 score 제외
- 2023 executed-path Top-10 freeze 후 2024+ report-only
- next-bar open, hold 48h, 0.5x, 편도 6bp
- full-window CAGR, strict intratrade MDD

총 504개 policy specification 중 2023 양수 후보는 206개였다.

## Top evidence

| Rank | Stream | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | stable24 shrink0.5, long, 365d q95 | 2023 select | +23.66% | 23.68% | 3.61% | 6.56 | 36 |
| 1 | same | 2024 Test | +37.69% | 37.60% | 7.94% | 4.73 | 56 |
| 1 | same | 2025 Eval | +8.90% | 8.90% | 7.25% | 1.23 | 44 |
| 1 | same | 2026 YTD | +6.17% | 15.48% | 5.70% | 2.72 | 20 |
| 2 | stable24 mean, long, 365d q95 | 2024 Test | +30.20% | 30.13% | 7.94% | 3.79 | 51 |
| 2 | same | 2025 Eval | +15.10% | 15.11% | 5.16% | 2.93 | 45 |
| 2 | same | 2026 YTD | +8.39% | 21.35% | 5.03% | 4.25 | 23 |

Rank 1 shrink는 2024를 개선했지만 2025/2026을 악화시켰다. Rank 2 mean은
이전 rolling 실험의 근접 후보와 동일하다. member agreement 4/6~6/6 후보는
2025에서 대부분 음수 또는 ratio 0.5 이하로 붕괴했다.

## Conclusion

불변 feature ensemble은 weak rank signal을 가지지만, 현재 fixed 48h tail
action으로 충분한 경제적 알파가 되지 않는다. 다음 실험은 gate 최적화가 아니라
숫자 시계열을 위해 사전학습된 time-series foundation model의 multivariate
forecast/latent representation으로 feature source 자체를 바꾼다.

## Artifacts

- Evaluator: `training/evaluate_invariant_ensemble_uncertainty.py`
- Test: `tests/test_invariant_ensemble_uncertainty.py`
- Frozen manifest: `results/invariant_ensemble_uncertainty_top10_manifest_2026-07-13.json`
- Result: `results/invariant_ensemble_uncertainty_alpha_scan_2026-07-13.json`
