# Chronos-2 fit-oriented alpha validation (2026-07-13)

## Verdict

각 zero-shot forecast score의 부호를 fit 2020-2022 Spearman으로만 고정했다.
6개 score 모두 fit에서 음수였으므로 `orientation=-1`이 적용됐다. 2023에서
고정한 Top-10의 `alpha_pool`/`live_grade`는 **0개**였다.

## Best evidence

| Rank | Stream | Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | median path mean, long, 365d q80 | 2023 select | +39.87% | 39.90% | 8.37% | 4.77 | 74 |
| 1 | same | 2024 Test | +26.08% | 26.02% | 10.77% | 2.42 | 67 |
| 1 | same | 2025 Eval | +5.35% | 5.35% | 11.04% | 0.48 | 65 |
| 1 | same | 2026 YTD | +2.87% | 7.03% | 5.21% | 1.35 | 38 |
| 3 | 24h/48h consensus, long, 180d q80 | 2024 Test | +28.85% | 28.78% | 7.83% | 3.68 | 72 |
| 3 | same | 2025 Eval | +2.23% | 2.23% | 11.04% | 0.20 | 58 |

fit orientation은 2023·2024·2026 rank correlation을 양수로 바꿨지만 2025는
모든 stream이 다시 음수로 반전했다. 따라서 Chronos-2 forecast prior는 이
BTC 실행 horizon에서 invariant alpha가 아니다.

## Validity

- orientation은 fit 2020-2022만 사용
- model/input/revision은 raw source manifest와 동일
- current score 제외 rolling percentile
- 2023 executed-path Top-10 freeze 후 2024+ metric 계산
- next-bar 5m open, hold 48h, 0.5x, 편도 6bp
- full-window CAGR, strict intratrade MDD

## Artifacts

- Evaluator: `training/evaluate_chronos2_fit_oriented_alpha.py`
- Test: `tests/test_chronos2_fit_oriented_alpha.py`
- Frozen manifest: `results/chronos2_fit_oriented_top10_manifest_2026-07-13.json`
- Result: `results/chronos2_fit_oriented_alpha_scan_2026-07-13.json`
