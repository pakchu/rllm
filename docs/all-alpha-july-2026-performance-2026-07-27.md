# 전체 atomic alpha — 2026년 7월 성과 감사

- 데이터: `2026-07-01 00:00:00` ~ `2026-07-27 15:00:00` UTC
- 완결 5분봉: **7,668개**
- 전수 범위: **24개 atomic alpha/variant**; 포트폴리오 조합과 raw scan row는 제외.
- 공통 계약: 0.5x, 6 bp/notional/side, next-bar open, overlap suppression, strict MDD.
- 셀: `절대수익 / 관측기간 연율화 CAGR / strict MDD / CAGR-MDD / 거래수`.
- 한 달 미만 CAGR은 불안정하므로 절대수익·MDD·거래수를 우선한다.
- 단일 알파 최대 거래수도 14회라 통계적 승격 근거로는 부족하다.

## 전수 결과

| 알파 | 상태 | 계열 | 성과 | L/S | 승률 |
|---|---|---|---:|---:|---:|
| `fresh_kimchi_fx` | promoted | kimchi_fx | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `frozen_annual_rank7` | promoted | rank7_ml | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `markov_transition_long` | promoted | state_model | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `cand_rex_veto_7` | promoted | rex | -0.3897% / -5.22% / 1.2769% / -4.08 / 1 | 0/1 | 0.0% |
| `rex_taker_low_range_position` | promoted | rex | -1.0433% / -13.40% / 1.5741% / -8.51 / 5 | 0/5 | 20.0% |
| `oi_divergence_pullback` | paper_candidate | open_interest | 2.8971% / 47.96% / 1.2832% / 37.38 / 4 | 4/0 | 75.0% |
| `oi_alt_ratio72_dynamic_exit` | legacy_live_candidate | cross_asset_oi | 1.4626% / 22.04% / 1.4510% / 15.19 / 2 | 2/0 | 50.0% |
| `oi_divergence_highfreq` | paper_candidate | open_interest | 0.7207% / 10.35% / 1.2268% / 8.44 / 12 | 12/0 | 50.0% |
| `oi_divergence_highfreq_selector` | paper_selector_proxy | open_interest_selector | 0.7207% / 10.35% / 1.2268% / 8.44 / 12 | 12/0 | 50.0% |
| `nonpb30_taker` | paper_candidate | price_flow | 0.4916% / 6.96% / 0.9480% / 7.34 / 3 | 3/0 | 66.7% |
| `oi_upbit_ratio288_low` | legacy_live_candidate | cross_venue_oi | 0.4407% / 6.22% / 0.6173% / 10.07 / 4 | 4/0 | 50.0% |
| `pb30_addon` | legacy_candidate | pb30 | 0.0289% / 0.40% / 1.3298% / 0.30 / 3 | 3/0 | 66.7% |
| `bocpd_funding_premium_long` | research_shadow_required | state_model | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `funding_premium_lr_impact_central` | research_shadow_required | funding_premium | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `kalman_funding_premium_long` | research_shadow_required | state_model | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `new_long_minimal_funding_premium` | superseded_live_candidate | funding_premium | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `pb30_base` | legacy_candidate | pb30 | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `rex_htf_range_veto` | weak_research_candidate | rex | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `semimarkov_funding_premium_long` | research_shadow_required | state_model | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `short_kimchi3d` | paper_zero_weight | kimchi_short | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `short_premium_panic` | legacy_live_candidate | premium_short | 0.0000% / 0.00% / 0.0000% / 0.00 / 0 | 0/0 | 0.0% |
| `legacy_rex_dual_regime_auto` | superseded_legacy | rex_legacy | -0.4630% / -6.17% / 3.5433% / -1.74 / 14 | 2/12 | 57.1% |
| `calendar_oi_funding_friday_asia_long` | weak_research_candidate | calendar_derivatives | -1.5017% / -18.74% / 3.2239% / -5.81 / 3 | 3/0 | 33.3% |
| `legacy_rex_dual_regime_short` | superseded_legacy | rex_legacy | -1.7109% / -21.08% / 3.5433% / -5.95 / 12 | 0/12 | 50.0% |

## 현재 승격 Gross 8

- 성과: **-1.0404% / -13.37% / 2.0359% / -6.56 / 6**
- 방향/승률: 0/6, 16.7%

## 전수성 및 계약 주의

- `results/*alpha*scan*.json` raw scan 산출물 109개는 고정 실행계약이 아니므로 통계에 섞지 않았다.
- PB30 `activity_flow_htf`는 July 전체 통계가 아니라 2020~2023 고정 통계로 재구축했다.
- PB30 quantile-only 파일은 threshold가 완결되지 않아 제외하고, 고정 base/addon module을 각각 재현했다.
- OI selector는 실제 LLM이 아니라 동결된 symbolic ALLOW/BLOCK proxy다.
- legacy 후보의 누락 stride offset은 원 연구 clock과 맞는 `stride-1`로 복구했다. 구형 generic live default 0과는 parity 위험이 있다.
- 캐시 사용 시에도 `asof` 뒤의 미완결/미래 market 및 funding row를 강제 절단한다.
- `legacy_rex_dual_regime_auto` 주말 FX fallback 차이 신호 0개; strict 결과 `-0.4630% / -6.17% / 3.5433% / -1.74 / 14`.
- `legacy_rex_dual_regime_short` 주말 FX fallback 차이 신호 0개; strict 결과 `-1.7109% / -21.08% / 3.5433% / -5.95 / 12`.
- BOCPD/Kalman/Semi-Markov는 2019-12-31부터의 5분봉을 causal warm-up으로 사용하고 July에는 backward-as-of state만 매핑했다.
- 전체 구간은 이미 연구에서 관찰된 retrospective 진단이며 pristine OOS가 아니다.

## 중복

```json
{
  "exact_signal": [
    [
      "oi_divergence_highfreq",
      "oi_divergence_highfreq_selector"
    ]
  ],
  "exact_path": [
    [
      "oi_divergence_highfreq",
      "oi_divergence_highfreq_selector"
    ]
  ],
  "zero_signal": [
    "bocpd_funding_premium_long",
    "fresh_kimchi_fx",
    "frozen_annual_rank7",
    "funding_premium_lr_impact_central",
    "kalman_funding_premium_long",
    "markov_transition_long",
    "new_long_minimal_funding_premium",
    "pb30_base",
    "rex_htf_range_veto",
    "semimarkov_funding_premium_long",
    "short_kimchi3d",
    "short_premium_panic"
  ],
  "zero_path": [
    "bocpd_funding_premium_long",
    "fresh_kimchi_fx",
    "frozen_annual_rank7",
    "funding_premium_lr_impact_central",
    "kalman_funding_premium_long",
    "markov_transition_long",
    "new_long_minimal_funding_premium",
    "pb30_base",
    "rex_htf_range_veto",
    "semimarkov_funding_premium_long",
    "short_kimchi3d",
    "short_premium_panic"
  ]
}
```
