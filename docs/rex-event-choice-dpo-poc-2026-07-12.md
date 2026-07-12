# REX Event Choice Preference-DPO POC (2026-07-12)

## 목적
기존 REX 이벤트 LLM 정책은 JSON/action SFT와 equal-label SFT 모두에서 후보 logprob prior가 강하게 생겼다. 특히 `CHOICE_B_SHORT`로 붕괴했고, calibration 없이는 `NO_TRADE`를 거의 선택하지 못했다. 이번 POC는 hard class SFT 대신 **offline path utility margin 기반 preference/DPO**로 바꿔 노이즈 큰 라벨과 약한 효용 차이를 제거할 수 있는지 확인했다.

## 데이터
- Source: `data/rex_event_reasoning_policy_sft_20260712.jsonl`
- Prompt: signal-time symbolic context only
- Future path: offline preference label 생성에만 사용
- Preference builder: `training/build_rex_event_choice_preference_data.py`
- Margin: `chosen_utility - rejected_utility >= 0.004`
- NO_TRADE utility: `0.001`
- Invalid trade filter: `net_return <= 0.001` 또는 `MAE > 0.035`면 해당 trade utility를 NO_TRADE 아래로 제한

### Preference rows
| split | rows |
|---|---:|
| all | 3,631 |
| train | 3,194 |

Train pair counts:
- `LONG>SHORT`: 589
- `SHORT>LONG`: 512
- `NO_TRADE>LONG`: 772
- `NO_TRADE>SHORT`: 868
- `LONG>NO_TRADE`: 250
- `SHORT>NO_TRADE`: 203

## 학습
- Script: `training/train_text_dpo.py`
- Base: `google/gemma-2-2b-it`
- Init adapter: `checkpoints/rex_event_choice_label_gemma2_2b_lora_s32_20260712`
- Output: `checkpoints/rex_event_choice_dpo_gemma2_2b_lora_s32_20260712`
- Sample: 384 balanced, chosen label 128/128/128
- Steps: 32
- Runtime: 128.2s
- Train loss: 0.6912

## Raw candidate-logprob 결과
Evaluation: `results/rex_event_choice_dpo_gemma2_s32_all_mean_2026-07-12.json`

Raw prediction collapsed to SHORT:
- all predictions: `CHOICE_B_SHORT` 1442 / `CHOICE_A_LONG` 2 / `CHOICE_C_SKIP` 0
- all accuracy: 23.96%
- all abs return: -60.60%
- all CAGR: -13.79%
- all strict MDD: 63.54%
- all CAGR/MDD: -0.22
- trade entries: 582

## Train-only prior calibration 결과
Calibration: `results/rex_event_choice_dpo_gemma2_s32_calibrated_2026-07-12.json`

Train mean candidate score:
- LONG: -2.4978
- SHORT: -2.3078
- SKIP: -2.5288

즉 DPO 후에도 SHORT token prior가 매우 강하게 남아 있다.

### Split stats, calibrated
| split | abs return | CAGR | strict MDD | CAGR/MDD | trades | note |
|---|---:|---:|---:|---:|---:|---|
| train | +49.18% | +8.53% | 11.79% | 0.72 | 383 | weak, not significant |
| test 2025 | +2.43% | +2.84% | 4.57% | 0.62 | 38 | fail |
| eval 2026H1 | +0.05% | +0.20% | 4.73% | 0.04 | 21 | fail |

Test/eval의 trade count도 작고 평균 수익 p-value가 높아 통계적으로 의미 없다.

## 결론
Preference-DPO 단독 전환은 실패했다.

핵심 원인:
1. equal label이어도 Gemma2-2B candidate logprob에는 강한 label prior가 남는다.
2. 32-step DPO는 SFT에서 생긴 SHORT collapse를 깨지 못했다.
3. 하나의 3-class 선택으로 gate(NO_TRADE)와 side(LONG/SHORT)를 동시에 학습시키면 skip/side calibration이 엉킨다.
4. target 자체가 future utility oracle이라 label noise가 큰 구간이 많고, margin filtering만으로는 충분하지 않았다.

## 다음 작업
하나의 LLM은 유지하되 출력 과제를 분리한다.

1. Gate head: `TRADE` vs `NO_TRADE`
2. Side head: `LONG` vs `SHORT` only when trade-worthy
3. Train-only calibration은 gate threshold에만 적용
4. Backtest에서는 gate 통과 시에만 side를 적용

이 구조는 analyzer/trader 이중 LLM이 아니라 **single Gemma policy with two sequential decisions**이므로 최근 방향(LLM 하나로 축소)과 맞다.
