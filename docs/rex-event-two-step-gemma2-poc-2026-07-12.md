# REX Event Single-LLM Two-Step Gemma2 POC (2026-07-12)

## 목적
3-class REX choice SFT/DPO가 label prior로 붕괴했기 때문에, 하나의 Gemma2 모델 안에서 과제를 분리했다.

- Step 1: `GATE` = `TRADE` vs `NO_TRADE`
- Step 2: `SIDE` = `LONG` vs `SHORT`, gate 통과 시에만 사용

Analyzer/trader 2모델 구조가 아니라 **single LLM, two sequential decisions** 구조다.

## 데이터
- Source: `data/rex_event_reasoning_policy_sft_20260712.jsonl`
- Builder: `training/build_rex_event_two_step_sft_data.py`
- Output:
  - `data/rex_event_two_step_sft_20260712.jsonl`
  - `data/rex_event_two_step_sft_train_20260712.jsonl`
- Prompt: signal-time symbolic context only
- Future path utility: offline target generation only
- Gate margin: 0.004
- Side margin: 0.004

### Label counts
| task | train labels |
|---|---:|
| gate | TRADE 453 / NO_TRADE 806 |
| side | LONG 589 / SHORT 512 |

Train SFT sampling은 balanced로 맞췄다.

## 학습
- Script: `training/train_text_sft.py`
- Base: `google/gemma-2-2b-it`
- Output: `checkpoints/rex_event_two_step_gemma2_2b_lora_s32_20260712`
- Sample: 512 balanced
  - gate 256 / side 256
  - targets LONG/SHORT/TRADE/NO_TRADE each 128
- Steps: 32
- Runtime: 104.3s
- Train loss: 1.719

## Raw 평가
Result: `results/rex_event_two_step_gemma2_s32_raw_2026-07-12.json`

Raw gate는 전부 `NO_TRADE`로 붕괴했다.

| split | abs return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| train | 0.00% | 0.00% | 0.00% | n/a | 0 |
| test 2025 | 0.00% | 0.00% | 0.00% | n/a | 0 |
| eval 2026H1 | 0.00% | 0.00% | 0.00% | n/a | 0 |

## Train-prior calibrated gate
Result: `results/rex_event_two_step_gemma2_s32_gate_calibrated_2026-07-12.json`

Gate prior fitted on train only:
- TRADE: -6.0156
- NO_TRADE: -3.1889

| split | abs return | CAGR | strict MDD | CAGR/MDD | trades | side counts |
|---|---:|---:|---:|---:|---:|---|
| train | +62.25% | +10.49% | 13.49% | 0.78 | 333 | L143/S190 |
| test 2025 | +9.58% | +11.27% | 5.87% | 1.92 | 31 | L13/S18 |
| eval 2026H1 | -2.79% | -10.02% | 4.89% | -2.05 | 16 | L7/S9 |

Test는 좋아졌지만 eval이 실패했다.

## Train-only gate bias sweep
Result: `results/rex_event_two_step_gemma2_s32_gate_bias_sweep_2026-07-12.json`

Bias는 train 기준으로만 선택했다. Best train bias는 `-0.1`.

| split | abs return | CAGR | strict MDD | CAGR/MDD | trades | side counts |
|---|---:|---:|---:|---:|---:|---|
| train | +53.24% | +9.46% | 8.95% | 1.06 | 206 | L98/S108 |
| test 2025 | -0.36% | -0.43% | 5.76% | -0.07 | 20 | L9/S11 |
| eval 2026H1 | -0.18% | -0.68% | 2.02% | -0.33 | 6 | L1/S5 |

Bias 최적화는 train overfit이며 OOS 성능은 사라진다.

## 결론
Two-step 분리는 3-class SHORT collapse보다 구조적으로 낫지만, 현재 REX event symbolic prompt/label로는 아직 live-grade가 아니다.

핵심 관찰:
1. Gate와 side를 분리해도 gate score prior가 매우 강하다.
2. Train calibration은 test 2025 일부 개선을 만들지만 eval 2026H1에서 유지되지 않는다.
3. Bias sweep은 train overfit을 명확히 보여준다.
4. 현재의 offline future utility label은 LLM이 일반화할 충분한 causal rule로 변환되지 못했다.

## 다음 방향
단순 action label SFT/DPO가 아니라, LLM이 강한 **연역적 규칙 검증**으로 전환해야 한다.

후속 후보:
1. LLM output을 action이 아니라 `valid_setup / invalid_setup / failure_mode`로 두고, action은 deterministic rule이 실행.
2. REX event를 더 작은 causal clauses로 분해: pullback quality, reclaim quality, macro stress, crowding, range location.
3. Trade target 대신 실패 원인 태그를 학습시켜 skip-gate를 강화.
4. Side는 LLM 판단이 아니라 REX base side + failure-mode veto로 단순화.
