# LLM+RL 기반 이미지 인식 트레이딩 봇 구축 가이드

## 프로젝트 취지 및 핵심 철학

본 프로젝트는 **멀티모달 대형 언어 모델(VLM)**과 **강화학습(RL)**을 결합하여, 인간 트레이더처럼 차트 이미지를 "보고" 패턴을 "이해"하며 거래 의사결정을 내리는 에이전트를 개발하는 것을 목표로 합니다.

### 왜 LLM+RL인가?

- **시각적 추론**: VLM은 사전 학습된 시각 인코더와 언어 모델의 결합으로, 차트의 복잡한 시각 패턴(지지/저항선, 헤드앤숄더, 캔들 형태 등)을 이해할 수 있는 잠재력을 가집니다.
- **문맥 통합**: 이미지뿐 아니라 스칼라 특징(포지션, 가격, 변동성)을 텍스트 프롬프트로 자연스럽게 융합하여 전체 상황을 파악할 수 있습니다.
- **강화학습 최적화**: 단순 지도학습이 아닌 PPO 등 RL 알고리즘으로 "수익 극대화"라는 명확한 목표를 향해 정책을 직접 파인튜닝합니다.
- **확장 가능성**: 초기에는 경량 CNN/CLIP으로 빠르게 실험하고, 이후 VLM으로 확장하여 더 높은 표현력과 일반화 성능을 추구할 수 있습니다.

본 README는 이 비전을 실현하기 위한 구체적인 설계 결정, 구현 가이드, 그리고 실험 전략을 제시합니다.

---

## 1. 프로젝트 개요

### 목표
최근 w(기본 96)개의 캔들 차트 이미지와 스칼라 특징(현재 포지션 방향, 미실현 수익률, 윈도우 변동성/추세)을 입력으로 받아 BTCUSDT에 대해 buy, hold, sell 액션을 결정하는 강화학습 에이전트를 개발한다. 롱/숏 모두 허용한다. 정책(policy)은 초기에는 CNN/CLIP 인코더 + SB3 PPO로 학습하고, 필요 시 VLM(+LoRA/TRL)로 확장한다.

### 핵심 아이디어
- **시각적 패턴 인식**: LLM(VLM)이 인간 트레이더처럼 차트의 시각적 패턴(예: 헤드앤숄더, 지지/저항선, 캔들 모양)을 학습하도록 한다.
- **데이터 융합**: 차트 이미지(시각 정보)와 스칼라 특징(수치 정보)을 동시에 처리하여 의사 결정을 내린다.
- **목적 기반 학습**: 수익 극대화 및 위험 최소화(예: 샤프 비율)를 목표로 하는 보상 함수를 통해 LLM을 직접 '파인튜닝'한다.

## 2. 핵심 기술 스택 (권장)

- **데이터 수집**: `pandas`, `yfinance` (주식), `ccxt` (암호화폐)
- **차트 이미지 생성**: `mplfinance` 또는 `plotly` (일관된 스타일의 차트 생성을 위해 중요)
- **RL 환경**: `gymnasium` (커스텀 환경 구축용)
- **RL 학습**: `stable-baselines3` PPO
- **로깅/실험관리**: Weights & Biases, Hydra
- **모델**: CNN/CLIP encoder(초기) → 필요 시 VLM(+LoRA/TRL) 확장
- **코어 라이브러리**: PyTorch

## 3. Step 1: 데이터 준비 및 전처리 (State 정의)

이 단계는 모델이 학습할 '상태'를 정의하는 가장 중요한 과정입니다.

### 원본 데이터 로드
OHLCV (시가, 고가, 저가, 종가, 거래량) 시계열 데이터를 로드합니다.
기본 원천은 Binance 1분봉이며, 초기 사용 타임프레임은 1m/5m/15m 입니다. 지정한 `timeframe`으로 좌폐우개(left-closed, right-open) 규칙으로 집계하며, 마지막 미완성 캔들은 제외합니다. 특히 5m/15m는 1분봉으로부터 집계되므로 경계가 확정되기 전의 마지막 집계 캔들은 항상 미완성으로 간주하여 제외합니다.

### 슬라이딩 윈도우
전체 데이터를 w 길이의 윈도우로 순회합니다.

### 차트 이미지 생성
- 각 윈도우(w개의 캔들)에 대해 `mplfinance` 등을 사용하여 차트 이미지를 생성합니다.
- **[중요]** 모든 이미지는 동일 규격(기본 320×320, 검은 배경, 축/범례 제거)으로 생성합니다. 해상도는 구성으로 조정 가능하며(224×224, 256×256 등), 품질과 연산 비용의 균형을 고려하여 선택합니다.
- **Y축 스케일링**: 가격 패널은 윈도우 내 OHLC와 "가격 패널 오버레이"(예: SMA, 볼린저 밴드 상·하단, 엔벨로프 상·하단)의 최저/최고를 모두 포함하도록 동적 y-리밋(+소량 패딩)을 사용합니다. RSI/MFI 등 오실레이터는 별도 서브패널에 자체 축(예: 0–100)으로 표시합니다.
- **인디케이터(옵션)**: `SMA(5, 20, 60, 120, 200, 300, 600)`, `Bollinger Bands(length=20, sigma=2)`, `Envelopes(length=96, pct=1, 3, 5, 10)`, `RSI(14)`, `MFI(14)` 지원. 볼린저 밴드/엔벨로프는 가격 패널 오버레이로 그리며 y축 범위 계산에 포함합니다. RSI/MFI는 별도 축 서브패널로 구성합니다. 계산은 반드시 t 시점 이전 데이터만 사용하며(미래참조 금지), 필요한 경우 w를 넘어서 사용합니다. 충분한 warm-up이 없는 경우가 생기지 않도록 시점보다 충분히 버퍼해 사용하고 인디케이터를 만든후 그 시점부터 사용합니다. 
- **레이아웃**: 가격 패널(상단) + 오실레이터 패널(하단) 수직 스택(예: 70:30 비율). 모든 샘플에서 레이아웃과 스타일을 고정해 일관성을 유지합니다.
 - **오실레이터 보조선**: RSI/MFI 패널에 5, 10, 20, 50, 80, 90, 95 수평 보조선을 표시합니다.

### 입력 데이터셋 구축
각 스텝 t에 대해 `(image_t, scalars_t)`를 저장합니다.
`image_t`는 w개의 캔들을 그린 이미지(픽셀 배열)이며, `scalars_t`는 다음을 포함합니다.
- `position_side`(-1/0/+1), `unrealized_pnl_pct`
- `range_volatility_pct = (max_high_{t-w+1:t} - min_low_{t-w+1:t}) / ((max_high_{t-w+1:t} + min_low_{t-w+1:t}) / 2)`
- `window_trend_pct = (close_t - close_{t-w+1}) / close_{t-w+1}`

## 4. Step 2: 트레이딩 환경 정의 (Gymnasium Env)

RL 에이전트가 상호작용할 커스텀 환경을 `gymnasium.Env` 클래스를 상속받아 구현합니다.

- `__init__(self)`: 데이터(이미지 리스트, 가격 리스트), 트랜잭션 비용(수수료), 초기 자본금 등을 초기화합니다.
- `reset(self)`: 환경을 초기 상태(보통 데이터의 시작점)로 리셋하고, 첫 번째 state를 반환합니다.
- `step(self, action)`: 에이전트의 action을 받아 실행하고 다음 상태, 보상, 종료 여부를 반환합니다.

### 📈 환경 핵심 요소

#### State ($S$)
- `(chart_image, scalars)` 조합입니다.
- 이미지는 픽셀 텐서, 스칼라는 표준화된 연속값으로 변환합니다: `position_side`, `unrealized_pnl_pct`, `range_volatility_pct`, `window_trend_pct`.

#### Action ($A$)
이산 공간(Discrete Space) `Discrete(3)`:
- `0`: Buy (매수 또는 롱 포지션 진입)
- `1`: Hold (기본: 플랫 유지/청산, 옵션: 포지션 유지)
- `2`: Sell (매도 또는 숏 포지션 진입/청산)

> **참고**: 포지션 관리 로직(예: 이미 'Buy' 상태일 때 'Buy' 금지)을 환경 내에 구현해야 합니다.
본 프로젝트는 롱과 숏 포지션 모두 허용합니다.

**포지션 사이징 규칙**
- 액션은 목표 "방향"만 결정합니다. 포지션 크기는 고정 100%(잔고 기준)로 운용합니다.
- 기본 레버리지는 1x이며, 옵션으로 최대 10x까지 설정 가능합니다.
- 방향 전환 시 기존 포지션을 전량 청산한 뒤, 다음 캔들 오픈에서 반대 방향으로 100% 진입합니다.
- 기본 모드(`hold_action_mode=flat`)에서 Hold는 목표 포지션을 0(플랫)으로 둡니다.
- 호환 모드(`hold_action_mode=maintain`)에서는 Hold가 현재 방향을 유지합니다.

#### Reward ($R$)
보상은 로그 수익 변화율 기반입니다.

- `r_t = log(Equity_{t+1} / Equity_t)` (체결/수수료/슬리피지 반영 후)
- 수수료: 편도 5bp(0.05%)
- 슬리피지: 1bp(0.01%)
- 선택 옵션: `directional_bias_penalty`를 사용하면 에피소드 내 순방향(롱/숏) 쏠림 평균 절대값에 비례한 패널티를 추가해 단방향 붕괴를 완화할 수 있습니다.
- 선택 옵션: `directional_symmetry_prob`를 사용하면 에피소드 단위로 buy/sell 의미를 랜덤 미러링해 방향 불변성을 강화할 수 있습니다.
- 선택 옵션: `action_balance_penalty`를 사용하면 policy buy/sell 누적 불균형에 패널티를 주어 한쪽 액션 고착을 줄일 수 있습니다.

#### 체결/비용
- 액션 시점 `t`에서 주문은 다음 캔들 오픈(`t+1 open`)에 체결합니다.
- 체결가에 슬리피지를 적용한 후, 수수료를 반영하여 포지션/현금 업데이트를 수행합니다.

#### 레버리지/청산
- 기본 레버리지 1x, 최대 레버리지 10x. 마진 부족으로 청산 발생 시, 큰 음의 보상을 부여하고(학습 붕괴 방지 수준) 에피소드를 자산 리셋으로 재시작합니다.

## 5. 모델 아키텍처 옵션

### 옵션 A: CNN/CLIP + SB3 PPO (빠른 프로토타입, 추천 시작점)

#### 개요
- **장점**: 구현 단순, 빠른 학습, VRAM 효율, 디버깅 용이, SB3 생태계
- **단점**: 표현력 제한, 복잡 패턴 이해 어려움, 확장성 낮음
- **권장**: 초기 POC, 빠른 실험, 베이스라인

#### 구성 스니펫
```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch, torch.nn as nn, gym

class ChartCLIPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=576):
        super().__init__(observation_space, features_dim)
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip.requires_grad_(False)  # 동결
        self.scalar_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )
    
    def forward(self, obs):
        img_feat = self.clip(pixel_values=obs['image']).pooler_output  # (B,512)
        scalar_feat = self.scalar_mlp(obs['scalars'])  # (B,64)
        return torch.cat([img_feat, scalar_feat], dim=1)  # (B,576)

policy_kwargs = dict(
    features_extractor_class=ChartCLIPExtractor,
    features_extractor_kwargs=dict(features_dim=576),
    net_arch=dict(pi=[256,128], vf=[256,128]),
    activation_fn=nn.ReLU,
)

model = PPO(
    "MultiInputPolicy", env, policy_kwargs=policy_kwargs,
    n_steps=4096, batch_size=256, n_epochs=10,
    gamma=0.999, gae_lambda=0.95, clip_range=0.2,
    learning_rate=2.5e-4, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, verbose=1, tensorboard_log="./runs/"
)
```

#### 하이퍼파라미터
- n_steps=4096, batch_size=256, n_epochs=10, lr=2.5e-4
- n_envs=8~16(5090 32GB), 이미지 320×320 기본

### 옵션 B: VLM + TRL PPO(LoRA) (메인 목표, 고성능)

#### 개요
- **장점**: 강력한 시각 이해, 멀티모달 문맥, 전이학습, 확장성
- **단점**: VRAM/속도, 복잡 프롬프트, 디버깅 어려움
- **권장**: 옵션 A 검증 후, 최종 프로덕션, 복잡 시장

#### 프롬프트 템플릿
```
You are a trading agent for BTCUSDT.

Chart: [IMAGE] (96 candles, {tf})

State:
- Position: {pos_pct:.2f}% (long>0, short<0, flat=0)
- Last Entry: ${last_entry:.2f} (N/A if flat)
- Window Vol: {vol_pct:.2f}%

Action (ONE token): <BUY|HOLD|SELL>
```

#### 구성 스니펫
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import torch

# 1. VLM 로드
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# 2. 액션 토큰 추가
processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<BUY>", "<HOLD>", "<SELL>"]})
model.resize_token_embeddings(len(processor.tokenizer))

# 3. LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Value Head
model = AutoModelForCausalLMWithValueHead.from_pretrained(model, is_peft_model=True)

# 5. PPO 설정
ppo_config = PPOConfig(
    learning_rate=5e-6, batch_size=8, mini_batch_size=2, ppo_epochs=4,
    gradient_accumulation_steps=4, log_with="wandb"
)
ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=processor.tokenizer)
```

#### 하이퍼파라미터
- lr=5e-6, batch=8, mini_batch=2, ppo_epochs=4
- LoRA r=8, alpha=16, 타깃: q_proj/v_proj
- VRAM ~20GB(bf16, GC on, 320×320)

### 옵션 C: Decision Transformer (오프라인 RL)

#### 개요
- **장점**: 오프라인 데이터 활용, 안정 학습, 베이스라인 용이
- **단점**: 온라인 탐험 불가, 데이터 품질 의존, 분포 외 일반화 제한
- **권장**: 역사 데이터 풍부, 온라인 불가, 모방학습

#### 간략 설명
Transformer로 `(return-to-go, state, action)` 시퀀스 모델링, 지도학습으로 액션 예측. VLM을 state 인코더로 활용 가능.

### 옵션 D: Hybrid (World Model + Planning)

#### 개요
- **장점**: 모델 기반 계획, 샘플 효율, 장기 전략
- **단점**: 극도로 복잡, 대량 리소스, 구현 난이도 극상
- **권장**: 연구 목적, 극한 성능, 충분 리소스

#### 간략 설명
VLM 잠재 인코딩 + World Model(Transformer) 예측 + MCTS 탐색. MuZero 스타일.

---

### 옵션 비교 요약

| 항목 | A (CNN/CLIP) | B (VLM+TRL) | C (DT) | D (Hybrid) |
|------|-------------|------------|--------|-----------|
| **구현** | ⭐ 쉬움 | ⭐⭐⭐ 어려움 | ⭐⭐ 보통 | ⭐⭐⭐⭐⭐ 극상 |
| **속도** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **VRAM** | ~8GB | ~20GB | ~12GB | ~28GB |
| **표현력** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **디버깅** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| **온라인** | ✅ | ✅ | ❌ | ✅ |
| **권장순위** | 1위(시작) | 2위(메인) | 3위(보조) | 4위(연구) |

**결론**: 옵션 A로 시작 → 옵션 B로 확장이 표준 경로. 옵션 C는 오프라인 전용. 옵션 D는 고급 연구.

## 6. Step 4: 학습 프로세스 (PPO 파인튜닝)

Stable-Baselines3(PPO)를 사용하여 이미지를 포함한 관측(space)에 대해 정책을 학습합니다.

### PPO 트레이너 설정
- SB3의 `PPO`를 설정합니다.
- actor/critic 네트워크, optimizer, PPO config (learning rate, 배치 크기 등)를 정의합니다.

### 옵션 B: VLM + TRL 학습 프로세스(개요)
1. 관측으로부터 이미지/스칼라를 받아 프롬프트를 구성하고 VLM에 입력합니다.
2. VLM이 단일 액션 토큰(`<BUY|HOLD|SELL>`)을 생성합니다.
3. 환경에서 보상(로그 수익 변화율)을 계산합니다.
4. TRL의 `PPOTrainer`로 LoRA 파라미터를 업데이트합니다(베이스는 동결).

### 멀티 타임프레임 학습 전략
- 접근 A: `1m`, `5m`, `15m` 각각 별도 학습(독립 실험)으로 진행합니다.
- 각 타임프레임은 별개의 데이터셋/전처리/체크포인트/로깅을 사용합니다.
- 타임프레임 간 샘플을 혼합하거나 같은 배치에서 섞지 않습니다.
- 필요 시 후속 단계에서 공용 정책 또는 타임프레임 토큰화를 고려할 수 있으나, 초기엔 분리 학습으로 재현성과 안정성을 확보합니다.

### 롤아웃(Rollout) 수집
while 루프를 돌며 에피소드를 실행합니다:

1. `gym_env.reset()`으로 환경 초기화.
2. 현재 state(이미지, 스칼라)를 정책 네트워크에 입력합니다.
3. 정책(Actor)이 action과 log_probs를 생성합니다.
4. 가치망(Critic)이 value(상태 가치)를 추정합니다.
5. `gym_env.step(action)`을 실행하여 next_state, reward를 받습니다.
6. 이 모든 경험 (state, action, log_prob, reward, value)을 버퍼에 저장합니다.

### 모델 업데이트 (학습)
1. 버퍼가 일정 크기(예: 2048 스텝)가 되면 학습 스텝을 수행합니다.
2. PPO 알고리즘이 버퍼의 데이터를 기반으로 Advantage와 Returns를 계산합니다.
3. 계산된 값을 바탕으로 Actor(Policy)와 Critic(Value)의 가중치를 업데이트합니다.

이 과정에서 VLM은 "보상을 많이 받는 차트 패턴과 가격"에 대해 "특정 액션"을 출력하도록 학습됩니다.

## 7. Step 5: 백테스팅 및 평가

### 훈련/검증/테스트 분리
시계열 데이터이므로 반드시 시간 순서대로 분리합니다.
- Train: 2020-01-01 ~ 2023-12-31
- Val: 2024-01-01 ~ 2024-06-30
- Test: 2024-07-01 ~ 최신

### Out-of-Sample 테스트
모델이 전혀 보지 못한 기간의 데이터로 백테스팅을 수행합니다.

### 주요 지표
- 누적 수익률 (Cumulative Return)
- 샤프 비율 (Sharpe Ratio)
- 최대 낙폭 (MDD, Max Drawdown)
- 승률 (Win Rate) 및 거래 횟수
- 최소 샤프 (min_sharpe)
  - 정의: `min(Sharpe_cash, Sharpe_underlying)`
  - `Sharpe_cash`: 로그수익률에서 무위험 수익률을 차감한 초과수익 기반 샤프
  - `Sharpe_underlying`: (전략 로그수익률의 평균 − 동일 기간 BTCUSDT 로그수익률의 평균) / 전략 로그수익률의 표준편차

## 8. 주요 챌린지 및 고려사항

### 과적합(Overfitting)
모델이 훈련 데이터의 특정 차트 패턴을 '암기'할 수 있습니다. (가장 큰 위험)
- **대응**: 데이터 증강(Augmentation) - 차트 색상 변경, 약간의 노이즈 추가 등.

### State 표현력
w개의 캔들 이미지가 시장의 모든 정보를 담고 있나요? (거래량, 다른 지표 등) VLM이 가격 정보(last_price)를 무시하고 이미지에만 의존할 수 있습니다.

### 보상 함수 설계
잘못된 보상 함수는 에이전트가 거래를 전혀 하지 않거나(Hold만), 너무 자주 거래(수수료로 손해)하게 만들 수 있습니다.

### 계산 비용
VLM을 RL로 파인튜닝하는 것은 매우 큰 계산 자원(GPU VRAM)을 필요로 합니다.

## 9. 미래참조 방지 규칙

- 이미지와 스칼라 생성은 오직 `[t-w+1, t]` 데이터만 사용합니다. 인디케이터 계산은 `t` 이전의 과거 데이터를 추가로 사용할 수 있습니다(w를 넘어서는 lookback 포함, 최대 600). `t+1` 이후는 금지합니다.
- 집계는 좌폐우개로 그룹화하고, 미완성 캔들은 제외합니다. 특히 1분봉에서 5m/15m로의 집계는 마지막 집계 캔들을 항상 미완성으로 간주하여 제외합니다.
- 액션은 t에 발생하고 체결은 `t+1 open`에서 일어나며, 체결가에 슬리피지(1bp) 적용 후 수수료(5bp/side)를 반영합니다.
- `range_volatility_pct` 등 스칼라는 윈도우 내 극값을 사용하되, `t+1` 정보를 절대 사용하지 않습니다.
- 인디케이터는 t 이전 데이터만 사용하며(최대 600), 미래 입력이 필요한 지표는 제외하거나 해당 구간을 마스킹합니다.

---

이 가이드를 바탕으로 프로젝트의 구체적인 구현을 시작할 수 있습니다.

필요 시 SB3 PPO 학습 스켈레톤, 이미지/스칼라 전처리 모듈, 평가/리포트 스크립트를 추가할 수 있습니다.

## 10. 현재 구현된 빠른 실행 경로 (MVP)

아래 명령은 현재 코드베이스에서 실제 동작합니다.

### 1) 관측 데이터셋 준비 (이미지 + 스칼라)
```bash
python main.py prepare-dataset \
  --window-size 32 \
  --resolution 32 \
  --output /tmp/rllm_obs.npz \
  --cache-dir /tmp/rllm_img_cache
```

### 2) Option A PPO 스모크 학습
```bash
python main.py train-smoke \
  --timesteps 256 \
  --window-size 32 \
  --hold-action-mode flat \
  --flat-hold-penalty 0.001 \
  --directional-bias-penalty 0.001 \
  --directional-symmetry-prob 0.5 \
  --action-balance-penalty 0.001 \
  --open-position-bonus 0.001 \
  --random-reset-start \
  --learning-rate 0.0002 \
  --ent-coef 0.02 \
  --synthetic-regime-amplitude 0.0004 \
  --synthetic-regime-period 720 \
  --save-path /tmp/rllm_model.zip
```

> PPO의 rollout 단위 특성상 실제 로그상의 `total_timesteps`는 요청값보다 크게 집계될 수 있습니다.

CNN 수렴 점검 드라이런:
```bash
python main.py cnn-dryrun \
  --hold-action-mode flat \
  --directional-bias-penalty 0.001 \
  --chunks 4 --chunk-steps 2048
```

### 2-b) Binance 실데이터 학습 (futures/spot)
```bash
python main.py train-real \
  --symbol BTCUSDT \
  --start-date 2025-01-01 \
  --end-date 2025-01-07 \
  --timeframe 1m \
  --market-type futures \
  --timesteps 2048 \
  --window-size 96 \
  --save-path /tmp/rllm_model_real.zip
```

### 2-c) RL-LLM(VLM) 스모크 학습 (GRPO + LoRA)
```bash
python main.py train-vlm-smoke \
  --model-name auto \
  --source synthetic \
  --window-size 96 \
  --resolution 224 \
  --max-samples 256 \
  --max-steps 10 \
  --num-generations 4 \
  --temperature 1.2 \
  --top-p 0.95 \
  --output-dir /tmp/rllm_vlm_grpo
```

실제 학습 전에 빠른 점검:
```bash
python main.py train-vlm-smoke --dry-run --max-samples 32 --output-dir /tmp/rllm_vlm_dry
```

권장 모델 우선순위:
1. `Qwen/Qwen3-VL-8B-Instruct` (최신 기본값)
2. `Qwen/Qwen2.5-VL-7B-Instruct` (호환성 fallback)

`--model-name auto`는 GPU VRAM을 감지해(예: 24GB 이상) 최신/호환 모델을 자동 선택합니다.

실행 시 `[train-vlm-smoke] model=... (VRAM: ... GB)` 로그로 실제 선택 결과를 확인할 수 있습니다.

### 2-d) VLM 평가 + 편향 보정(추론 캘리브레이션)
기본 생성 모드(`decision-mode generate`) 대신, 라벨 우도 기반(`decision-mode likelihood`) 추론과 클래스 bias를 사용해 BUY/SELL 쏠림을 완화할 수 있습니다.

평가 시 액션 점수 저장:
```bash
python main.py eval-vlm \
  --adapter-dir checkpoints/vlm_grpo_best_bias_balanced \
  --source csv \
  --input-csv data/2025-01-01_2025-03-01_e5d95c62df8ce0d9d5aa1e5d43352641.csv.gz \
  --start-date 2025-01-01 --end-date 2025-02-28 \
  --hold-band 0.001 --target-horizon 1 \
  --max-samples 300 --sample-mode balanced --sample-seed 33 \
  --decision-mode likelihood \
  --store-action-scores true \
  --output /tmp/rllm_eval_likelihood_scores.json
```

bias 그리드 탐색:
```bash
python main.py calibrate-vlm-bias \
  --input-report /tmp/rllm_eval_likelihood_scores.json \
  --output /tmp/rllm_bias_calibration.json
```

권장 추론 예시:
```bash
python main.py eval-vlm \
  --adapter-dir checkpoints/vlm_grpo_best_bias_balanced \
  --source csv --input-csv data/2025-01-01_2025-03-01_e5d95c62df8ce0d9d5aa1e5d43352641.csv.gz \
  --start-date 2025-01-01 --end-date 2025-02-28 \
  --hold-band 0.001 --target-horizon 1 \
  --max-samples 300 --sample-mode balanced --sample-seed 33 \
  --decision-mode likelihood \
  --action-bias-buy 0.3 --action-bias-hold -0.4 --action-bias-sell -0.8 \
  --output /tmp/rllm_eval_likelihood_calibrated.json
```

### 3) 백테스트 리포트 생성
```bash
python main.py backtest \
  --source synthetic \
  --model-path /tmp/rllm_model.zip \
  --window-size 32 \
  --use-images auto \
  --hold-action-mode auto \
  --flat-hold-penalty 0.001 \
  --deterministic true \
  --output /tmp/rllm_backtest.json
```

CSV/바이낸스 소스도 동일하게 `--source csv|binance`로 사용 가능합니다.
`--use-images auto`는 체크포인트가 기대하는 입력 shape를 읽어 이미지 렌더링 여부를 자동 결정합니다.
`--hold-action-mode auto`는 체크포인트 메타데이터의 HOLD 의미(flat/maintain)를 자동 적용합니다.
`--debiased-action mirror_scalar`는 스칼라 반전 패스를 함께 사용해 BUY/SELL 로그잇의 방향 편향을 완화합니다.
`--decision-mode score_band`는 `p_buy-p_sell` 점수 임계값 기반(진입/전환 분리)으로 과도한 방향 고착/스위칭을 줄입니다.
`--decision-mode blend_score_band`는 두 모델의 액션 확률을 가중 평균한 뒤 score-band 로직을 적용합니다(앙상블 실행).
`--decision-mode regime_switch`는 가격 기반 레짐 스코어(수익률/EMA 기울기/변동성/드로우다운)로 UP/DOWN 상태를 판별해 모델을 라우팅합니다.
`--regime-score-mode raw|zscore`로 스코어 계산 방식을 바꿀 수 있습니다.
`--deterministic false`로 샘플링 기반 행동 평가도 가능합니다.
`--flat-start-policy prefer_entry`는 deterministic 평가에서 flat 상태의 HOLD를 강제로 BUY/SELL 진입으로 바꿔 점검할 수 있습니다.
`--directional-tie-hold-eps`를 0보다 크게 주면 BUY/SELL 확률이 거의 동률일 때 HOLD로 중립 처리해 방향 고정 편향을 완화합니다.
`--trend-guard hard`는 윈도우 추세 스칼라 기반의 하드 오버라이드(실험용)입니다.

블렌드 스코어밴드 예시(모델 앙상블):
```bash
python main.py backtest \
  --source binance --symbol BTCUSDT --timeframe 5m --market-type futures \
  --start-date 2025-12-01 --end-date 2026-02-28 \
  --model-path checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip \
  --decision-mode blend_score_band \
  --blend-model-a checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip \
  --blend-model-b checkpoints/ppo_option_a_real_exp4_nosym.zip \
  --blend-weight-mode static --blend-weight-a 0.75 \
  --debiased-action mirror_scalar \
  --score-centering ema --score-center-alpha 0.02 \
  --score-entry-threshold 0.005 --score-flip-threshold 0.02 --score-neutral-band 0.001 \
  --output /tmp/rllm_backtest_blend.json
```

레짐 스위치 예시(상승장 모델/하락장 모델 분리):
```bash
python main.py backtest \
  --source binance --symbol BTCUSDT --timeframe 1m --market-type futures \
  --start-date 2025-03-01 --end-date 2025-11-30 \
  --model-path checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip \
  --decision-mode regime_switch \
  --regime-model-up checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip \
  --regime-model-down checkpoints/ppo_option_a_real_exp4_nosym.zip \
  --regime-neutral-policy hold --regime-transition-mode force_align \
  --regime-enter-threshold 0.7 --regime-confirm-bars 3 \
  --output /tmp/rllm_backtest_regime_switch.json
```

여러 시드로 일반화 점검:
```bash
python main.py backtest \
  --source synthetic \
  --model-path /tmp/rllm_model.zip \
  --eval-seeds 101,102,103 \
  --output /tmp/rllm_backtest_multiseed.json
```

Option A 파라미터 탐색(간단 스윕):
```bash
python main.py robust-sweep --max-candidates 8 --timesteps 2048
```

Option A 워크포워드(실거래 구간 정책 선택):
```bash
python main.py optiona-walkforward \
  --model-path checkpoints/ppo_option_a_real_exp4_nosym.zip \
  --source binance --symbol BTCUSDT --timeframe 1m --market-type futures \
  --window-size 96 \
  --flat-hold-penalty 0.001 \
  --output results/optiona_walkforward_exp4.json
```

`optiona-walkforward`는 fold별 검증 구간에서 정책 후보(policy_det / scoreband_active_A / scoreband_safe_D)를 선택하고, 다음 테스트 구간 성능을 리포트합니다.

### 구현된 핵심 모듈
- 데이터 집계/누출방지: `preprocessing/timeframe.py`
- 차트 렌더링/캐시: `preprocessing/chart_generator.py`
- 스칼라/인디케이터: `preprocessing/scalars.py`, `preprocessing/indicators.py`
- Gym 환경: `envs/trading_env.py`
- Option A 모델/학습: `models/option_a.py`, `training/train_sb3.py`
- 평가: `evaluation/metrics.py`, `evaluation/backtest.py`
- 통합 CLI: `main.py`

### 데이터 소스
- `prepare-dataset`는 `--source synthetic|csv|binance` 지원
- `train-real`은 Binance 데이터를 직접 다운로드해 학습 파이프라인에 연결
