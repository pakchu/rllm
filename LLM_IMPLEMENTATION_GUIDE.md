# LLM 구현 가이드라인 프롬프트

> 본 문서는 AI 코딩 어시스턴트가 rllm 프로젝트를 구현할 때 참조해야 할 핵심 가이드라인입니다.

---

## 🎯 프로젝트 핵심 목표

당신은 **LLM+RL 기반 BTCUSDT 트레이딩 에이전트**를 개발하고 있습니다.

### 핵심 철학
- **멀티모달 VLM**과 **강화학습(PPO)**을 결합하여 차트 이미지를 "보고" 패턴을 "이해"하는 에이전트
- 인간 트레이더처럼 시각적 패턴(지지/저항, 캔들 형태, 인디케이터)을 인식
- "수익 극대화"라는 명확한 목표를 향한 RL 최적화

---

## 🚨 절대 준수 사항 (CRITICAL)

### 미래참조 방지 (Future Leak Prevention)

**절대 금지**:
```python
# ❌ 금지: t+1 이후 데이터 사용
window = data[t-w+1:t+2]  # t+1까지 포함 → 누출!

# ❌ 금지: 미완성 집계 캔들 포함
df_5m = df_1m.resample('5T').agg(...)  # 마지막 캔들 미완성 가능

# ❌ 금지: 인디케이터에 미래 데이터 사용
sma = df['close'].rolling(20).mean()  # 괜찮음
sma_future = df['close'].shift(-1).rolling(20).mean()  # 금지!
```

**필수 규칙**:
1. 이미지/스칼라는 `[t-w+1, t]` 범위만 사용
2. 인디케이터는 `t` 이전 데이터만 (lookback은 w 초과 가능, 최대 600)
3. 1m→5m/15m 집계 시 **마지막 집계 캔들은 항상 제외** (미완성)
4. 액션은 `t`에서 결정, 체결은 `t+1 open`
5. 체결가에 슬리피지(1bp) 적용 후 수수료(5bp/side) 반영

### 데이터 제약사항
```python
# ✅ 올바른 예시
def make_window(data, t, w=96):
    """t 시점 윈도우 생성 (미래참조 없음)"""
    window = data.iloc[t-w+1:t+1]  # [t-w+1, t] inclusive
    return window

def aggregate_timeframe(df_1m, timeframe='5T'):
    """좌폐우개 집계 + 마지막 미완성 캔들 제외"""
    df_agg = df_1m.resample(timeframe, closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 
        'close': 'last', 'volume': 'sum'
    }).dropna()
    
    # 마지막 캔들 제외 (미완성)
    df_agg = df_agg.iloc[:-1]
    return df_agg
```

---

## 📋 핵심 스펙 요약

### 데이터
- **원본**: Binance 1분봉 BTCUSDT (2020-01 ~ 최신)
- **타임프레임**: 1m, 5m, 15m (별도 학습, 혼합 금지)
- **분할**: Train(2020~2023), Val(2024-H1), Test(2024-H2)
- **윈도우**: 96 캔들 (기본)

### 차트 이미지
- **해상도**: 320×320 (기본), 224/256 옵션
- **스타일**: 검은 배경, 축/범례 제거
- **레이아웃**: 가격 패널(70%) + 오실레이터 패널(30%)
- **Y축**: OHLC + SMA + Bollinger + Envelopes 극값 모두 포함 (동적 + 소량 패딩)

### 인디케이터
**가격 패널 오버레이**:
- `SMA(5, 20, 60, 120, 200, 300, 600)`
- `Bollinger Bands(length=20, sigma=2)`
- `Envelopes(length=96, pct=1, 3, 5, 10)`

**오실레이터 서브패널** (0-100 고정 축):
- `RSI(14)`, `MFI(14)`
- 보조선: 5, 10, 20, 50, 80, 90, 95

### 상태(State)
```python
observation = {
    'image': np.array,  # (3, 320, 320), float32, [0,1]
    'scalars': {
        'position_size_pct': float,  # 총자산 대비 %, 롱(+), 숏(-), 플랫(0)
        'last_entry_price': float,   # 마지막 진입가 (플랫이면 0 또는 NaN)
        'range_volatility_pct': float  # (max_high - min_low) / midpoint
    }
}
```

### 액션(Action)
```python
actions = {
    0: 'BUY',   # 롱 진입/추가
    1: 'HOLD',  # 현재 포지션 유지
    2: 'SELL'   # 숏 진입/추가 또는 롱 청산
}
```
- 포지션 크기: 고정 100% (잔고 기준)
- 방향 전환 시: 기존 전량 청산 → 다음 오픈에 100% 진입

### 보상(Reward)
```python
r_t = log(Equity_{t+1} / Equity_t)  # 로그 수익 변화율
# Equity는 체결/수수료/슬리피지 반영 후
```

### 체결/수수료/레버리지
- **체결**: `t`에서 주문 → `t+1 open`에 체결
- **슬리피지**: 1bp (0.01%)
- **수수료**: 편도 5bp (0.05%)
- **레버리지**: 기본 1x, 최대 10x
- **청산**: 마진 부족 시 큰 음의 보상 + 자산 reset

---

## 🏗️ 모델 아키텍처 옵션

### 옵션 A: CNN/CLIP + SB3 PPO (시작점)
```python
# 참조: README.md 라인 108-154
# 특징: 빠름, 단순, VRAM 효율 (~8GB)
# 구현: ChartCLIPExtractor + MultiInputPolicy
# 하이퍼파라미터: n_steps=4096, batch=256, lr=2.5e-4
```

### 옵션 B: VLM + TRL PPO(LoRA) (메인 목표)
```python
# 참조: README.md 라인 156-216
# 특징: 강력, 복잡, VRAM 높음 (~20GB)
# 구현: LLaVA/Qwen2-VL + LoRA + PPOTrainer
# 프롬프트: 고정 템플릿 + 액션 토큰 <BUY|HOLD|SELL>
# 하이퍼파라미터: lr=5e-6, batch=8, LoRA r=8
```

**권장 경로**: A로 베이스라인 → B로 확장

---

## 📐 구현 우선순위 및 체크리스트

### Phase 1: 데이터 파이프라인 (최우선)
- [ ] 1분봉 다운로더 검증 (`downloader/`)
- [ ] 타임프레임 집계기 (1m→5m/15m, 미완성 캔들 제외)
- [ ] 차트 이미지 생성기
  - [ ] 320×320 기본, 검은 배경
  - [ ] 동적 y축 (가격+인디케이터 극값 포함)
  - [ ] SMA(7종), Bollinger, Envelopes 렌더링
  - [ ] RSI/MFI 서브패널 (보조선 포함)
  - [ ] 누출 검증 테스트 작성
- [ ] 스칼라 추출기 (`range_volatility_pct` 포함)

### Phase 2: 환경 구현
- [ ] Gymnasium Env 상속
  - [ ] Observation space: Dict(image, scalars)
  - [ ] Action space: Discrete(3)
  - [ ] `reset()`, `step()`, `render()` 구현
- [ ] 체결 로직
  - [ ] `t+1 open` 체결
  - [ ] 슬리피지(1bp) + 수수료(5bp/side)
- [ ] 포지션 관리
  - [ ] 100% 고정 크기
  - [ ] 롱/숏 전환
  - [ ] 레버리지 (1x~10x)
- [ ] 청산 로직
  - [ ] 마진 체크
  - [ ] 큰 음의 보상
  - [ ] 자산 reset
- [ ] 보상 계산 (로그 수익)

### Phase 3: 옵션 A 구현 (베이스라인)
- [ ] `ChartCLIPExtractor` 구현
  - [ ] CLIP ViT-B/16 로드 (동결)
  - [ ] 스칼라 MLP (3→64)
  - [ ] 융합 벡터 (512+64=576)
- [ ] SB3 `MultiInputPolicy` 설정
- [ ] PPO 트레이너 설정
- [ ] 병렬 환경 (`n_envs=8~16`)
- [ ] 로깅 (Tensorboard/W&B)

### Phase 4: 평가 및 유틸
- [ ] `utils.py` 확장
  - [ ] `log_returns(series)`
  - [ ] `range_volatility_pct(highs, lows)`
  - [ ] `sharpe_ratio_log(log_rets, periods_per_year)`
  - [ ] `min_sharpe(equity, underlying, rf, tf)`
- [ ] 백테스팅 스크립트
- [ ] 메트릭 리포트 (Sharpe, MDD, WinRate, min_sharpe)

### Phase 5: 옵션 B 구현 (선택)
- [ ] VLM 로드 (LLaVA/Qwen2-VL)
- [ ] 액션 토큰 추가
- [ ] LoRA 설정 (r=8, alpha=16)
- [ ] 프롬프트 포매터
- [ ] Value Head (공유 or 분리)
- [ ] TRL `PPOTrainer` 설정

---

## 🎨 코드 스타일 가이드

### 파일 구조
```
rllm/
├── downloader/          # 데이터 다운로드
├── preprocessing/       # 차트 생성, 스칼라 추출
│   ├── chart_generator.py
│   ├── indicators.py
│   └── scalars.py
├── envs/               # Gymnasium Env
│   └── trading_env.py
├── models/             # 모델 아키텍처
│   ├── option_a.py     # CNN/CLIP
│   └── option_b.py     # VLM+TRL
├── training/           # 학습 스크립트
│   ├── train_sb3.py
│   └── train_trl.py
├── evaluation/         # 백테스팅, 메트릭
│   ├── backtest.py
│   └── metrics.py
├── utils.py            # 공통 유틸
├── config/             # Hydra 설정
└── README.md
```

### 함수/클래스 작성 규칙
```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    간결한 한 줄 설명.
    
    Args:
        arg1: 설명
        arg2: 설명
    
    Returns:
        설명
    
    Raises:
        ValueError: 조건
    
    Example:
        >>> function_name(1, 2)
        3
    """
    # 미래참조 방지 검증
    assert t >= w, "insufficient history"
    
    # 구현
    ...
```

### 테스트 필수
```python
# 모든 핵심 함수에 대해:
def test_no_future_leak():
    """미래참조 없음을 검증"""
    window = make_window(data, t=100, w=96)
    assert window.index.max() <= data.index[100]
    assert len(window) == 96
```

---

## 🔍 디버깅 체크포인트

구현 중 다음을 반드시 확인:

1. **인덱싱**: `[t-w+1:t+1]` (inclusive) vs `[t-w+1:t]` (exclusive)
2. **집계**: 좌폐우개 (`closed='left'`) + 마지막 제외
3. **체결**: `t`에 주문 → `t+1` 체결 (실제 env step은 `t+1`에서 보상 계산)
4. **이미지 정규화**: `[0,1]` 또는 `[-1,1]` 일관성
5. **스칼라 표준화**: `(x - mean) / std` 또는 `MinMaxScaler`
6. **보상 스케일**: 로그 수익 → 대략 -0.1 ~ +0.1 범위

---

## 📊 실험 로깅

### W&B 필수 로그
```python
wandb.log({
    'episode_reward': float,
    'sharpe_ratio': float,
    'max_drawdown': float,
    'win_rate': float,
    'min_sharpe': float,
    'avg_trade_duration': int,
    'num_trades': int,
    'liquidations': int,
})
```

### Hydra 설정 구조
```yaml
# config/train.yaml
model:
  type: 'clip'  # or 'vlm'
  freeze_encoder: true
  
data:
  timeframe: '5m'
  window_size: 96
  resolution: 320
  
training:
  n_steps: 4096
  batch_size: 256
  learning_rate: 2.5e-4
```

---

## 💡 구현 팁

### 성능 최적화
1. **이미지 캐싱**: 동일 윈도우 재사용 시 캐시
2. **배치 처리**: 여러 환경 병렬화 (`VecEnv`)
3. **혼합정밀도**: `torch.cuda.amp` 또는 `bfloat16`

### 안정성
1. **Gradient Clipping**: `max_grad_norm=0.5`
2. **Reward Clipping**: 극단값 제한 (optional)
3. **Learning Rate Scheduler**: Warm-up + Cosine Decay

### 재현성
```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## 🚀 개발 워크플로우

```bash
# 1. 데이터 준비
python -m downloader.download --symbol BTCUSDT --start 2020-01-01

# 2. 차트 생성 테스트
python -m preprocessing.chart_generator --test

# 3. 환경 검증
python -m envs.trading_env --test

# 4. 옵션 A 학습
python -m training.train_sb3 config=train_1m.yaml

# 5. 백테스트
python -m evaluation.backtest --checkpoint checkpoints/best_model.zip

# 6. 메트릭 리포트
python -m evaluation.metrics --results results/backtest.json
```

---

## 📚 참조 문서

- **메인 README**: `/home/pakchu/rllm/README.md`
- **옵션 A 스니펫**: README 라인 108-154
- **옵션 B 스니펫**: README 라인 156-216
- **미래참조 규칙**: README 라인 162-168
- **평가 지표**: README 라인 150-158

---

## ⚠️ 최종 점검

코드를 커밋하기 전 다음을 확인:

- [ ] 모든 함수에 docstring 존재
- [ ] 미래참조 방지 테스트 통과
- [ ] 타입 힌트 추가됨
- [ ] 로깅 구현됨
- [ ] 설정 파일 업데이트됨
- [ ] README 또는 문서 업데이트됨

**Remember**: "미래참조 없음"이 최우선. 의심스러우면 항상 재확인!

---

이 가이드라인을 따라 rllm 프로젝트를 단계별로 완성하세요. 궁금한 점은 README.md를 참조하거나 명확히 질문하세요.
