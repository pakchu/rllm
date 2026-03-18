"""Model architecture modules."""

from models.option_a import (
    DEFAULT_HF_VISUAL_BACKBONE,
    VISUAL_BACKBONE_PRESETS,
    ChartScalarExtractor,
    FrozenHFChartScalarExtractor,
    build_policy_kwargs,
)
from models.option_b_vlm import (
    AUTO_MODEL_NAME,
    FALLBACK_VLM_MODEL,
    RECOMMENDED_VLM_MODEL,
    TradingPromptState,
    auto_select_vlm_model,
    build_trading_prompt,
    detect_gpu_vram_gb,
    parse_action_label,
    recommended_vlm_models,
)

__all__ = [
    "ChartScalarExtractor",
    "FrozenHFChartScalarExtractor",
    "build_policy_kwargs",
    "VISUAL_BACKBONE_PRESETS",
    "DEFAULT_HF_VISUAL_BACKBONE",
    "TradingPromptState",
    "build_trading_prompt",
    "parse_action_label",
    "recommended_vlm_models",
    "detect_gpu_vram_gb",
    "auto_select_vlm_model",
    "AUTO_MODEL_NAME",
    "RECOMMENDED_VLM_MODEL",
    "FALLBACK_VLM_MODEL",
]
