"""Option B (VLM) helpers: model selection, prompting, action parsing."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable


RECOMMENDED_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ACTION_LABELS = ("BUY", "HOLD", "SELL")
AUTO_MODEL_NAME = "auto"
TRADING_ACTION_SYSTEM_PROMPT = (
    "You are a BTCUSDT trading policy. "
    "Reply with exactly one uppercase token: BUY, HOLD, or SELL. "
    "Do not include any other words, punctuation, or explanation."
)


@dataclass(frozen=True)
class TradingPromptState:
    timeframe: str
    position_size_pct: float
    last_entry_price: float
    range_volatility_pct: float
    regime_label: str = "RANGE"
    volatility_label: str = "MID"
    momentum_label: str = "NEUTRAL"
    trend_strength_label: str = "WEAK"
    extra_numeric_features: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    extra_symbolic_features: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    context_tags: tuple[str, ...] = field(default_factory=tuple)


def recommended_vlm_models() -> list[str]:
    """
    Ordered model candidates for RL-VLM stage.

    1) Qwen3-VL-8B-Instruct: latest default.
    2) Qwen2.5-VL-7B-Instruct: compatibility fallback.
    """
    return [RECOMMENDED_VLM_MODEL, FALLBACK_VLM_MODEL]


def detect_gpu_vram_gb() -> float | None:
    """Best-effort VRAM detection via nvidia-smi (returns GB)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        first = out.strip().splitlines()[0].strip()
        mib = float(first)
        return mib / 1024.0
    except Exception:
        return None


def auto_select_vlm_model(prefer_latest: bool = True) -> str:
    """
    Select model based on GPU VRAM and preference.

    Current policy:
      - >=24GB: latest (Qwen3-VL-8B) if prefer_latest else fallback
      - otherwise: fallback (Qwen2.5-VL-7B)
    """
    vram_gb = detect_gpu_vram_gb()
    if vram_gb is None:
        return RECOMMENDED_VLM_MODEL if prefer_latest else FALLBACK_VLM_MODEL
    if vram_gb >= 24:
        return RECOMMENDED_VLM_MODEL if prefer_latest else FALLBACK_VLM_MODEL
    return FALLBACK_VLM_MODEL


def _format_numeric_value(label: str, value: float) -> str:
    label_key = label.lower()
    if "ratio" in label_key or "position" in label_key or "z-score" in label_key or "drawdown" in label_key:
        return f"{label}: {value:.3f}"
    if "rsi" in label_key or "mfi" in label_key:
        return f"{label}: {value:.1f}"
    if "%" in label or "momentum" in label_key or "volatility" in label_key:
        return f"{label}: {value:.4f}"
    if "price" in label_key:
        return f"{label}: {value:.6f}"
    return f"{label}: {value:.4f}"


def build_trading_prompt(state: TradingPromptState, prompt_style: str = "numeric") -> str:
    """Build a strict action-only prompt for VLM policy inference."""
    style = str(prompt_style).strip().lower()
    if style not in {"numeric", "symbolic", "hybrid"}:
        raise ValueError(
            "prompt_style must be one of {'numeric','symbolic','hybrid'}, "
            f"got {prompt_style}"
        )

    numeric_lines = [
        _format_numeric_value("Position Size (%)", float(state.position_size_pct)),
        _format_numeric_value("Last Entry Price", float(state.last_entry_price)),
        _format_numeric_value("Window Volatility (%)", float(state.range_volatility_pct)),
    ]
    numeric_lines.extend(
        _format_numeric_value(label, float(value))
        for label, value in state.extra_numeric_features
    )
    numeric_block = "\n".join(numeric_lines)

    symbolic_lines = [
        f"Regime: {state.regime_label}",
        f"Volatility Level: {state.volatility_label}",
        f"Momentum: {state.momentum_label}",
        f"Trend Strength: {state.trend_strength_label}",
    ]
    symbolic_lines.extend(
        f"{label}: {value}" for label, value in state.extra_symbolic_features
    )
    if state.context_tags:
        symbolic_lines.append("Tags: " + " | ".join(state.context_tags))
    symbolic_block = "\n".join(symbolic_lines)

    if style == "numeric":
        feature_block = numeric_block
    elif style == "symbolic":
        feature_block = symbolic_block
    else:
        feature_block = f"{numeric_block}\n{symbolic_block}"

    return (
        f"Timeframe: {state.timeframe}\n"
        "Chart: [IMAGE]\n"
        f"{feature_block}\n\n"
        "Output format: one uppercase token only (BUY/HOLD/SELL).\n"
        "Answer:"
    )


def parse_action_label(text: str, default: str = "HOLD") -> str:
    """
    Parse action label from arbitrary model output.

    Uses whole-token regex matching and prefers the last matched token,
    which is more robust when prompts contain "BUY/HOLD/SELL" instruction text.
    """
    upper = text.upper()
    matches = re.findall(r"\b(BUY|HOLD|SELL)\b", upper)
    if matches:
        return matches[-1]
    return default


def action_to_id(action_label: str) -> int:
    """Map BUY/HOLD/SELL -> 0/1/2."""
    mapping = {"BUY": 0, "HOLD": 1, "SELL": 2}
    return mapping.get(action_label.upper(), 1)


def id_to_action(action_id: int) -> str:
    """Map action id 0/1/2 -> BUY/HOLD/SELL."""
    mapping = {0: "BUY", 1: "HOLD", 2: "SELL"}
    return mapping.get(int(action_id), "HOLD")


def validate_action_labels(labels: Iterable[str]) -> bool:
    """Check labels are a subset of BUY/HOLD/SELL."""
    valid = set(ACTION_LABELS)
    return all(label in valid for label in labels)
