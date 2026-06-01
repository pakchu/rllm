"""Option B (VLM) helpers: model selection, prompting, action parsing."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable


GEMMA4_E4B_IT_MODEL = "google/gemma-4-E4B-it"
QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
RECOMMENDED_VLM_MODEL = GEMMA4_E4B_IT_MODEL
ACTION_LABELS = ("BUY", "HOLD", "SELL")
ACTIONS_TRADE_GATE = ("TRADE", "NO_TRADE")
ACTIONS_TRADE_SIDE = ("LONG", "SHORT")
ACTION_SCHEMA_LABELS = {
    "buy_hold_sell": ACTION_LABELS,
    "trade_gate": ACTIONS_TRADE_GATE,
    "trade_side": ACTIONS_TRADE_SIDE,
}
ACTION_SCHEMA_DEFAULTS = {
    "buy_hold_sell": "HOLD",
    "trade_gate": "NO_TRADE",
    "trade_side": "LONG",
}
AUTO_MODEL_NAME = "auto"


def get_action_labels(action_schema: str = "buy_hold_sell") -> tuple[str, ...]:
    """Resolve action label set for the selected schema."""
    key = str(action_schema).strip().lower()
    if key not in ACTION_SCHEMA_LABELS:
        raise ValueError(
            "action_schema must be one of "
            f"{sorted(ACTION_SCHEMA_LABELS)}, got {action_schema}"
        )
    return ACTION_SCHEMA_LABELS[key]


def get_default_action_label(action_schema: str = "buy_hold_sell") -> str:
    """Default fallback label for the selected action schema."""
    key = str(action_schema).strip().lower()
    if key not in ACTION_SCHEMA_DEFAULTS:
        raise ValueError(
            "action_schema must be one of "
            f"{sorted(ACTION_SCHEMA_DEFAULTS)}, got {action_schema}"
        )
    return ACTION_SCHEMA_DEFAULTS[key]


def make_action_system_prompt(action_schema: str = "buy_hold_sell") -> str:
    """Strict system prompt for the selected action schema."""
    labels = get_action_labels(action_schema)
    labels_text = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 1 else labels[0]
    return (
        "You are a BTCUSDT trading policy. "
        f"Reply with exactly one uppercase token: {labels_text}. "
        "Do not include any other words, punctuation, or explanation."
    )


TRADING_ACTION_SYSTEM_PROMPT = make_action_system_prompt("buy_hold_sell")


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

    1) google/gemma-4-E4B-it: preferred small Gemma 4 instruction model.
    2) Qwen/Qwen3-VL-8B-Instruct: vision-language compatibility alternative.
    3) Qwen/Qwen2.5-VL-7B-Instruct: compatibility fallback.
    """
    return [RECOMMENDED_VLM_MODEL, QWEN3_VL_MODEL, FALLBACK_VLM_MODEL]


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
      - prefer_latest=True and >=12GB VRAM: Gemma 4 E4B instruction model
      - prefer_latest=True below 12GB: Qwen2.5-VL fallback
      - prefer_latest=False: Qwen2.5-VL fallback for legacy reproducibility
    """
    vram_gb = detect_gpu_vram_gb()
    if not prefer_latest:
        return FALLBACK_VLM_MODEL
    if vram_gb is None or vram_gb >= 12:
        return RECOMMENDED_VLM_MODEL
    return FALLBACK_VLM_MODEL


def resolve_vlm_model_alias(model_name: str, *, prefer_latest: bool = True) -> str:
    """Resolve friendly model aliases to concrete Hugging Face ids."""
    key = str(model_name or "").strip()
    low = key.lower()
    if low in {AUTO_MODEL_NAME, ""}:
        return auto_select_vlm_model(prefer_latest=prefer_latest)
    aliases = {
        "gemma4": GEMMA4_E4B_IT_MODEL,
        "gemma-4": GEMMA4_E4B_IT_MODEL,
        "gemma4-e4b": GEMMA4_E4B_IT_MODEL,
        "gemma-4-e4b": GEMMA4_E4B_IT_MODEL,
        "gemma4-e4b-it": GEMMA4_E4B_IT_MODEL,
        "gemma-4-e4b-it": GEMMA4_E4B_IT_MODEL,
        "qwen3-vl": QWEN3_VL_MODEL,
        "qwen2.5-vl": FALLBACK_VLM_MODEL,
    }
    return aliases.get(low, key)


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


def build_trading_prompt(
    state: TradingPromptState,
    prompt_style: str = "numeric",
    action_schema: str = "buy_hold_sell",
    modality: str = "multimodal",
) -> str:
    """Build a strict action-only prompt for VLM policy inference."""
    style = str(prompt_style).strip().lower()
    if style not in {"numeric", "symbolic", "hybrid"}:
        raise ValueError(
            "prompt_style must be one of {'numeric','symbolic','hybrid'}, "
            f"got {prompt_style}"
        )
    action_labels = get_action_labels(action_schema)
    action_text = "/".join(action_labels)
    modality_key = str(modality).strip().lower()
    if modality_key not in {"multimodal", "text_only"}:
        raise ValueError(
            "modality must be one of {'multimodal','text_only'}, "
            f"got {modality}"
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

    chart_line = "Chart: [IMAGE]\n" if modality_key == "multimodal" else ""
    return (
        f"Timeframe: {state.timeframe}\n"
        f"{chart_line}"
        f"{feature_block}\n\n"
        f"Output format: one uppercase token only ({action_text}).\n"
        "Answer:"
    )


def parse_action_label(
    text: str,
    default: str | None = None,
    labels: Iterable[str] = ACTION_LABELS,
) -> str:
    """
    Parse action label from arbitrary model output.

    Uses whole-token regex matching across the supplied label set and
    prefers the last matched token, which is more robust when prompts
    contain instruction text mentioning the label choices.
    """
    label_tuple = tuple(str(label).upper() for label in labels)
    if not label_tuple:
        raise ValueError("labels must not be empty")
    upper = text.upper()
    pattern = r"(?<![A-Z0-9_])(" + "|".join(
        re.escape(label) for label in sorted(label_tuple, key=len, reverse=True)
    ) + r")(?![A-Z0-9_])"
    matches = re.findall(pattern, upper)
    if matches:
        return matches[-1]
    if default is None:
        return label_tuple[0]
    return str(default).upper()


def action_to_id(action_label: str, labels: Iterable[str] = ACTION_LABELS) -> int:
    """Map action label to its index within the selected label set."""
    label_tuple = tuple(str(label).upper() for label in labels)
    mapping = {label: i for i, label in enumerate(label_tuple)}
    default = mapping.get(label_tuple[0], 0)
    return mapping.get(action_label.upper(), default)


def id_to_action(action_id: int, labels: Iterable[str] = ACTION_LABELS) -> str:
    """Map action id to label within the selected label set."""
    label_tuple = tuple(str(label).upper() for label in labels)
    idx = int(action_id)
    if 0 <= idx < len(label_tuple):
        return label_tuple[idx]
    return label_tuple[0]


def validate_action_labels(
    labels: Iterable[str],
    *,
    action_schema: str = "buy_hold_sell",
) -> bool:
    """Check labels are a subset of the selected schema labels."""
    valid = set(get_action_labels(action_schema))
    return all(str(label).upper() in valid for label in labels)
