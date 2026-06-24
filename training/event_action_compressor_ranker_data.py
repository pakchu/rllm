"""Convert event-action LLM rows into compressor/ranker feature rows.

The direct LLM selector experiments failed mostly at label/position-prior and
high-utility-tail identification.  This converter keeps the LLM-relevant
symbolic context, but exposes it as a transparent candidate-ranking table:

- ``feature_snapshot``: numeric past-only state plus candidate action fields.
- ``state_tokens``: deterministic coarse descriptors that a single LLM
  compressor can be fine-tuned to emit and a small ranker can consume.
- ``reward``: future utility metadata for training/evaluation only.

No future path field is copied into features or tokens.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventActionCompressorRankerConfig:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    reward_field: str = "rank_utility"


_JSON_LINE_RE = re.compile(r"^(Past-only state|Action book|Candidate action):\s*(.+)$")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_prompt_sections(prompt: str) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    for line in str(prompt).splitlines():
        match = _JSON_LINE_RE.match(line.strip())
        if not match:
            continue
        key, raw = match.groups()
        sections[key] = json.loads(raw)
    missing = {"Past-only state", "Action book", "Candidate action"}.difference(sections)
    if missing:
        raise ValueError(f"prompt missing parseable sections: {sorted(missing)}")
    return sections


def _bucket_signed(value: float, *, small: float, large: float) -> str:
    x = float(value)
    if x <= -large:
        return "strong_down"
    if x <= -small:
        return "down"
    if x < small:
        return "flat"
    if x < large:
        return "up"
    return "strong_up"


def _bucket_abs(value: float, *, small: float, large: float) -> str:
    x = abs(float(value))
    if x < small:
        return "low"
    if x < large:
        return "medium"
    return "high"


def _bucket_range_pos(value: float) -> str:
    x = float(value)
    if x < -0.25:
        return "below_range"
    if x < 0.25:
        return "lower_range"
    if x < 0.75:
        return "mid_range"
    if x < 1.25:
        return "upper_range"
    return "above_range"


def state_tokens(state: dict[str, Any], action: dict[str, Any]) -> dict[str, str]:
    side = str(action.get("side", "NONE")).upper()
    side_sign = 1.0 if side == "LONG" else -1.0 if side == "SHORT" else 0.0
    hold = int(action.get("hold_bars", 0) or 0)
    strength = float(action.get("strength", 0.0) or 0.0)
    return {
        "family": str(action.get("family", "UNKNOWN")),
        "side": side,
        "hold_bucket": "short" if hold <= 72 else "medium" if hold <= 144 else "long" if hold <= 288 else "very_long",
        "strength_bucket": _bucket_abs(strength, small=0.15, large=0.45),
        "trend_24": _bucket_signed(float(state.get("trend_24", 0.0) or 0.0), small=0.003, large=0.012),
        "trend_96": _bucket_signed(float(state.get("trend_96", 0.0) or 0.0), small=0.006, large=0.025),
        "side_trend_24": _bucket_signed(side_sign * float(state.get("trend_24", 0.0) or 0.0), small=0.003, large=0.012),
        "side_trend_96": _bucket_signed(side_sign * float(state.get("trend_96", 0.0) or 0.0), small=0.006, large=0.025),
        "range_location": _bucket_range_pos(float(state.get("range_pos", 0.0) or 0.0)),
        "bb_pressure": _bucket_signed(float(state.get("bb_z", 0.0) or 0.0), small=0.75, large=1.75),
        "rsi_pressure": _bucket_signed(float(state.get("rsi_norm", 0.0) or 0.0), small=0.20, large=0.55),
        "volume_state": _bucket_signed(float(state.get("volume_zscore", 0.0) or 0.0), small=0.75, large=1.75),
        "taker_flow": _bucket_signed(float(state.get("taker_imbalance", 0.0) or 0.0), small=0.025, large=0.08),
        "dxy_pressure": _bucket_signed(float(state.get("dxy_zscore", 0.0) or 0.0), small=0.75, large=1.75),
        "usdkrw_pressure": _bucket_signed(float(state.get("usdkrw_zscore", 0.0) or 0.0), small=0.75, large=1.75),
        "kimchi_level": _bucket_signed(float(state.get("kimchi_premium_zscore", 0.0) or 0.0), small=0.75, large=1.75),
        "kimchi_change": _bucket_signed(float(state.get("kimchi_premium_change", 0.0) or 0.0), small=0.0005, large=0.002),
        "htf_4h": _bucket_signed(float(state.get("htf_4h_return_4", 0.0) or 0.0), small=0.006, large=0.02),
        "htf_1d": _bucket_signed(float(state.get("htf_1d_return_1", 0.0) or 0.0), small=0.01, large=0.04),
        "htf_1w": _bucket_signed(float(state.get("htf_1w_return_4", 0.0) or 0.0), small=0.02, large=0.08),
        "drawdown_state": _bucket_abs(float(state.get("window_drawdown", 0.0) or 0.0), small=0.02, large=0.08),
    }


def feature_snapshot(state: dict[str, Any], action: dict[str, Any]) -> dict[str, float]:
    out = {str(k): float(v or 0.0) for k, v in state.items() if isinstance(v, (int, float))}
    side = str(action.get("side", "NONE")).upper()
    out["action_strength"] = float(action.get("strength", 0.0) or 0.0)
    out["action_hold_bars"] = float(action.get("hold_bars", 0) or 0)
    out["action_hold_norm"] = out["action_hold_bars"] / 432.0
    out["action_side_sign"] = 1.0 if side == "LONG" else -1.0 if side == "SHORT" else 0.0
    return dict(sorted(out.items()))


def compressor_prompt(tokens: dict[str, str]) -> str:
    lines = [
        "Task: compress a BTCUSDT futures candidate into leakage-safe regime tokens.",
        "Use only the provided past-state/action fields. Return compact JSON tokens; do not choose a trade.",
        "Fields:",
    ]
    for key in sorted(tokens):
        lines.append(f"- {key}: {tokens[key]}")
    return "\n".join(lines)


def convert_row(row: dict[str, Any], reward_field: str) -> dict[str, Any]:
    sections = parse_prompt_sections(str(row.get("prompt", "")))
    state = sections["Past-only state"]
    action = dict(row.get("action") or sections["Candidate action"])
    audit = row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}
    reward_value = float(audit.get(reward_field, audit.get("rank_utility", 0.0)) or 0.0)
    tokens = state_tokens(state, action)
    side = str(action.get("side", "NONE")).upper()
    return {
        "task": "event_action_compressor_ranker",
        "date": row.get("date"),
        "signal_pos": int(row.get("signal_pos", -1) or -1),
        "side": side,
        "candidate": {
            "family": action.get("family"),
            "side": side,
            "hold_bars": int(action.get("hold_bars", 0) or 0),
            "strength": float(action.get("strength", 0.0) or 0.0),
        },
        "feature_snapshot": feature_snapshot(state, action),
        "state_tokens": tokens,
        "llm_compressor_prompt": compressor_prompt(tokens),
        "llm_compressor_target": json.dumps(tokens, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "reward": {
            "net_return_pct": reward_value,
            "rank_utility": float(audit.get("rank_utility", 0.0) or 0.0),
            "net_return": float(audit.get("net_return", 0.0) or 0.0),
            "mae": float(audit.get("mae", 0.0) or 0.0),
            "mfe": float(audit.get("mfe", 0.0) or 0.0),
            "reward_field": reward_field,
        },
        "leakage_guard": {
            "features_from_prompt_past_only_state": True,
            "action_metadata_prompt_visible": True,
            "reward_is_label_only": True,
            "does_not_use_target_code_as_feature": True,
        },
    }


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _summary(rows: list[dict[str, Any]], cfg: EventActionCompressorRankerConfig) -> dict[str, Any]:
    rewards = [float(r["reward"]["net_return_pct"]) for r in rows]
    return {
        "config": asdict(cfg),
        "rows": len(rows),
        "signals": len({(r["date"], r["signal_pos"]) for r in rows}),
        "families": dict(sorted(Counter(str(r["candidate"].get("family")) for r in rows).items())),
        "sides": dict(sorted(Counter(str(r["side"]) for r in rows).items())),
        "holds": dict(sorted(Counter(str(r["candidate"].get("hold_bars")) for r in rows).items())),
        "feature_count": len(rows[0]["feature_snapshot"]) if rows else 0,
        "token_count": len(rows[0]["state_tokens"]) if rows else 0,
        "reward": {
            "mean": sum(rewards) / max(1, len(rewards)),
            "positive_frac": sum(1 for x in rewards if x > 0) / max(1, len(rewards)),
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
        },
        "leakage_guard": {
            "features_from_prompt_past_only_state": True,
            "future_reward_label_only": True,
        },
    }


def run(cfg: EventActionCompressorRankerConfig) -> dict[str, Any]:
    rows = [convert_row(row, cfg.reward_field) for row in _read_jsonl(cfg.input_jsonl)]
    _write_jsonl(cfg.output_jsonl, rows)
    summary = _summary(rows, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert event-action LLM rows to compressor/ranker feature rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--reward-field", default=EventActionCompressorRankerConfig.reward_field)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventActionCompressorRankerConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
