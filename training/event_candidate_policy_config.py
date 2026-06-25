"""Load auditable event-candidate policy presets.

Policy JSON files keep the current research candidate separate from generated
backtest artifacts.  The loader converts the preset into the existing
walk-forward config without silently enabling live trading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.event_candidate_pairwise_walkforward import EventCandidatePairwiseWalkForwardCfg

DEFAULT_POLICY_PATH = Path("configs/policies/current_event_candidate_policy.json")


def load_policy_preset(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    preset_path = Path(path)
    data = json.loads(preset_path.read_text())
    required = {"name", "status", "live_ready", "data", "walk_forward"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"policy preset missing required keys: {missing}")
    if bool(data.get("live_ready")):
        raise ValueError("current event-candidate preset must not be marked live_ready without a separate live-readiness audit")
    return data


def build_walk_forward_cfg(
    preset: dict[str, Any],
    *,
    output: str,
    work_dir: str,
    input_jsonl: str | None = None,
    market_csv: str | None = None,
) -> EventCandidatePairwiseWalkForwardCfg:
    data = dict(preset["data"])
    params = dict(preset["walk_forward"])
    return EventCandidatePairwiseWalkForwardCfg(
        input_jsonl=input_jsonl or str(data["input_jsonl"]),
        market_csv=market_csv or str(data["market_csv"]),
        output=output,
        work_dir=work_dir,
        **params,
    )
