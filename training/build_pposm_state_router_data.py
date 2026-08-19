"""Build the causal three-route dataset for the frozen PPOSM policy.

Every active signal decision is represented, including signals suppressed by
the portfolio lifecycle.  Labels are the frozen formula, never future P&L:
capitulation routes to TP4, premium overheat routes to SKIP, and the remainder
routes to TP12.  Capitulation has priority when both masks are true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_sft_rlvr_data as frozen
from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import _load_bundle

DEFAULT_MANIFEST = frozen.DEFAULT_MANIFEST
DEFAULT_TRAIN_OUTPUT = Path("data/pposm_state_router_train_pre2024.jsonl")
DEFAULT_OOS_OUTPUT = Path("data/pposm_state_router_oos_2024_2026.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("results/pposm_state_router_data_summary.json")

SPLIT_WINDOWS = frozen.SPLIT_WINDOWS
ROUTES = ("SKIP", "TP4", "TP12")


@dataclass(frozen=True)
class Config:
    manifest: Path = DEFAULT_MANIFEST
    train_output: Path = DEFAULT_TRAIN_OUTPUT
    oos_output: Path = DEFAULT_OOS_OUTPUT
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def route_label(*, capitulation: bool, overheat: bool) -> str:
    """Apply the frozen state priority without consulting an outcome."""

    if capitulation:
        return "TP4"
    if overheat:
        return "SKIP"
    return "TP12"


def signal_identity(window: str, signal_position: int) -> str:
    return f"pposm-state-router|{window}|{int(signal_position)}"


def frozen_formula(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "priority": list(manifest["spec"]["state_priority"]),
        "capitulation": manifest["spec"]["capitulation"],
        "premium_overheat": (
            "premium_index_change>=premium_index_change_q67 AND "
            "rex_576_range_pos>=rex_576_range_pos_q67"
        ),
        "routes": {
            "capitulation": "TP4",
            "overheat": "SKIP",
            "normal": "TP12",
        },
        "thresholds": manifest["state_thresholds"],
    }


def _signal_features(state: pd.DataFrame, signal_position: int) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in pposm.FEATURE_QUANTILES:
        value = float(state.iloc[signal_position][name])
        if not np.isfinite(value):
            raise RuntimeError(f"active signal has non-finite signal feature: {name}")
        output[name] = value
    return output


def causal_prompt(features: dict[str, float], manifest: dict[str, Any]) -> str:
    """Expose signal-time values and the frozen rule, but no routed state/outcome."""

    return "\n".join(
        (
            "Frozen PPOSM state router. Use only the signal-time features and frozen formula.",
            "Return exactly one token: SKIP, TP4, or TP12.",
            f"signal_features: {canonical_json(features)}",
            f"frozen_formula: {canonical_json(frozen_formula(manifest))}",
        )
    )


def build_row(
    *,
    split: str,
    window: str,
    signal_position: int,
    market: pd.DataFrame,
    state: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    row = state.iloc[int(signal_position)]
    target = route_label(
        capitulation=bool(row["capitulation"]), overheat=bool(row["overheat"])
    )
    signal_time = str(pd.Timestamp(market.iloc[int(signal_position)]["date"]))
    return {
        "task": "pposm_state_router",
        "split": split,
        "prompt": causal_prompt(_signal_features(state, signal_position), manifest),
        "target": target,
        "metadata": {
            "identity": signal_identity(window, signal_position),
            "window": window,
            "signal_position": int(signal_position),
            "signal_time": signal_time,
            "entry_rule": "next_5m_open_if_admitted",
            "hold_bars": int(manifest["spec"]["hold_bars"]),
            "target_source": "frozen_formula_only",
            "future_return_present": False,
        },
    }


def decision_positions(
    market: pd.DataFrame, active: Sequence[bool]
) -> dict[str, tuple[int, ...]]:
    dates = pd.to_datetime(market["date"])
    mask = np.asarray(active, dtype=bool)
    if len(mask) != len(market):
        raise ValueError("active and market must have equal length")
    output: dict[str, tuple[int, ...]] = {}
    for _, window, start, end in SPLIT_WINDOWS:
        period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
        output[window] = tuple(int(value) for value in np.flatnonzero(mask & period))
    return output


def rows_from_decisions(
    market: pd.DataFrame,
    state: pd.DataFrame,
    active: Sequence[bool],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = decision_positions(market, active)
    train: list[dict[str, Any]] = []
    oos: list[dict[str, Any]] = []
    for split, window, _, _ in SPLIT_WINDOWS:
        destination = train if split == "train" else oos
        destination.extend(
            build_row(
                split=split,
                window=window,
                signal_position=signal,
                market=market,
                state=state,
                manifest=manifest,
            )
            for signal in positions[window]
        )
    identities = [row["metadata"]["identity"] for row in train + oos]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate active-signal identity")
    return train, oos


def replay_frozen_decisions(
    manifest: dict[str, Any], strategy_cfg: pposm.Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Replay validated frozen features and return every active decision."""

    selection_dates = frozen._validate_pre2024(manifest, strategy_cfg)
    market, raw_features, funding, _ = _load_bundle(
        strategy_cfg,
        cutoff="2026-06-02",
        premium_tolerance=strategy_cfg.live_premium_tolerance,
    )
    dates, features, active, base_thresholds = frozen._state_inputs(
        market, raw_features, strategy_cfg
    )
    if base_thresholds != manifest["base_thresholds"]:
        raise RuntimeError("full replay changed frozen base thresholds")
    prefix = (dates < pd.Timestamp(pposm.SELECTION_END)).to_numpy(bool)
    if not dates.loc[prefix].reset_index(drop=True).equals(selection_dates.reset_index(drop=True)):
        raise RuntimeError("full replay changed the pre-2024 clock")
    if pposm.feature_hash(features, prefix) != manifest["feature_prefix_hash"]:
        raise RuntimeError("full replay changed pre-2024 feature bytes")
    capitulation, overheat = pposm.build_state_masks(
        features, manifest["state_thresholds"], pposm.FROZEN_CHAMPION["overheat"]
    )
    state = features.copy()
    state["capitulation"] = capitulation
    state["overheat"] = overheat
    return market, funding, state, np.asarray(active, dtype=bool)


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode() for row in rows)


def build(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = frozen.load_frozen_manifest(cfg.manifest)
    market, _, state, active = replay_frozen_decisions(manifest, strategy_cfg)
    train_rows, oos_rows = rows_from_decisions(market, state, active, manifest)
    train_bytes = _jsonl_bytes(train_rows)
    oos_bytes = _jsonl_bytes(oos_rows)
    for path, payload in ((cfg.train_output, train_bytes), (cfg.oos_output, oos_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    all_rows = train_rows + oos_rows
    positions = decision_positions(market, active)
    summary = {
        "protocol": "pposm_frozen_causal_state_router_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "rows": {"train": len(train_rows), "oos": len(oos_rows), "total": len(all_rows)},
        "windows": {name: len(values) for name, values in positions.items()},
        "targets": dict(sorted(Counter(row["target"] for row in all_rows).items())),
        "output_sha256": {
            "train": hashlib.sha256(train_bytes).hexdigest(),
            "oos": hashlib.sha256(oos_bytes).hexdigest(),
        },
        "causality": {
            "all_active_signal_decisions": True,
            "future_return_in_target_or_reward": False,
            "prompt_is_signal_time_only": True,
            "split": "train<2024; oos>=2024 through 2026-06-02 exclusive",
        },
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--oos-output", type=Path, default=DEFAULT_OOS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
