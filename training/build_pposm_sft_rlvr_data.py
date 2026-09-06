"""Build deterministic causal SFT/RLVR data for the frozen PPOSM policy.

The builder does not search, refit, resample, or replace trades.  It replays
the frozen pullback-premium-overheat state machine through 2026-06-02 and
turns each trade in its four disjoint schedules into one binary gate example.
Signal-time features are prompt-visible; post-signal execution is used only
for the offline target and (for training rows only) utility metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import (
    _activation_hash,
    _execution_config,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
)

DEFAULT_MANIFEST = Path(
    "results/pullback_premium_overheat_state_machine_manifest_2026-07-15.json"
)
DEFAULT_TRAIN_OUTPUT = Path("data/pposm_sft_rlvr_train_pre2024.jsonl")
DEFAULT_OOS_OUTPUT = Path("data/pposm_sft_rlvr_oos_2024_2026.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("results/pposm_sft_rlvr_data_summary.json")

# One scheduler invocation per boundary is intentional.  Carrying a position
# across a boundary or scheduling the combined OOS span would change identity.
SPLIT_WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    ("train", "pre_2024", "2020-07-01", "2024-01-01"),
    ("oos", "test_2024", "2024-01-01", "2025-01-01"),
    ("oos", "eval_2025", "2025-01-01", "2026-01-01"),
    ("oos", "holdout_2026", "2026-01-01", "2026-06-02"),
)


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sealed_manifest(opened: dict[str, Any]) -> dict[str, Any]:
    """Recover the immutable object whose freeze hash was committed.

    The selection manifest was later marked ``oos_opened`` in place.  That
    mutable audit marker is outside the originally sealed core, so validation
    must restore its pre-opening value rather than weakening upstream checks.
    """

    sealed = deepcopy(opened)
    sealed["oos_opened"] = False
    sealed.pop("oos_opened_at", None)
    sealed.pop("oos_output", None)
    return sealed


def load_frozen_manifest(path: str | Path) -> tuple[dict[str, Any], pposm.Config]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("frozen PPOSM manifest must be a JSON object")
    execution = manifest.get("frozen_execution_config")
    if not isinstance(execution, dict):
        raise TypeError("frozen PPOSM execution config is absent")
    strategy_cfg = pposm.Config(**execution, manifest_output=str(Path(path)))
    pposm._validate_manifest(strategy_cfg, _sealed_manifest(manifest))
    if manifest.get("spec", {}).get("frozen_champion") != pposm.FROZEN_CHAMPION:
        raise RuntimeError("frozen PPOSM champion differs from implementation")
    if manifest.get("selection_end") != pposm.SELECTION_END:
        raise RuntimeError("frozen PPOSM selection boundary differs")
    return manifest, strategy_cfg


def _state_inputs(
    market: pd.DataFrame, raw_features: pd.DataFrame, strategy_cfg: pposm.Config
) -> tuple[pd.Series, pd.DataFrame, np.ndarray, dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(
        dates, "live_hour_signal_bar", window_size=strategy_cfg.window_size
    )
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    return dates, pposm.state_feature_frame(features), active, base_thresholds


def _validate_pre2024(
    manifest: dict[str, Any], strategy_cfg: pposm.Config
) -> pd.Series:
    market, raw_features, funding, source_hashes = _load_bundle(
        strategy_cfg,
        cutoff=pposm.SELECTION_END,
        premium_tolerance=strategy_cfg.live_premium_tolerance,
    )
    if source_hashes != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefix hashes changed")
    dates, features, active, base_thresholds = _state_inputs(
        market, raw_features, strategy_cfg
    )
    if base_thresholds != manifest["base_thresholds"]:
        raise RuntimeError("pre-2024 base thresholds changed")
    if pposm.feature_hash(features) != manifest["feature_prefix_hash"]:
        raise RuntimeError("pre-2024 state feature hash changed")
    fitted = pposm.fit_state_thresholds(features, dates, active)
    if fitted != manifest["state_thresholds"]:
        raise RuntimeError("pre-2024 state thresholds changed")
    capitulation, overheat = pposm.build_state_masks(
        features, fitted, pposm.FROZEN_CHAMPION["overheat"]
    )
    observed = (
        _activation_hash(active, dates),
        _activation_hash(capitulation, dates),
        _activation_hash(overheat, dates),
    )
    expected = (
        manifest["activation_hash"],
        manifest["capitulation_hash"],
        manifest["overheat_hash"],
    )
    if observed != expected:
        raise RuntimeError("pre-2024 state activation hashes changed")
    engine = ExecutionEngine(market, funding, _execution_config(strategy_cfg, strategy_cfg.leverage))
    trades = pposm.schedule_window(
        engine,
        active,
        capitulation,
        overheat,
        overheat_action=pposm.FROZEN_CHAMPION["action"],
        start="2020-07-01",
        end=pposm.SELECTION_END,
    )
    if _schedule_hash(trades) != manifest["selection_schedule_hashes"]["pre_2024"]:
        raise RuntimeError("pre-2024 frozen schedule hash changed")
    return dates


def replay_frozen_schedules(
    manifest: dict[str, Any], strategy_cfg: pposm.Config
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Trade]]]:
    """Recompute features and the exact four non-overlapping schedules."""

    selection_dates = _validate_pre2024(manifest, strategy_cfg)
    market, raw_features, funding, _ = _load_bundle(
        strategy_cfg,
        cutoff="2026-06-02",
        premium_tolerance=strategy_cfg.live_premium_tolerance,
    )
    dates, features, active, base_thresholds = _state_inputs(
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
    hashes = (
        _activation_hash(active[prefix], dates.loc[prefix].reset_index(drop=True)),
        _activation_hash(capitulation[prefix], dates.loc[prefix].reset_index(drop=True)),
        _activation_hash(overheat[prefix], dates.loc[prefix].reset_index(drop=True)),
    )
    if hashes != (
        manifest["activation_hash"],
        manifest["capitulation_hash"],
        manifest["overheat_hash"],
    ):
        raise RuntimeError("full replay changed pre-2024 state activations")

    engine = ExecutionEngine(market, funding, _execution_config(strategy_cfg, strategy_cfg.leverage))
    schedules = {
        window: pposm.schedule_window(
            engine,
            active,
            capitulation,
            overheat,
            overheat_action=pposm.FROZEN_CHAMPION["action"],
            start=start,
            end=end,
        )
        for _, window, start, end in SPLIT_WINDOWS
    }
    if _schedule_hash(schedules["pre_2024"]) != manifest["selection_schedule_hashes"]["pre_2024"]:
        raise RuntimeError("full replay did not reproduce frozen pre-2024 schedule")
    assert_no_replacement_identity(schedules)
    state = features.copy()
    state["capitulation"] = capitulation
    state["overheat"] = overheat
    return market, state, schedules


def trade_identity(window: str, trade: Trade) -> str:
    return "|".join(
        (
            "pposm",
            window,
            str(trade.signal_position),
            str(trade.entry_position),
            str(trade.exit_position),
            str(trade.side),
            trade.entry_date,
        )
    )


def assert_no_replacement_identity(schedules: dict[str, Sequence[Trade]]) -> None:
    identities: list[str] = []
    positions: list[tuple[int, int]] = []
    for _, window, _, _ in SPLIT_WINDOWS:
        trades = schedules[window]
        identities.extend(trade_identity(window, trade) for trade in trades)
        positions.extend((trade.entry_position, trade.exit_position) for trade in trades)
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate frozen trade identity")
    ordered = sorted(positions)
    if any(right[0] <= left[1] for left, right in pairwise(ordered)):
        raise RuntimeError("frozen split schedules overlap")


def exact_base_cost_net_factor(trade: Trade, strategy_cfg: pposm.Config) -> float:
    """Match ``equity_stats``' exact base fee/slippage compounding."""

    one_side = 1.0 - float(strategy_cfg.leverage) * (
        float(strategy_cfg.fee_rate) + float(strategy_cfg.slippage_rate)
    )
    return float(one_side * trade.price_factor * trade.funding_factor * one_side)


def _scalar_features(state: pd.DataFrame, signal_position: int) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in pposm.FEATURE_QUANTILES:
        value = float(state.iloc[signal_position][name])
        if not np.isfinite(value):
            raise RuntimeError(f"scheduled trade has non-finite signal feature: {name}")
        output[name] = value
    return output


def _policy_state(state: pd.DataFrame, signal_position: int) -> tuple[str, int]:
    row = state.iloc[signal_position]
    if bool(row["capitulation"]):
        return "capitulation", int(pposm.SPEC["capitulation_take_bps"])
    if bool(row["overheat"]):
        raise RuntimeError("skip-state signal appeared in executable schedule")
    return "normal", int(pposm.SPEC["normal_take_bps"])


def causal_prompt(
    *, state_name: str, take_bps: int, features: dict[str, float], manifest: dict[str, Any]
) -> str:
    formula = {
        "capitulation": manifest["spec"]["capitulation"],
        "overheat": "premium_index_change>=premium_index_change_q67 AND rex_576_range_pos>=rex_576_range_pos_q67",
        "priority": manifest["spec"]["state_priority"],
        "thresholds": manifest["state_thresholds"],
    }
    return "\n".join(
        (
            "Frozen PPOSM offline gate. Use only the signal-time state below.",
            "Return exactly one token: TRADE or NO_TRADE.",
            f"signal_features: {canonical_json(features)}",
            f"frozen_state: {state_name}",
            "frozen_action: execute_long",
            f"frozen_take_profit_bps: {take_bps}",
            f"frozen_hold_bars: {int(manifest['spec']['hold_bars'])}",
            f"frozen_formula: {canonical_json(formula)}",
        )
    )


def build_row(
    *,
    split: str,
    window: str,
    trade: Trade,
    market: pd.DataFrame,
    state: pd.DataFrame,
    manifest: dict[str, Any],
    strategy_cfg: pposm.Config,
) -> dict[str, Any]:
    state_name, take_bps = _policy_state(state, trade.signal_position)
    features = _scalar_features(state, trade.signal_position)
    net_factor = exact_base_cost_net_factor(trade, strategy_cfg)
    clock = {
        "signal_position": trade.signal_position,
        "signal_time": str(pd.Timestamp(market.iloc[trade.signal_position]["date"])),
        "entry_position": trade.entry_position,
        "entry_time": str(pd.Timestamp(market.iloc[trade.entry_position]["date"])),
        "exit_position": trade.exit_position,
        "exit_time": str(pd.Timestamp(market.iloc[trade.exit_position]["date"])),
        "entry_rule": "next_5m_open",
    }
    metadata: dict[str, Any] = {
        "identity": trade_identity(window, trade),
        "window": window,
        "state": state_name,
        "side": trade.side,
        "take_profit_bps": take_bps,
        "hold_bars": int(manifest["spec"]["hold_bars"]),
        "clock": clock,
        "utility_available_for_training": split == "train",
        "offline_label_only": split != "train",
        "leakage_guard": "future execution is absent from prompt",
        "net_return": net_factor - 1.0,
    }
    return {
        "task": "pposm_exact_base_cost_gate",
        "split": split,
        "prompt": causal_prompt(
            state_name=state_name,
            take_bps=take_bps,
            features=features,
            manifest=manifest,
        ),
        "target": "TRADE" if net_factor > 1.0 else "NO_TRADE",
        "metadata": metadata,
    }


def rows_from_schedules(
    market: pd.DataFrame,
    state: pd.DataFrame,
    schedules: dict[str, Sequence[Trade]],
    manifest: dict[str, Any],
    strategy_cfg: pposm.Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert_no_replacement_identity(schedules)
    train_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    for split, window, _, _ in SPLIT_WINDOWS:
        destination = train_rows if split == "train" else oos_rows
        destination.extend(
            build_row(
                split=split,
                window=window,
                trade=trade,
                market=market,
                state=state,
                manifest=manifest,
                strategy_cfg=strategy_cfg,
            )
            for trade in schedules[window]
        )
    return train_rows, oos_rows


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def build(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = load_frozen_manifest(cfg.manifest)
    market, state, schedules = replay_frozen_schedules(manifest, strategy_cfg)
    train_rows, oos_rows = rows_from_schedules(
        market, state, schedules, manifest, strategy_cfg
    )
    train_bytes, oos_bytes = _jsonl_bytes(train_rows), _jsonl_bytes(oos_rows)
    _write(cfg.train_output, train_bytes)
    _write(cfg.oos_output, oos_bytes)
    all_rows = train_rows + oos_rows
    summary = {
        "protocol": "pposm_frozen_causal_sft_rlvr_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "rows": {"train": len(train_rows), "oos": len(oos_rows), "total": len(all_rows)},
        "windows": {window: len(schedules[window]) for _, window, _, _ in SPLIT_WINDOWS},
        "targets": dict(sorted(Counter(row["target"] for row in all_rows).items())),
        "schedule_sha256": {
            window: _schedule_hash(schedules[window]) for _, window, _, _ in SPLIT_WINDOWS
        },
        "identity_sha256": sha256_bytes(
            "\n".join(row["metadata"]["identity"] for row in all_rows).encode("utf-8")
        ),
        "output_sha256": {
            "train": sha256_bytes(train_bytes),
            "oos": sha256_bytes(oos_bytes),
        },
        "causality": {
            "prompt_features_available_at_signal_time": True,
            "future_outcome_in_prompt": False,
            "oos_net_return_metadata_is_offline_label_only": True,
            "train_net_return_for_utility_reward": True,
            "entry_clock": "signal boundary then next 5m open",
        },
        "no_replacement": True,
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--oos-output", type=Path, default=DEFAULT_OOS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    summary = build(Config(**vars(parser.parse_args())))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
