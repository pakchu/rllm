"""Build causal counterfactual-action data for every frozen PPOSM signal.

Prompts contain only the raw signal-time state and symbolic predicates.  The
post-signal TP4 and TP12 executions are computed independently and retained as
offline metadata; they never affect another signal's eligibility or appear in
the prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_pposm_sft_rlvr_data as frozen
from training import build_pposm_state_router_data as numeric
from training import build_pposm_symbolic_router_data as symbolic
from training import search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade

DEFAULT_MANIFEST = numeric.DEFAULT_MANIFEST
DEFAULT_TRAIN_OUTPUT = Path("data/pposm_counterfactual_action_train_pre2024.jsonl")
DEFAULT_OOS_OUTPUT = Path("data/pposm_counterfactual_action_oos_2024_2026.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("results/pposm_counterfactual_action_data_summary.json")

ACTIONS = ("SKIP", "TP4", "TP12")


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


def signal_identity(window: str, signal_position: int) -> str:
    return f"pposm-counterfactual-action|{window}|{int(signal_position)}"


def causal_predicates(
    features: dict[str, float], thresholds: dict[str, Any]
) -> dict[str, bool]:
    """Resolve only predicates available at the frozen signal boundary."""

    return symbolic.predicates(features, thresholds)


def causal_prompt(
    features: dict[str, float], predicates: dict[str, bool]
) -> str:
    """Render the model input without post-signal execution or label data."""

    return "\n".join(
        (
            "Frozen PPOSM counterfactual action choice.",
            "Return exactly one token: SKIP, TP4, or TP12.",
            f"causal_predicates: {canonical_json(predicates)}",
            f"signal_time_state: {canonical_json(features)}",
            "predicate_priority: capitulation predicates before premium-overheat predicates.",
        )
    )


def best_action(utilities: dict[str, float]) -> str:
    """Choose maximum utility with frozen first-wins tie order."""

    if set(utilities) != set(ACTIONS):
        raise ValueError(f"action utilities must contain exactly {ACTIONS}")
    return max(ACTIONS, key=utilities.__getitem__)


def _take_bps(manifest: dict[str, Any]) -> dict[str, int]:
    spec = manifest["spec"]
    return {
        "TP4": int(spec["capitulation_take_bps"]),
        "TP12": int(spec["normal_take_bps"]),
    }


def counterfactual_trades(
    engine: ExecutionEngine,
    signal_position: int,
    manifest: dict[str, Any],
) -> dict[str, Trade]:
    """Execute TP4 and TP12 independently from the same frozen signal."""

    spec = manifest["spec"]
    trades: dict[str, Trade] = {}
    for action, take_bps in _take_bps(manifest).items():
        trade = engine.trade_at(
            int(signal_position),
            int(spec["side"]),
            int(spec["hold_bars"]),
            take_bps,
            int(spec["stop_bps"]),
        )
        if trade is None:
            raise RuntimeError(
                f"active signal {int(signal_position)} has no executable {action} path"
            )
        trades[action] = trade
    return trades


def action_utilities(
    trades: dict[str, Trade], strategy_cfg: pposm.Config
) -> dict[str, float]:
    """Return exact base-fee/slippage net returns for all three actions."""

    if set(trades) != {"TP4", "TP12"}:
        raise ValueError("counterfactual trades must contain TP4 and TP12")
    return {
        "SKIP": 0.0,
        "TP4": frozen.exact_base_cost_net_factor(trades["TP4"], strategy_cfg) - 1.0,
        "TP12": frozen.exact_base_cost_net_factor(trades["TP12"], strategy_cfg) - 1.0,
    }


def _executable_positions(trades: dict[str, Trade]) -> dict[str, dict[str, int]]:
    return {
        action: {
            "entry_position": int(trade.entry_position),
            "exit_position": int(trade.exit_position),
        }
        for action, trade in trades.items()
    }


def build_row(
    *,
    split: str,
    window: str,
    signal_position: int,
    market: pd.DataFrame,
    state: pd.DataFrame,
    manifest: dict[str, Any],
    strategy_cfg: pposm.Config,
    engine: ExecutionEngine,
) -> dict[str, Any]:
    features = numeric._signal_features(state, signal_position)
    predicates = causal_predicates(features, manifest["state_thresholds"])
    trades = counterfactual_trades(engine, signal_position, manifest)
    utilities = action_utilities(trades, strategy_cfg)
    return {
        "task": "pposm_counterfactual_action",
        "split": split,
        "prompt": causal_prompt(features, predicates),
        "target": best_action(utilities),
        "metadata": {
            "identity": signal_identity(window, signal_position),
            "window": window,
            "signal_position": int(signal_position),
            "signal_time": str(pd.Timestamp(market.iloc[int(signal_position)]["date"])),
            "side": int(manifest["spec"]["side"]),
            "hold_bars": int(manifest["spec"]["hold_bars"]),
            "action_take_profit_bps": _take_bps(manifest),
            "action_utilities": utilities,
            "executable_positions": _executable_positions(trades),
            "entry_rule": "next_5m_open",
            "utility_source": "offline_exact_base_cost_execution",
            "offline_label_only": split == "oos",
            "future_outcome_present_in_prompt": False,
        },
    }


def rows_from_decisions(
    market: pd.DataFrame,
    state: pd.DataFrame,
    active: Sequence[bool],
    manifest: dict[str, Any],
    strategy_cfg: pposm.Config,
    engine: ExecutionEngine,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = numeric.decision_positions(market, active)
    train: list[dict[str, Any]] = []
    oos: list[dict[str, Any]] = []
    for split, window, _, _ in numeric.SPLIT_WINDOWS:
        destination = train if split == "train" else oos
        destination.extend(
            build_row(
                split=split,
                window=window,
                signal_position=signal,
                market=market,
                state=state,
                manifest=manifest,
                strategy_cfg=strategy_cfg,
                engine=engine,
            )
            for signal in positions[window]
        )
    identities = [row["metadata"]["identity"] for row in train + oos]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate active-signal identity")
    if any(pd.Timestamp(row["metadata"]["signal_time"]) >= pd.Timestamp("2024-01-01") for row in train):
        raise RuntimeError("training row is not pre-2024")
    return train, oos


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = frozen.load_frozen_manifest(cfg.manifest)
    market, funding, state, active = numeric.replay_frozen_decisions(
        manifest, strategy_cfg
    )
    engine = ExecutionEngine(
        market,
        funding,
        _execution_config(strategy_cfg, strategy_cfg.leverage),
    )
    train, oos = rows_from_decisions(
        market, state, active, manifest, strategy_cfg, engine
    )
    train_bytes, oos_bytes = _jsonl_bytes(train), _jsonl_bytes(oos)
    for path, payload in ((cfg.train_output, train_bytes), (cfg.oos_output, oos_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    all_rows = train + oos
    positions = numeric.decision_positions(market, active)
    identity_bytes = "\n".join(
        row["metadata"]["identity"] for row in all_rows
    ).encode("utf-8")
    execution_bytes = canonical_json(
        [row["metadata"]["executable_positions"] for row in all_rows]
    ).encode("utf-8")
    summary = {
        "protocol": "pposm_frozen_causal_counterfactual_action_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "action_tie_order": list(ACTIONS),
        "rows": {"train": len(train), "oos": len(oos), "total": len(all_rows)},
        "windows": {window: len(values) for window, values in positions.items()},
        "targets": dict(sorted(Counter(row["target"] for row in all_rows).items())),
        "output_sha256": {"train": _sha256(train_bytes), "oos": _sha256(oos_bytes)},
        "identity_sha256": _sha256(identity_bytes),
        "executable_positions_sha256": _sha256(execution_bytes),
        "causality": {
            "all_active_signals": True,
            "prompt_signal_time_state_only": True,
            "future_outcome_in_prompt": False,
            "counterfactuals_scheduled_independently": True,
            "oos_utilities_are_offline_labels_only": True,
            "train_signal_time_pre2024_only": True,
        },
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
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
