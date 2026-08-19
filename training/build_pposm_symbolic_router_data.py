"""Build causal symbolic-predicate PPOSM state-router SFT/RLVR data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_pposm_state_router_data as numeric


@dataclass(frozen=True)
class Config:
    manifest: Path = numeric.DEFAULT_MANIFEST
    train_output: Path = Path("data/pposm_symbolic_router_train_pre2024.jsonl")
    oos_output: Path = Path("data/pposm_symbolic_router_oos_2024_2026.jsonl")
    summary_output: Path = Path("results/pposm_symbolic_router_data_summary.json")


def predicates(features: dict[str, float], thresholds: dict[str, Any]) -> dict[str, bool]:
    return {
        "week_low": features["htf_1w_return_1"] <= thresholds["htf_1w_return_1_q50"],
        "range_wide": features["rex_576_range_width_pct"] >= thresholds["rex_576_range_width_pct_q50"],
        "quote_dry": features["quote_vol_z_1d"] <= thresholds["quote_vol_z_1d_q20"],
        "premium_hot": features["premium_index_change"] >= thresholds["premium_index_change_q67"],
        "range_high": features["rex_576_range_pos"] >= thresholds["rex_576_range_pos_q67"],
    }


def symbolic_prompt(values: dict[str, bool]) -> str:
    return "\n".join(
        (
            "Frozen PPOSM symbolic state router.",
            "Return exactly one token: SKIP, TP4, or TP12.",
            f"causal_predicates: {json.dumps(values, sort_keys=True, separators=(',', ':'))}",
            "frozen_rule: if week_low AND (range_wide OR quote_dry), return TP4; "
            "else if premium_hot AND range_high, return SKIP; else return TP12.",
            "priority: capitulation before premium_overheat before normal.",
        )
    )


def build_rows(
    market: pd.DataFrame,
    state: pd.DataFrame,
    active,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = numeric.decision_positions(market, active)
    train: list[dict[str, Any]] = []
    oos: list[dict[str, Any]] = []
    for split, window, _, _ in numeric.SPLIT_WINDOWS:
        destination = train if split == "train" else oos
        for signal in positions[window]:
            row = state.iloc[signal]
            raw = numeric._signal_features(state, signal)
            target = numeric.route_label(
                capitulation=bool(row["capitulation"]), overheat=bool(row["overheat"])
            )
            destination.append(
                {
                    "task": "pposm_symbolic_state_router",
                    "split": split,
                    "prompt": symbolic_prompt(predicates(raw, manifest["state_thresholds"])),
                    "target": target,
                    "metadata": {
                        "identity": numeric.signal_identity(window, signal),
                        "window": window,
                        "signal_position": signal,
                        "signal_time": str(pd.Timestamp(market.iloc[signal]["date"])),
                        "target_source": "frozen_boolean_formula_only",
                        "future_return_present": False,
                    },
                }
            )
    return train, oos


def _bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def build(cfg: Config) -> dict[str, Any]:
    manifest, strategy_cfg = numeric.frozen.load_frozen_manifest(cfg.manifest)
    market, _, state, active = numeric.replay_frozen_decisions(manifest, strategy_cfg)
    train, oos = build_rows(market, state, active, manifest)
    train_bytes, oos_bytes = _bytes(train), _bytes(oos)
    for path, content in ((cfg.train_output, train_bytes), (cfg.oos_output, oos_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    report = {
        "protocol": "pposm_causal_symbolic_router_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "rows": {"train": len(train), "oos": len(oos)},
        "targets": {
            "train": dict(sorted(Counter(row["target"] for row in train).items())),
            "oos": dict(sorted(Counter(row["target"] for row in oos).items())),
        },
        "sha256": {
            "train": hashlib.sha256(train_bytes).hexdigest(),
            "oos": hashlib.sha256(oos_bytes).hexdigest(),
        },
        "causality": {"predicates_signal_time_only": True, "future_return_used": False},
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Config.manifest)
    parser.add_argument("--train-output", type=Path, default=Config.train_output)
    parser.add_argument("--oos-output", type=Path, default=Config.oos_output)
    parser.add_argument("--summary-output", type=Path, default=Config.summary_output)
    print(json.dumps(build(Config(**vars(parser.parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
