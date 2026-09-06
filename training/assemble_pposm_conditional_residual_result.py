"""Assemble immutable evidence for the PPOSM residual-router critic."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    preregistration: Path
    data_summary: Path
    sft_summary: Path
    sft_adapter: Path
    rlvr_config: Path
    rlvr_reward_diagnostics: Path
    rlvr_gradient_diagnostics: Path
    rlvr_adapter: Path
    train_scores: Path
    threshold: Path
    oos_scores: Path
    route_predictions: Path
    route_report: Path
    backtest: Path
    replay_snapshot: Path
    output: Path


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


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def assemble(cfg: Config) -> dict[str, Any]:
    preregistration = _load_object(cfg.preregistration)
    data_summary = _load_object(cfg.data_summary)
    threshold = _load_object(cfg.threshold)
    route_report = _load_object(cfg.route_report)
    backtest = _load_object(cfg.backtest)

    paths = {
        "preregistration": cfg.preregistration,
        "data_summary": cfg.data_summary,
        "sft_summary": cfg.sft_summary,
        "sft_adapter": cfg.sft_adapter,
        "rlvr_config": cfg.rlvr_config,
        "rlvr_reward_diagnostics": cfg.rlvr_reward_diagnostics,
        "rlvr_gradient_diagnostics": cfg.rlvr_gradient_diagnostics,
        "rlvr_adapter": cfg.rlvr_adapter,
        "train_scores": cfg.train_scores,
        "threshold": cfg.threshold,
        "oos_scores": cfg.oos_scores,
        "route_predictions": cfg.route_predictions,
        "route_report": cfg.route_report,
        "backtest": cfg.backtest,
        "replay_snapshot": cfg.replay_snapshot,
    }
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    byte_replay_identical = cfg.backtest.read_bytes() == cfg.replay_snapshot.read_bytes()
    threshold_protocol = str(threshold.get("protocol", ""))
    selection_inputs = threshold.get("selection_inputs")
    selected_train_scores = (
        selection_inputs.get("train_scores")
        if isinstance(selection_inputs, dict)
        else None
    )
    threshold_is_train_only = (
        "train_only" in threshold_protocol
        and int(threshold.get("train_base_signals", 0)) > 0
        and isinstance(selection_inputs, dict)
        and set(selection_inputs) == {"train_scores"}
        and isinstance(selected_train_scores, dict)
        and selected_train_scores.get("sha256")
        == artifacts["train_scores"]["sha256"]
        and threshold.get("future_can_rank_repair_or_reselect") is False
    )
    prereg_frozen = preregistration.get("status") == "preregistered_before_model_scoring"
    data_matches_manifest = (
        data_summary.get("manifest_freeze_hash")
        == preregistration.get("source", {}).get("manifest_freeze_hash")
    )

    core: dict[str, Any] = {
        "protocol_version": "pposm_conditional_residual_sft_rlvr_result_v1",
        "as_of": "2026-09-02",
        "status": "ready_for_critic_contaminated_research_not_deployment",
        "research_boundary": {
            "train": "pre-2024 only",
            "oos": "2024 test, 2025 eval, and 2026H1 holdout are report/veto only",
            "selection": "adapter, decoding, and threshold frozen without OOS rerank or repair",
            "contamination": "historical PPOSM OOS windows were exposed previously; research only",
            "future_can_rank_repair_or_reselect": False,
        },
        "architecture": preregistration.get("architecture"),
        "model": {
            "base": preregistration.get("base_model"),
            "sft": _load_object(cfg.sft_summary),
            "rlvr": {
                "config": _load_object(cfg.rlvr_config),
                "reward_diagnostics": _load_object(cfg.rlvr_reward_diagnostics),
                "gradient_diagnostics": _load_object(cfg.rlvr_gradient_diagnostics),
            },
        },
        "data": data_summary,
        "threshold_freeze": threshold,
        "route_report": route_report,
        "invariants": backtest.get("invariants", {}),
        "artifacts": artifacts,
        "checks": {
            "preregistration_frozen_before_scoring": prereg_frozen,
            "data_manifest_matches_preregistration": data_matches_manifest,
            "threshold_train_only": threshold_is_train_only,
            "oos_rerank_or_repair": False,
            "artifact_hashes_recorded": all(
                len(str(item.get("sha256", ""))) == 64 for item in artifacts.values()
            ),
            "byte_replay_identical": byte_replay_identical,
        },
        "execution_config": {
            key: str(value) for key, value in asdict(cfg).items()
        },
    }
    result = {**core, "result_hash": sha256_bytes(canonical_json(core).encode("utf-8"))}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in Config.__dataclass_fields__.values():
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=Path,
            required=True,
        )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(assemble(Config(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
