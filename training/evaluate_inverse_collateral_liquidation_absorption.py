"""Write-once staged evaluator for the frozen ICLA-60 absorption alpha."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training.preregister_inverse_collateral_liquidation_absorption import (
    MIN_ABSOLUTE_LIQUIDATION_IMBALANCE,
    Config as PreregistrationConfig,
    SPLITS,
    build_clocks,
    canonical_hash,
    derive_wave_state,
    load_sources,
)


SUPPORT_COMMIT = "1ca72c92c373c2469821f1652b97a217675bb131"
EXECUTION_SOURCE_COMMIT = "1af8b80207fe8d3290c0890837daf9201576ccc3"
SUPPORT_MANIFEST_HASH = (
    "22b98432da41ca69e01b3c37d2ca2903f959cb2f24a37262d7d248b3e16a0712"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_inverse_collateral_liquidation_absorption.py"
)
EVALUATOR_FREEZE = Path(
    "results/inverse_collateral_liquidation_absorption_evaluator_freeze_2026-07-19.json"
)
SUPPORT_RESULT = Path(
    "results/inverse_collateral_liquidation_absorption_support_2026-07-19.json"
)
EXECUTION_MANIFEST = Path("results/clbr_execution_sources_2023_2024_manifest.json")
COMBINED_CLOCKS = Path(
    "data/inverse_collateral_liquidation_absorption_clocks_2023_2024.csv.gz"
)
SPLIT_CLOCK_DIR = Path(
    "data/inverse_collateral_liquidation_absorption_clocks_split_2023_2024"
)
CONTROL_CLOCK_DIR = Path(
    "data/inverse_collateral_liquidation_absorption_controls_split_2023_2024"
)
RESULT_PATHS = {
    stage: Path(
        f"results/inverse_collateral_liquidation_absorption_{stage}_2026-07-19.json"
    )
    for stage in SPLITS
}
STATIC_INPUT_SHA256 = {
    "training/preregister_inverse_collateral_liquidation_absorption.py": (
        "8b3e8fe650b0289fe899f399218b008d8685c10adbb8107352901dc43d9f28bb"
    ),
    "docs/inverse-collateral-liquidation-absorption-preregistration-2026-07-19.md": (
        "e1f6cf6fcb06c34d060294cb3e5602b4c2b14da3b3277691aa60386ea40679bb"
    ),
    "tests/test_preregister_inverse_collateral_liquidation_absorption.py": (
        "e9cd6ecb74b4da0569819d958b0e2096b5878dfc98d58c17c2235d9f63b8f2e2"
    ),
    "tests/test_inverse_collateral_liquidation_absorption_support_artifact.py": (
        "aa0a4c000e3f4766e422cc602853ac189d5d465a031b36c0d53dcb4aa311d158"
    ),
    str(COMBINED_CLOCKS): (
        "a55c23a7a0c296b98bb7a8958f713548c4313c0c682f1693c8f8be80b70dd053"
    ),
    str(SUPPORT_RESULT): (
        "3cd925b2028c9b685c980fe4a4accfb0f58cbcad76c0498bb786771828354636"
    ),
    "training/build_clbr_execution_sources.py": (
        "08cab3889241fa935097e8847a7335d2382100a4de9cda26bfdbef44e43905f5"
    ),
    "docs/clbr-execution-source-contract-2026-07-19.md": (
        "d1e5874277b4e0bd54dde923a0913e6bb152b0128b038fd1dc880174b2aa8e65"
    ),
    "tests/test_build_clbr_execution_sources.py": (
        "a8005ef36056ea73c1c1a2f2cf789eec5d3a28b314be628abf919f2d44f26c9d"
    ),
    "tests/test_clbr_execution_sources_artifact.py": (
        "4c30b207a76502463bee6450614e0b4be778c893b79304e9b0e941e81cc8f257"
    ),
    str(EXECUTION_MANIFEST): (
        "50b86d6ab896a1c913ee83311f416f67392f29f6fb5a143f59f3abc08448d0c6"
    ),
    "docs/icla-strict-evaluator-contract-2026-07-19.md": (
        "5da04978589444acd371fdbf637f742a813baddb4ab124fe866fb9b85c7d93b7"
    ),
    "tests/test_evaluate_inverse_collateral_liquidation_absorption.py": (
        "f0998c2f3cfe6e62d6df2ce80cd01fcae5c0031d0fe935695304495bf865a5eb"
    ),
}
EXECUTION_FILE_SHA256 = {
    "train": {
        "market": "fa78e344e576ed3d1e911325613bce1465bfc76c259c0a3733cb350e1cdac2e4",
        "funding": "b94daae411b41d447e52dd0490a269ffd28eaf9316ddbea0da8a6a293d7d44ce",
    },
    "test": {
        "market": "3cbc1198ee32b5d77cdfa468bdaf9ed34af346a962b7026c70c70f3ff0ba7af7",
        "funding": "4b16e60417d30592679d41eeac2d08231c0bd37d337a73dbe9b8c0e43d285414",
    },
    "eval": {
        "market": "212a441e2e8213eda528e2cd586853515785f51ee4291ef8bf8f05ae0d6e52f4",
        "funding": "07dc50bbdff43f6704d819bea0ef0e32c5ff93d7072cdb8252753c122bec8fbd",
    },
}
EXPECTED_MARKET_ROWS = {"train": 32_256, "test": 52_704, "eval": 52_704}
EXPECTED_FUNDING_ROWS = {"train": 336, "test": 549, "eval": 549}
EXPECTED_CLOCK_ROWS = {"train": 30, "test": 111, "eval": 108}
EXPECTED_LIQUIDATION_ONLY_ROWS = {"train": 134, "test": 394, "eval": 337}
STAGES = tuple(SPLITS)
CANDIDATE = "ICLA-60"
HOLD_BARS = 12
CONTROL_NAMES = (
    "direction_flip",
    "liquidation_only_fade",
    "delayed_5m",
    "random_clocks",
)
BAR = cast(pd.Timedelta, pd.Timedelta(minutes=5))
YEAR_SECONDS = 365.2425 * 24.0 * 60.0 * 60.0
SCHEMA_VERSION = 1
RANDOM_CLOCK_SEED = 20_260_719
MAX_CLBR_ENTRY_JACCARD = 0.10


@dataclass(frozen=True)
class EvaluationConfig:
    combined_clocks: str = str(COMBINED_CLOCKS)
    split_clock_dir: str = str(SPLIT_CLOCK_DIR)
    control_clock_dir: str = str(CONTROL_CLOCK_DIR)
    support_result: str = str(SUPPORT_RESULT)
    execution_manifest: str = str(EXECUTION_MANIFEST)
    freeze_output: str = str(EVALUATOR_FREEZE)
    train_result: str = str(RESULT_PATHS["train"])
    test_result: str = str(RESULT_PATHS["test"])
    eval_result: str = str(RESULT_PATHS["eval"])
    leverage: float = 1.0
    base_cost_rate_per_side: float = 0.0006
    stress_cost_rate_per_side: float = 0.0010
    bootstrap_mean_block_trades: int = 4
    bootstrap_resamples: int = 50_000
    bootstrap_seed: int = 20_260_719


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if bool(pd.isna(parsed)):
        raise ValueError("timestamp cannot be NaT")
    return cast(pd.Timestamp, parsed)


def _stable_hash(payload: dict[str, Any], hash_field: str) -> str:
    stable = {key: value for key, value in payload.items() if key != hash_field}
    return canonical_hash(stable)


def _write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(frame.to_csv(index=False).encode())
    return raw.getvalue()


def _write_gzip_csv_exclusive(path: str | Path, frame: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_gzip_csv_bytes(frame))


def _require_canonical_config(cfg: EvaluationConfig) -> None:
    if asdict(cfg) != asdict(EvaluationConfig()):
        raise ValueError("all evaluator paths and parameters are frozen")


def _result_path(cfg: EvaluationConfig, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return Path(getattr(cfg, f"{stage}_result"))


def _split_clock_path(cfg: EvaluationConfig, stage: str) -> Path:
    return Path(cfg.split_clock_dir) / f"{stage}_clocks.csv.gz"


def _control_clock_path(cfg: EvaluationConfig, stage: str, control: str) -> Path:
    if control not in CONTROL_NAMES:
        raise ValueError(f"unknown control: {control}")
    return Path(cfg.control_clock_dir) / f"{stage}_{control}_clocks.csv.gz"


def _verify_static_dependencies() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen dependency changed: {path}")

    support = _read_json(SUPPORT_RESULT)
    if support.get("manifest_hash") != SUPPORT_MANIFEST_HASH:
        raise ValueError("support manifest hash changed")
    protocol = support.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("support stage opened outcomes")
    if protocol.get("execution_prices_opened") is not False:
        raise ValueError("support stage opened executable prices")
    if protocol.get("funding_opened") is not False:
        raise ValueError("support stage opened funding")
    if protocol.get("return_labels_constructed") is not False:
        raise ValueError("support stage constructed return labels")
    if (
        support.get("clocks", {}).get("sha256")
        != STATIC_INPUT_SHA256[str(COMBINED_CLOCKS)]
    ):
        raise ValueError("support points to different clocks")
    _source_only_preconditions(support)

    execution = _read_json(EXECUTION_MANIFEST)
    execution_protocol = execution.get("protocol", {})
    if execution_protocol.get("outcomes_opened") is not False:
        raise ValueError("execution source stage opened outcomes")
    if execution_protocol.get("strategy_returns_computed") is not False:
        raise ValueError("execution source stage computed returns")
    if execution_protocol.get("clbr_clocks_loaded") is not False:
        raise ValueError("execution source stage loaded strategy clocks")
    for stage in STAGES:
        for source_kind in ("market", "funding"):
            meta = execution["files"][stage][source_kind]
            if meta.get("sha256") != EXECUTION_FILE_SHA256[stage][source_kind]:
                raise ValueError(f"execution manifest changed {stage} {source_kind}")
    return support, execution


def _source_only_preconditions(support: dict[str, Any]) -> dict[str, Any]:
    overlap = support.get("clock_overlap", {})
    jaccard = float(overlap.get("entry_jaccard", float("inf")))
    maximum = float(overlap.get("maximum_entry_jaccard_allowed", -1.0))
    preconditions = {
        "support_passes": support.get("support_passes") is True,
        "clbr_alias_rejected": overlap.get("passes") is True
        and maximum == MAX_CLBR_ENTRY_JACCARD
        and jaccard <= MAX_CLBR_ENTRY_JACCARD,
        "clbr_entry_jaccard": jaccard,
        "maximum_clbr_entry_jaccard": maximum,
        "clbr_clock_sha256": overlap.get("clbr_clock_sha256"),
    }
    if not preconditions["support_passes"]:
        raise ValueError("source-only support gate failed")
    if not preconditions["clbr_alias_rejected"]:
        raise ValueError("source-only CLBR alias gate failed")
    return preconditions


def _load_combined_clocks(
    cfg: EvaluationConfig, support: dict[str, Any]
) -> pd.DataFrame:
    if _sha256(cfg.combined_clocks) != STATIC_INPUT_SHA256[str(COMBINED_CLOCKS)]:
        raise ValueError("combined clock bytes changed")
    time_columns = [
        "first_bar_open_time",
        "last_bar_open_time",
        "wave_completed_time",
        "feature_available_time",
        "entry_time",
        "planned_exit_time",
    ]
    clocks = pd.read_csv(
        cfg.combined_clocks, compression="gzip", parse_dates=time_columns
    )
    if len(clocks) != int(support["clocks"]["rows"]):
        raise ValueError("combined clock count changed")
    if not cast(pd.Series, clocks["candidate"]).eq("ICLA-60").all():
        raise ValueError("combined clocks contain another candidate")
    if not cast(pd.Series, clocks["direction"]).isin((-1, 1)).all():
        raise ValueError("combined clocks contain an invalid direction")
    if not cast(pd.Series, clocks["entry_time"]).is_monotonic_increasing:
        raise ValueError("combined clocks are not chronological")
    if cast(pd.Series, clocks["entry_time"]).duplicated().any():
        raise ValueError("combined clocks contain duplicate entries")
    if (
        not cast(pd.Series, clocks["entry_time"])
        .gt(clocks["feature_available_time"])
        .all()
    ):
        raise ValueError("clock entry precedes feature availability")
    return clocks


def _expected_split_clock_artifacts(
    cfg: EvaluationConfig, support: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    clocks = _load_combined_clocks(cfg, support)
    metadata: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for stage, (start_text, end_text) in SPLITS.items():
        start = _timestamp(start_text)
        end = _timestamp(end_text)
        subset = cast(pd.DataFrame, clocks.loc[clocks["split"].eq(stage)].copy())
        if len(subset) != EXPECTED_CLOCK_ROWS[stage]:
            raise ValueError(f"{stage} clock support changed")
        if (
            not cast(pd.Series, subset["entry_time"]).ge(start).all()
            or not cast(pd.Series, subset["planned_exit_time"]).lt(end).all()
        ):
            raise ValueError(f"{stage} clock crosses its physical boundary")
        entries = cast(pd.Series, subset["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, subset["planned_exit_time"]).reset_index(drop=True)
        if (
            len(subset) > 1
            and not entries.iloc[1:]
            .reset_index(drop=True)
            .ge(exits.iloc[:-1].reset_index(drop=True))
            .all()
        ):
            raise ValueError(f"{stage} clocks overlap")
        path = _split_clock_path(cfg, stage)
        metadata[stage] = {
            "path": str(path),
            "sha256": hashlib.sha256(_gzip_csv_bytes(subset)).hexdigest(),
            "rows": int(len(subset)),
            "start_inclusive": str(start),
            "end_exclusive": str(end),
        }
        frames[stage] = subset
    return metadata, frames


def _freeze_split_clocks(
    cfg: EvaluationConfig, support: dict[str, Any]
) -> dict[str, Any]:
    metadata, frames = _expected_split_clock_artifacts(cfg, support)
    for stage in STAGES:
        path = _split_clock_path(cfg, stage)
        _write_gzip_csv_exclusive(path, frames[stage])
        if _sha256(path) != metadata[stage]["sha256"]:
            raise ValueError(f"{stage} split clock write is not deterministic")
    return metadata


def _liquidation_only_clocks() -> pd.DataFrame:
    liquidation, activity = load_sources(PreregistrationConfig())
    state = derive_wave_state(liquidation, activity)
    total = cast(pd.Series, state["wave_total_liquidation_usd"])
    imbalance = cast(pd.Series, state["wave_liquidation_imbalance"])
    direction = cast(pd.Series, state["direction"])
    state["is_candidate"] = (
        cast(pd.Series, state["wave_source_valid"]).astype(bool)
        & cast(pd.Series, state["wave_event_count"]).ge(1.0)
        & total.ge(state["prior_wave_threshold_usd"])
        & imbalance.abs().ge(MIN_ABSOLUTE_LIQUIDATION_IMBALANCE)
        & direction.ne(0)
    )
    clocks = build_clocks(state)
    clocks["candidate"] = "control_liquidation_only_fade"
    return clocks


def _random_control_clocks(
    primary: pd.DataFrame, stage: str, stage_index: int
) -> pd.DataFrame:
    start = _timestamp(SPLITS[stage][0])
    end = _timestamp(SPLITS[stage][1])
    slots = pd.date_range(start, end, freq="1h", inclusive="left")
    slots = slots[slots + BAR * HOLD_BARS < end]
    if len(slots) < len(primary):
        raise ValueError(f"{stage} has too few non-overlapping random slots")
    rng = np.random.default_rng(RANDOM_CLOCK_SEED + stage_index)
    chosen = np.sort(rng.choice(len(slots), size=len(primary), replace=False))
    directions = cast(pd.Series, primary["direction"]).to_numpy(dtype=int).copy()
    rng.shuffle(directions)
    entries = pd.Series(slots[chosen])
    return pd.DataFrame(
        {
            "candidate": "control_random_clocks",
            "split": stage,
            "entry_time": entries,
            "planned_exit_time": entries + BAR * HOLD_BARS,
            "direction": directions,
        }
    ).sort_values("entry_time", ignore_index=True)


def _expected_control_clock_artifacts(
    cfg: EvaluationConfig,
    primary_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, pd.DataFrame]]]:
    liquidation_only = _liquidation_only_clocks()
    metadata: dict[str, dict[str, Any]] = {}
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for stage_index, stage in enumerate(STAGES):
        primary = primary_frames[stage]
        direction_flip = primary.copy()
        direction_flip["candidate"] = "control_direction_flip"
        direction_flip["direction"] = -cast(
            pd.Series, direction_flip["direction"]
        ).to_numpy(dtype=int)

        delayed = primary.copy()
        delayed["candidate"] = "control_delayed_5m"
        delayed["entry_time"] = delayed["entry_time"] + BAR
        delayed["planned_exit_time"] = delayed["planned_exit_time"] + BAR

        liquid = cast(
            pd.DataFrame,
            liquidation_only.loc[liquidation_only["split"].eq(stage)].copy(),
        )
        if len(liquid) != EXPECTED_LIQUIDATION_ONLY_ROWS[stage]:
            raise ValueError(f"{stage} liquidation-only support changed")

        stage_frames = {
            "direction_flip": direction_flip,
            "liquidation_only_fade": liquid,
            "delayed_5m": delayed,
            "random_clocks": _random_control_clocks(primary, stage, stage_index),
        }
        metadata[stage] = {}
        frames[stage] = stage_frames
        start = _timestamp(SPLITS[stage][0])
        end = _timestamp(SPLITS[stage][1])
        for control, frame in stage_frames.items():
            _validate_clock_frame(
                frame,
                stage,
                start,
                end,
                expected_rows=len(frame),
                expected_candidate=f"control_{control}",
            )
            path = _control_clock_path(cfg, stage, control)
            metadata[stage][control] = {
                "path": str(path),
                "sha256": hashlib.sha256(_gzip_csv_bytes(frame)).hexdigest(),
                "rows": int(len(frame)),
                "long": int(cast(pd.Series, frame["direction"]).gt(0).sum()),
                "short": int(cast(pd.Series, frame["direction"]).lt(0).sum()),
            }
    return metadata, frames


def _freeze_control_clocks(
    cfg: EvaluationConfig,
    primary_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    metadata, frames = _expected_control_clock_artifacts(cfg, primary_frames)
    for stage in STAGES:
        for control in CONTROL_NAMES:
            path = _control_clock_path(cfg, stage, control)
            _write_gzip_csv_exclusive(path, frames[stage][control])
            if _sha256(path) != metadata[stage][control]["sha256"]:
                raise ValueError(f"{stage} {control} clock write is not deterministic")
    return metadata


def _build_freeze_report(
    cfg: EvaluationConfig,
    split_clocks: dict[str, Any],
    control_clocks: dict[str, dict[str, Any]],
    source_only_preconditions: dict[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "support_commit": SUPPORT_COMMIT,
        "execution_source_commit": EXECUTION_SOURCE_COMMIT,
        "support_manifest_hash": SUPPORT_MANIFEST_HASH,
        "evaluation_source": str(EVALUATOR_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "static_input_sha256": STATIC_INPUT_SHA256,
        "execution_file_sha256": EXECUTION_FILE_SHA256,
        "split_clocks": split_clocks,
        "control_clocks": control_clocks,
        "source_only_preconditions": source_only_preconditions,
        "config": asdict(cfg),
        "execution_contract": {
            "position_sizing": "fixed quantity = pre-entry equity * leverage / entry open",
            "held_bars": "[entry_position, planned_exit_position)",
            "time_exit": "planned-exit bar open",
            "stop_or_take_profit": "none",
            "funding_inclusion": (
                "interior symmetric; exact entry/exit credits dropped and debits retained"
            ),
            "funding_cash": "-direction * fixed_quantity * settlement_mark * funding_rate",
            "strict_mdd": (
                "global/pre-entry HWM, entry fee, funding marks, favorable-before-adverse "
                "held OHLC, virtual adverse-mark exit fee, actual exit fee"
            ),
            "calendar_year_seconds": YEAR_SECONDS,
        },
        "bootstrap_contract": {
            "method": "circular stationary block bootstrap of net trade returns under centered null",
            "mean_block_trades": cfg.bootstrap_mean_block_trades,
            "resamples": cfg.bootstrap_resamples,
            "seed": cfg.bootstrap_seed,
            "p_value": "(1 + count(centered_bootstrap_mean >= observed_mean)) / (B + 1)",
        },
        "control_contract": {
            "names": list(CONTROL_NAMES),
            "random_clock_seed": RANDOM_CLOCK_SEED,
            "random_clock_support": (
                "same stage trade count and long/short count as primary"
            ),
            "promotion_rule": "every control must fail the complete stage gate",
        },
        "promotion_gates": promotion_gate_contract(),
        "opened_windows": [],
        "sealed_windows": list(STAGES),
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    report["freeze_hash"] = _stable_hash(report, "freeze_hash")
    return report


def freeze_evaluator(cfg: EvaluationConfig = EvaluationConfig()) -> dict[str, Any]:
    """Freeze evaluator bytes and physically split source-only clocks."""

    _require_canonical_config(cfg)
    if Path(cfg.freeze_output).exists():
        raise ValueError("evaluator freeze already exists and cannot be replaced")
    existing_results = [
        str(_result_path(cfg, stage))
        for stage in STAGES
        if _result_path(cfg, stage).exists()
    ]
    if existing_results:
        raise ValueError(
            f"stage results already exist before freeze: {existing_results}"
        )
    existing_clocks = [
        str(_split_clock_path(cfg, stage))
        for stage in STAGES
        if _split_clock_path(cfg, stage).exists()
    ]
    if existing_clocks:
        raise ValueError(f"split clocks already exist before freeze: {existing_clocks}")
    existing_controls = [
        str(_control_clock_path(cfg, stage, control))
        for stage in STAGES
        for control in CONTROL_NAMES
        if _control_clock_path(cfg, stage, control).exists()
    ]
    if existing_controls:
        raise ValueError(
            f"control clocks already exist before freeze: {existing_controls}"
        )

    support, execution = _verify_static_dependencies()
    for stage in STAGES:
        for source_kind in ("market", "funding"):
            path = execution["files"][stage][source_kind]["path"]
            expected = EXECUTION_FILE_SHA256[stage][source_kind]
            if _sha256(path) != expected:
                raise ValueError(f"frozen {stage} {source_kind} bytes changed")
    split_clocks = _freeze_split_clocks(cfg, support)
    _split_metadata, primary_frames = _expected_split_clock_artifacts(cfg, support)
    control_clocks = _freeze_control_clocks(cfg, primary_frames)
    source_only_preconditions = _source_only_preconditions(support)
    report = _build_freeze_report(
        cfg,
        split_clocks,
        control_clocks,
        source_only_preconditions,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_exclusive(cfg.freeze_output, report)
    return report


def verify_evaluator_freeze(
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    """Verify sealed metadata without reading any split market/funding frame."""

    _require_canonical_config(cfg)
    support, _execution = _verify_static_dependencies()
    freeze = _read_json(cfg.freeze_output)
    expected_split_clocks, primary_frames = _expected_split_clock_artifacts(
        cfg, support
    )
    expected_control_clocks, _control_frames = _expected_control_clock_artifacts(
        cfg, primary_frames
    )
    source_only_preconditions = _source_only_preconditions(support)
    created_at = freeze.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("evaluator freeze lacks its creation timestamp")
    expected_freeze = _build_freeze_report(
        cfg,
        expected_split_clocks,
        expected_control_clocks,
        source_only_preconditions,
        created_at=created_at,
    )
    if freeze != expected_freeze:
        raise ValueError("evaluator freeze does not reproduce from frozen inputs")
    for stage, expected in expected_split_clocks.items():
        path = _split_clock_path(cfg, stage)
        if str(path) != expected["path"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"{stage} split clock bytes changed after freeze")
    for stage in STAGES:
        for control in CONTROL_NAMES:
            expected = expected_control_clocks[stage][control]
            path = _control_clock_path(cfg, stage, control)
            if str(path) != expected["path"] or _sha256(path) != expected["sha256"]:
                raise ValueError(f"{stage} {control} clock bytes changed after freeze")
    return freeze


def _validate_market(
    market: pd.DataFrame, stage: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    expected = pd.Series(
        pd.date_range(start, end, freq="5min", inclusive="left"), name="date"
    )
    if len(market) != EXPECTED_MARKET_ROWS[stage] or not market["date"].equals(
        expected
    ):
        raise ValueError(f"{stage} market is not the exact five-minute grid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError(f"{stage} market contains invalid prices")
    if (
        not cast(pd.Series, market["high"])
        .ge(market[["open", "close"]].max(axis=1))
        .all()
        or not cast(pd.Series, market["low"])
        .le(market[["open", "close"]].min(axis=1))
        .all()
    ):
        raise ValueError(f"{stage} market violates OHLC envelope")


def _validate_funding(
    funding: pd.DataFrame, stage: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    times = cast(pd.Series, funding["funding_time"])
    if len(funding) != EXPECTED_FUNDING_ROWS[stage]:
        raise ValueError(f"{stage} funding count changed")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError(f"{stage} funding times are invalid")
    if not times.ge(start).all() or not times.lt(end).all():
        raise ValueError(f"{stage} funding crosses its boundary")
    values = funding[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or (funding["settlement_mark_price"] <= 0.0).any():
        raise ValueError(f"{stage} funding contains invalid values")


def _validate_clock_frame(
    clocks: pd.DataFrame,
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    expected_rows: int,
    expected_candidate: str,
) -> None:
    if len(clocks) != expected_rows:
        raise ValueError(f"{stage} clock count changed")
    if "stop_price" in clocks.columns:
        raise ValueError(f"{stage} clocks contain a forbidden stop")
    if not cast(pd.Series, clocks["candidate"]).eq(expected_candidate).all():
        raise ValueError(f"{stage} clocks contain another candidate")
    if not cast(pd.Series, clocks["split"]).eq(stage).all():
        raise ValueError(f"{stage} split clock contains another stage")
    entries = cast(pd.Series, clocks["entry_time"])
    exits = cast(pd.Series, clocks["planned_exit_time"])
    if entries.duplicated().any() or not entries.is_monotonic_increasing:
        raise ValueError(f"{stage} entries are invalid")
    if not entries.ge(start).all() or not exits.lt(end).all():
        raise ValueError(f"{stage} clocks cross the stage boundary")
    if not exits.sub(entries).eq(BAR * HOLD_BARS).all():
        raise ValueError(f"{stage} hold differs from frozen 12 bars")
    if (
        len(clocks) > 1
        and not entries.iloc[1:]
        .reset_index(drop=True)
        .ge(exits.iloc[:-1].reset_index(drop=True))
        .all()
    ):
        raise ValueError(f"{stage} clocks overlap")
    if not cast(pd.Series, clocks["direction"]).isin((-1, 1)).all():
        raise ValueError(f"{stage} direction is invalid")


def _validate_clocks(
    clocks: pd.DataFrame, stage: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    _validate_clock_frame(
        clocks,
        stage,
        start,
        end,
        expected_rows=EXPECTED_CLOCK_ROWS[stage],
        expected_candidate=CANDIDATE,
    )


def _load_stage_inputs(
    stage: str, cfg: EvaluationConfig, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load only the requested stage's market, funding, and split clock files."""

    execution = _read_json(cfg.execution_manifest)
    start = _timestamp(SPLITS[stage][0])
    end = _timestamp(SPLITS[stage][1])
    market_meta = execution["files"][stage]["market"]
    funding_meta = execution["files"][stage]["funding"]
    clock_meta = freeze["split_clocks"][stage]
    for kind, meta, expected in (
        ("market", market_meta, EXECUTION_FILE_SHA256[stage]["market"]),
        ("funding", funding_meta, EXECUTION_FILE_SHA256[stage]["funding"]),
        ("clock", clock_meta, clock_meta["sha256"]),
    ):
        if _sha256(meta["path"]) != expected:
            raise ValueError(f"{stage} {kind} bytes changed")
    market = pd.read_csv(market_meta["path"], compression="gzip", parse_dates=["date"])
    funding = pd.read_csv(
        funding_meta["path"], compression="gzip", parse_dates=["funding_time"]
    )
    clock_times = [
        "first_bar_open_time",
        "last_bar_open_time",
        "wave_completed_time",
        "feature_available_time",
        "entry_time",
        "planned_exit_time",
    ]
    clocks = pd.read_csv(
        clock_meta["path"], compression="gzip", parse_dates=clock_times
    )
    _validate_market(market, stage, start, end)
    _validate_funding(funding, stage, start, end)
    _validate_clocks(clocks, stage, start, end)
    return market, funding, clocks, start, end


def _load_stage_control_clocks(
    stage: str,
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for control in CONTROL_NAMES:
        meta = freeze["control_clocks"][stage][control]
        if _sha256(meta["path"]) != meta["sha256"]:
            raise ValueError(f"{stage} {control} clock bytes changed")
        frame = pd.read_csv(
            meta["path"],
            compression="gzip",
            parse_dates=["entry_time", "planned_exit_time"],
        )
        _validate_clock_frame(
            frame,
            stage,
            start,
            end,
            expected_rows=int(meta["rows"]),
            expected_candidate=f"control_{control}",
        )
        loaded[control] = frame
    return loaded


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    leverage: float,
    cost_rate_per_side: float,
) -> dict[str, Any]:
    """Simulate fixed one-hour trades with hardened path accounting."""

    if leverage <= 0.0 or cost_rate_per_side < 0.0:
        raise ValueError("leverage must be positive and cost must be non-negative")
    positions = {
        cast(pd.Timestamp, value): index
        for index, value in enumerate(cast(pd.Series, market["date"]))
    }
    funding_times = cast(pd.Series, funding["funding_time"])
    realized_equity = 1.0
    high_water_mark = 1.0
    maximum_drawdown = 0.0
    records: list[dict[str, Any]] = []

    def update_path(value: float) -> None:
        nonlocal high_water_mark, maximum_drawdown
        if not np.isfinite(value):
            raise ValueError("non-finite strict equity path")
        high_water_mark = max(high_water_mark, value)
        if high_water_mark > 0.0:
            maximum_drawdown = max(
                maximum_drawdown, (high_water_mark - value) / high_water_mark
            )

    for clock in clocks.to_dict(orient="records"):
        entry_time = _timestamp(clock["entry_time"])
        exit_time = _timestamp(clock["planned_exit_time"])
        entry_position = positions.get(entry_time)
        exit_position = positions.get(exit_time)
        if entry_position is None or exit_position is None:
            raise ValueError("frozen clock is absent from the market grid")
        if exit_position - entry_position != HOLD_BARS:
            raise ValueError("frozen hold is not exactly 12 bars")
        direction = int(clock["direction"])
        if direction not in (-1, 1):
            raise ValueError("clock direction must be -1 or 1")

        entry_price = float(market.iloc[entry_position]["open"])
        exit_price = float(market.iloc[exit_position]["open"])
        pre_entry_equity = realized_equity
        quantity = pre_entry_equity * leverage / entry_price
        entry_fee = quantity * entry_price * cost_rate_per_side
        cash = pre_entry_equity - entry_fee
        update_path(cash)

        included_funding = funding.loc[
            funding_times.ge(entry_time) & funding_times.le(exit_time)
        ].copy()
        next_funding = 0
        funding_cash = 0.0
        applied_funding_events = 0
        dropped_boundary_credits = 0

        def apply_funding_through(upper: pd.Timestamp) -> None:
            nonlocal cash
            nonlocal funding_cash
            nonlocal next_funding
            nonlocal applied_funding_events
            nonlocal dropped_boundary_credits
            while next_funding < len(included_funding):
                event = included_funding.iloc[next_funding]
                event_time = cast(pd.Timestamp, event["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(event["settlement_mark_price"])
                cash_flow = (
                    -direction
                    * quantity
                    * settlement_mark
                    * float(event["funding_rate"])
                )
                is_boundary = event_time in (entry_time, exit_time)
                if is_boundary and cash_flow > 0.0:
                    dropped_boundary_credits += 1
                else:
                    cash += cash_flow
                    funding_cash += cash_flow
                    applied_funding_events += 1
                    marked = cash + direction * quantity * (
                        settlement_mark - entry_price
                    )
                    virtual_exit_fee = quantity * settlement_mark * cost_rate_per_side
                    update_path(marked - virtual_exit_fee)
                next_funding += 1

        for position in range(entry_position, exit_position):
            bar = market.iloc[position]
            bar_time = cast(pd.Timestamp, bar["date"])
            accounting_bar_end = _timestamp(bar_time + BAR - pd.Timedelta(1, unit="ns"))
            apply_funding_through(accounting_bar_end)
            favorable_price = float(bar["high"] if direction > 0 else bar["low"])
            favorable_equity = cash + direction * quantity * (
                favorable_price - entry_price
            )
            update_path(favorable_equity)
            adverse_price = float(bar["low"] if direction > 0 else bar["high"])
            adverse_equity = cash + direction * quantity * (adverse_price - entry_price)
            adverse_exit_fee = quantity * adverse_price * cost_rate_per_side
            update_path(adverse_equity - adverse_exit_fee)

        apply_funding_through(exit_time)
        if next_funding != len(included_funding):
            raise ValueError("funding event was not applied before exit")
        gross_pnl = direction * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * cost_rate_per_side
        realized_equity = cash + gross_pnl - exit_fee
        update_path(realized_equity)
        net_return = realized_equity / pre_entry_equity - 1.0
        gross_return = gross_pnl / pre_entry_equity
        records.append(
            {
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "planned_exit_time": str(exit_time),
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": "time_exit",
                "bars_held": HOLD_BARS,
                "pre_entry_equity": pre_entry_equity,
                "fixed_quantity": quantity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "funding_cash": funding_cash,
                "funding_events": applied_funding_events,
                "dropped_boundary_funding_credits": dropped_boundary_credits,
                "gross_pnl": gross_pnl,
                "gross_return": gross_return,
                "net_return": net_return,
                "post_exit_equity": realized_equity,
            }
        )

    years = (end - start).total_seconds() / YEAR_SECONDS
    if years <= 0.0:
        raise ValueError("evaluation window must have positive duration")
    absolute_return = realized_equity - 1.0
    cagr = realized_equity ** (1.0 / years) - 1.0 if realized_equity > 0.0 else -1.0
    ratio = cagr / max(maximum_drawdown, 1e-12)
    net_returns = np.asarray([float(row["net_return"]) for row in records], dtype=float)
    gross_returns = np.asarray(
        [float(row["gross_return"]) for row in records], dtype=float
    )
    directions = np.asarray([int(row["direction"]) for row in records], dtype=int)
    exposure_seconds = sum(
        (_timestamp(row["exit_time"]) - _timestamp(row["entry_time"])).total_seconds()
        for row in records
    )
    return {
        "metrics": {
            "absolute_return_pct": absolute_return * 100.0,
            "cagr_pct": cagr * 100.0,
            "strict_mdd_pct": maximum_drawdown * 100.0,
            "cagr_to_strict_mdd": ratio,
            "candidate_clocks": int(len(clocks)),
            "executable_trades": int(len(records)),
            "invalid_crossed_stops": 0,
            "long_trades": int((directions > 0).sum()),
            "short_trades": int((directions < 0).sum()),
            "win_rate_pct": (
                float((net_returns > 0.0).mean() * 100.0) if len(net_returns) else 0.0
            ),
            "mean_gross_trade_bps": (
                float(gross_returns.mean() * 10_000.0) if len(gross_returns) else 0.0
            ),
            "mean_net_trade_bps": (
                float(net_returns.mean() * 10_000.0) if len(net_returns) else 0.0
            ),
            "total_funding_cash": float(
                sum(float(row["funding_cash"]) for row in records)
            ),
            "exposure_pct": (exposure_seconds / (end - start).total_seconds() * 100.0),
            "calendar_start": str(start),
            "calendar_end_exclusive": str(end),
            "calendar_years": years,
            "cost_rate_per_side": cost_rate_per_side,
            "leverage": leverage,
        },
        "trades": records,
        "net_trade_returns": net_returns,
    }


def stationary_bootstrap_p_value(
    returns: np.ndarray,
    *,
    mean_block_trades: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires at least two finite trade returns")
    if mean_block_trades < 1 or resamples < 1:
        raise ValueError("bootstrap parameters must be positive")
    observed = float(values.mean())
    centered = values - observed
    restart_probability = 1.0 / mean_block_trades
    rng = np.random.default_rng(seed)
    exceedances = 0
    n = len(values)
    for _ in range(resamples):
        index = int(rng.integers(n))
        total = float(centered[index])
        for _position in range(1, n):
            if float(rng.random()) < restart_probability:
                index = int(rng.integers(n))
            else:
                index = (index + 1) % n
            total += float(centered[index])
        if total / n >= observed:
            exceedances += 1
    return {
        "method": "circular_stationary_block_bootstrap_centered_null",
        "observed_mean_net_return": observed,
        "mean_block_trades": mean_block_trades,
        "resamples": resamples,
        "seed": seed,
        "exceedances": exceedances,
        "one_sided_p_value": (1 + exceedances) / (resamples + 1),
    }


def promotion_gate_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for stage in STAGES:
        contract[stage] = {
            "absolute_return_positive": True,
            "minimum_cagr_to_strict_mdd": 3.0 if stage == "eval" else 2.0,
            "maximum_strict_mdd_pct": 15.0,
            "minimum_executable_trades": 25 if stage == "train" else 90,
            "minimum_trades_per_side": 8 if stage == "train" else 20,
            "stress_absolute_return_positive": True,
            "maximum_bootstrap_p_value": 0.10,
        }
    return contract


def _evaluate_gates(
    stage: str,
    base: dict[str, Any],
    stress: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    metrics = base["metrics"]
    stress_metrics = stress["metrics"]
    contract = promotion_gate_contract()[stage]
    side_floor = int(contract["minimum_trades_per_side"])
    checks = {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"]
        >= float(contract["minimum_cagr_to_strict_mdd"]),
        "strict_mdd": metrics["strict_mdd_pct"]
        <= float(contract["maximum_strict_mdd_pct"]),
        "executable_trades": metrics["executable_trades"]
        >= int(contract["minimum_executable_trades"]),
        "minimum_long_trades": metrics["long_trades"] >= side_floor,
        "minimum_short_trades": metrics["short_trades"] >= side_floor,
        "stress_absolute_return_positive": stress_metrics["absolute_return_pct"] > 0.0,
        "bootstrap_p_value": bootstrap["one_sided_p_value"]
        <= float(contract["maximum_bootstrap_p_value"]),
    }
    return {"checks": checks, "passes": bool(all(checks.values()))}


def _apply_mechanism_control_veto(
    promotion: dict[str, Any], controls: dict[str, Any]
) -> dict[str, Any]:
    controls_rejected = not any(
        bool(row["complete_gate"]["passes"]) for row in controls.values()
    )
    promotion["checks"]["mechanism_controls_rejected"] = controls_rejected
    promotion["passes"] = bool(promotion["passes"] and controls_rejected)
    return promotion


def _apply_source_only_preconditions(
    promotion: dict[str, Any], preconditions: dict[str, Any]
) -> dict[str, Any]:
    promotion["checks"]["source_support_passes"] = bool(preconditions["support_passes"])
    promotion["checks"]["clbr_alias_rejected"] = bool(
        preconditions["clbr_alias_rejected"]
    )
    promotion["passes"] = bool(
        promotion["passes"]
        and promotion["checks"]["source_support_passes"]
        and promotion["checks"]["clbr_alias_rejected"]
    )
    return promotion


def _compute_stage_report(
    stage: str,
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    market, funding, clocks, start, end = _load_stage_inputs(stage, cfg, freeze)
    control_clocks = _load_stage_control_clocks(stage, cfg, freeze, start, end)
    base = simulate_strict(
        market,
        funding,
        clocks,
        start=start,
        end=end,
        leverage=cfg.leverage,
        cost_rate_per_side=cfg.base_cost_rate_per_side,
    )
    stress = simulate_strict(
        market,
        funding,
        clocks,
        start=start,
        end=end,
        leverage=cfg.leverage,
        cost_rate_per_side=cfg.stress_cost_rate_per_side,
    )
    bootstrap = stationary_bootstrap_p_value(
        base["net_trade_returns"],
        mean_block_trades=cfg.bootstrap_mean_block_trades,
        resamples=cfg.bootstrap_resamples,
        seed=cfg.bootstrap_seed,
    )
    promotion = _evaluate_gates(stage, base, stress, bootstrap)
    controls: dict[str, Any] = {}
    for control in CONTROL_NAMES:
        control_base = simulate_strict(
            market,
            funding,
            control_clocks[control],
            start=start,
            end=end,
            leverage=cfg.leverage,
            cost_rate_per_side=cfg.base_cost_rate_per_side,
        )
        control_stress = simulate_strict(
            market,
            funding,
            control_clocks[control],
            start=start,
            end=end,
            leverage=cfg.leverage,
            cost_rate_per_side=cfg.stress_cost_rate_per_side,
        )
        control_bootstrap = stationary_bootstrap_p_value(
            control_base["net_trade_returns"],
            mean_block_trades=cfg.bootstrap_mean_block_trades,
            resamples=cfg.bootstrap_resamples,
            seed=cfg.bootstrap_seed,
        )
        control_gate = _evaluate_gates(
            stage, control_base, control_stress, control_bootstrap
        )
        controls[control] = {
            "base": {"metrics": control_base["metrics"]},
            "stress": {"metrics": control_stress["metrics"]},
            "bootstrap": control_bootstrap,
            "complete_gate": control_gate,
        }
    promotion = _apply_mechanism_control_veto(promotion, controls)
    promotion = _apply_source_only_preconditions(
        promotion, freeze["source_only_preconditions"]
    )
    stage_index = STAGES.index(stage)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "stage": stage,
        "evaluator_freeze_hash": freeze["freeze_hash"],
        "evaluation_source_sha256": freeze["evaluation_source_sha256"],
        "protocol": {
            "opened_windows": list(STAGES[: stage_index + 1]),
            "sealed_windows": list(STAGES[stage_index + 1 :]),
            "loaded_market_windows": [stage],
            "loaded_funding_windows": [stage],
            "loaded_clock_windows": [stage],
            "loaded_control_clock_windows": [stage],
            "parameters_mutated_after_freeze": False,
        },
        "window": {"start_inclusive": str(start), "end_exclusive": str(end)},
        "base": {"metrics": base["metrics"], "trades": base["trades"]},
        "stress": {"metrics": stress["metrics"]},
        "bootstrap": bootstrap,
        "controls": controls,
        "promotion": promotion,
    }
    report["result_hash"] = _stable_hash(report, "result_hash")
    return report


def _verify_prior_result(
    stage: str, cfg: EvaluationConfig, freeze: dict[str, Any]
) -> dict[str, Any]:
    result = _read_json(_result_path(cfg, stage))
    if result.get("result_hash") != _stable_hash(result, "result_hash"):
        raise ValueError(f"{stage} result hash changed")
    created_at = result.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"{stage} result lacks its creation timestamp")
    expected = _compute_stage_report(
        stage,
        cfg,
        freeze,
        created_at=created_at,
    )
    if result != expected:
        raise ValueError(f"{stage} result does not reproduce from frozen inputs")
    if result["promotion"]["passes"] is not True:
        raise ValueError(f"{stage} failed; later windows remain sealed")
    return result


def evaluate_stage(
    stage: str, cfg: EvaluationConfig = EvaluationConfig()
) -> dict[str, Any]:
    _require_canonical_config(cfg)
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    freeze = verify_evaluator_freeze(cfg)
    output = _result_path(cfg, stage)
    if output.exists():
        raise ValueError(f"{stage} result already exists and cannot be replaced")
    stage_index = STAGES.index(stage)
    for later in STAGES[stage_index + 1 :]:
        if _result_path(cfg, later).exists():
            raise ValueError(f"later stage exists before {stage}: {later}")
    for prior in STAGES[:stage_index]:
        if not _result_path(cfg, prior).exists():
            raise ValueError(f"{prior} must pass before opening {stage}")
        _verify_prior_result(prior, cfg, freeze)

    report = _compute_stage_report(
        stage,
        cfg,
        freeze,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_exclusive(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", *STAGES))
    args = parser.parse_args()
    if args.action == "freeze":
        report = freeze_evaluator()
        summary = {
            "freeze_hash": report["freeze_hash"],
            "opened_windows": report["opened_windows"],
            "sealed_windows": report["sealed_windows"],
        }
    else:
        report = evaluate_stage(args.action)
        summary = {
            "stage": args.action,
            "base": report["base"]["metrics"],
            "stress_absolute_return_pct": report["stress"]["metrics"][
                "absolute_return_pct"
            ],
            "bootstrap_p_value": report["bootstrap"]["one_sided_p_value"],
            "promotion": report["promotion"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
