"""Sequential hardened evaluator for frozen DLPD-12 clocks.

The evaluator freeze is source-only: it opens no BTC execution OHLC or funding
row and hashes no execution-data bytes.  Once frozen, each stage receives an
exact physical execution-data slice.  Later stages remain sealed unless every
prior frozen gate passed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import evaluate_flow_centrality_incubation_relay as strict_engine  # noqa: E402
from training import evaluate_stablecoin_quote_flow_diffusion as strict_source  # noqa: E402
from training import preregister_btcdom_leverage_polarity_decomposition as dlpd  # noqa: E402
from training.build_six_alt_price_free_flow_panel import (  # noqa: E402
    deterministic_gzip_csv,
)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("DLPD-12 timestamp is NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


POLICY_ID = dlpd.POLICY_ID
SUPPORT_COMMIT = "5f7fb3baa4bb4e5cfb07bdc3577e3add75a76932"
PREREGISTRATION = Path(
    "results/btcdom_leverage_polarity_decomposition_preregistration_2026-07-20.json"
)
SUPPORT_RESULT = Path(
    "results/btcdom_leverage_polarity_decomposition_support_2026-07-20.json"
)
SOURCE_CLOCKS = Path(
    "data/btcdom_leverage_polarity_decomposition_clocks_2022_2023.csv.gz"
)
EVALUATION_CLOCKS = Path(
    "data/btcdom_leverage_polarity_decomposition_evaluation_clocks_2022_2023.csv.gz"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_btcdom_leverage_polarity_decomposition.py"
)
EVALUATOR_FREEZE = Path(
    "results/btcdom_leverage_polarity_decomposition_evaluator_freeze_2026-07-20.json"
)

LEGACY_MARKET = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
LEGACY_MARKET_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
LEGACY_MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
LEGACY_MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
LEGACY_FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
LEGACY_FUNDING_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
LEGACY_FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
LEGACY_FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

STATIC_INPUT_SHA256 = {
    str(dlpd.PREREGISTRATION_SOURCE): (
        "0b3797487daf156b66f1799a20b474e38a7bd7387c831462af6d363113b663ab"
    ),
    "training/build_btcdom_leverage_polarity_decomposition_support.py": (
        "f662128d5f0a09f8ed80182791df6328ca12efe10e8f8df9682b907d33fb5a23"
    ),
    str(PREREGISTRATION): (
        "6d5ba05072d7e1677239e2a6dba9ec8dab79bfb7a7e25fe89b3396e269adc9ff"
    ),
    str(SUPPORT_RESULT): (
        "1107694d5ff304aabaabbb962e9aeeaa64075001e494a0432dba3261ceace4f6"
    ),
    str(SOURCE_CLOCKS): (
        "b33990f1629465caa837aa1f6f74430054b7185b68ece47b8c7540f9c11bf0fb"
    ),
    "docs/btcdom-leverage-polarity-decomposition-preregistration-2026-07-20.md": (
        "f03f7b90587486ccdbc1d7d4e91bb96820d8e0ce38d0497f4a3170b22d2208a8"
    ),
    "docs/btcdom-leverage-polarity-decomposition-support-pass-2026-07-20.md": (
        "d20d1809b1d71ee8487ce5f2d8ec649a059008cdbd3e1dd2487910dd698ed6dd"
    ),
    str(LEGACY_MARKET_MANIFEST): LEGACY_MARKET_MANIFEST_SHA256,
    str(LEGACY_FUNDING_MANIFEST): LEGACY_FUNDING_MANIFEST_SHA256,
    str(strict_engine.EVALUATOR_SOURCE): (
        "036b22442a2080e7ea5ffe914c605a9b1b1a55b128a315a2f2f05be7b37a736d"
    ),
    str(strict_source.EVALUATOR_SOURCE): (
        "0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b"
    ),
}

STAGE_ORDER = ("train", "test", "eval", "final")
STAGE_SPLITS = {"train": "2022", "test": "2023"}
STAGE_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (_utc("2022-01-01"), _utc("2023-01-01")),
    "test": (_utc("2023-01-01"), _utc("2024-01-01")),
    "eval": (_utc("2024-01-01"), _utc("2026-01-01")),
    "final": (_utc("2026-01-01"), _utc("2026-07-01")),
}
HALF_WINDOWS: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {
    "train": {
        "2022_h1": (_utc("2022-01-01"), _utc("2022-07-01")),
        "2022_h2": (_utc("2022-07-01"), _utc("2023-01-01")),
    },
    "test": {
        "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
        "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
    },
    "eval": {
        "2024": (_utc("2024-01-01"), _utc("2025-01-01")),
        "2025": (_utc("2025-01-01"), _utc("2026-01-01")),
    },
    "final": {
        "2026_q1": (_utc("2026-01-01"), _utc("2026-04-01")),
        "2026_q2": (_utc("2026-04-01"), _utc("2026-07-01")),
    },
}
STAGE_OUTPUTS = {
    "train": Path(
        "results/btcdom_leverage_polarity_decomposition_train_2022_2026-07-20.json"
    ),
    "test": Path(
        "results/btcdom_leverage_polarity_decomposition_test_2023_2026-07-20.json"
    ),
    "eval": Path(
        "results/btcdom_leverage_polarity_decomposition_eval_2024_2025_2026-07-20.json"
    ),
    "final": Path(
        "results/btcdom_leverage_polarity_decomposition_final_2026h1_2026-07-20.json"
    ),
}
STAGE_DOCS = {
    stage: Path(
        f"docs/btcdom-leverage-polarity-decomposition-{stage}-result-2026-07-20.md"
    )
    for stage in STAGE_ORDER
}
STAGE_SOURCE_MANIFESTS = {
    stage: Path(
        "results/"
        f"btcdom_leverage_polarity_decomposition_{stage}_execution_source_2026-07-20.json"
    )
    for stage in STAGE_ORDER
}
STAGE_SOURCE_DIRS = {
    stage: Path("data/btcdom_leverage_polarity_decomposition_execution") / stage
    for stage in STAGE_ORDER
}
FUTURE_SIGNAL_MANIFESTS = {
    stage: Path(
        "results/"
        f"btcdom_leverage_polarity_decomposition_{stage}_signal_source_2026-07-20.json"
    )
    for stage in ("eval", "final")
}

DERIVED_SAME_CLOCK_CONTROLS = ("direction_flip", "deterministic_random_side")
LATENCY_CONTROLS = ("extra_latency_1h",)
DERIVED_CONTROLS = (*DERIVED_SAME_CLOCK_CONTROLS, *LATENCY_CONTROLS)
ALL_CONTROLS = (*dlpd.CONTROLS, *DERIVED_CONTROLS)
EVALUATION_CLOCK_COLUMNS = dlpd.EVENT_COLUMNS


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 144
    exact_cluster_max: int = 20
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_719


FROZEN_CONFIG = EvaluationConfig()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"DLPD-12 expected a JSON object: {path}")
    return payload


def _verify_manifest(payload: dict[str, Any], *, label: str) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError(f"DLPD-12 {label} manifest hash changed")


def _verify_config() -> None:
    strict_cfg = strict_engine.EvaluationConfig()
    if asdict(FROZEN_CONFIG) != asdict(strict_cfg):
        raise ValueError("DLPD-12 strict-engine configuration drifted")


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"DLPD-12 frozen input changed: {path}")
    prereg = _load_json(PREREGISTRATION)
    support = _load_json(SUPPORT_RESULT)
    _verify_manifest(prereg, label="preregistration")
    _verify_manifest(support, label="support")
    if prereg.get("candidate") != POLICY_ID or support.get("candidate") != POLICY_ID:
        raise ValueError("DLPD-12 frozen identity changed")
    if prereg.get("outcomes_opened") is not False:
        raise ValueError("DLPD-12 preregistration opened outcomes")
    if support.get("outcomes_opened") is not False:
        raise ValueError("DLPD-12 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("DLPD-12 support opened an outcome source")
    if support.get("support_passed") is not True:
        raise ValueError("DLPD-12 source support did not pass")
    if support.get("clock_sha256") != STATIC_INPUT_SHA256[str(SOURCE_CLOCKS)]:
        raise ValueError("DLPD-12 support no longer binds its clock")
    expected_sequence = [
        "train_2022",
        "test_2023",
        "eval_2024_2025",
        "final_2026H1",
    ]
    gate = prereg.get("conditional_outcome_gate", {})
    if gate.get("sequence") != expected_sequence:
        raise ValueError("DLPD-12 outcome sequence changed")
    _verify_config()
    return prereg, support


def _deterministic_random_side(decision_time: Any) -> int:
    timestamp = _utc(decision_time).strftime("%Y-%m-%dT%H:%M:%SZ")
    nibble = int(hashlib.sha256(f"{POLICY_ID}|{timestamp}".encode()).hexdigest()[0], 16)
    return 1 if nibble % 2 == 0 else -1


def _read_source_clocks() -> pd.DataFrame:
    _, support = _verify_static_inputs()
    frame = pd.read_csv(SOURCE_CLOCKS)
    if tuple(frame.columns) != dlpd.EVENT_COLUMNS:
        raise ValueError("DLPD-12 source clock schema changed")
    frame["split"] = frame["split"].astype(str)
    for column in (
        "source_hour_start",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if len(frame) != int(support["clock_rows_all_controls"]):
        raise ValueError("DLPD-12 source clock count changed")
    if set(frame["candidate"]) != {POLICY_ID}:
        raise ValueError("DLPD-12 source clock identity changed")
    if set(frame["control"]) != set(dlpd.CONTROLS):
        raise ValueError("DLPD-12 source-only control family changed")
    return frame


def derive_evaluation_clocks() -> pd.DataFrame:
    """Derive evaluation-only controls without opening BTC outcomes."""
    source = _read_source_clocks()
    primary = cast(pd.DataFrame, source.loc[source["control"].eq("primary")].copy())
    parts = [source]
    for control in DERIVED_CONTROLS:
        frame = primary.copy()
        frame["control"] = control
        if control == "direction_flip":
            frame["side"] = -frame["side"].astype(int)
        elif control == "deterministic_random_side":
            frame["side"] = frame["decision_time"].map(_deterministic_random_side)
        elif control == "extra_latency_1h":
            frame["entry_time"] = frame["entry_time"] + pd.Timedelta(hours=1)
            frame["exit_time"] = frame["exit_time"] + pd.Timedelta(hours=1)
        parts.append(frame)
    clocks = pd.concat(parts, ignore_index=True)
    clocks = clocks.sort_values(["control", "entry_time"], kind="mergesort").reset_index(
        drop=True
    )
    return clocks.loc[:, list(EVALUATION_CLOCK_COLUMNS)]


def _write_evaluation_clocks(frame: pd.DataFrame, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        deterministic_gzip_csv(frame, temporary)
        rebuilt = temporary.read_bytes()
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != rebuilt:
                raise RuntimeError("refusing to overwrite frozen DLPD evaluation clock")
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(output)


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "control": str(row["control"]),
            "split": str(row["split"]),
            "decision_time": _utc(row["decision_time"]).isoformat(),
            "entry_time": _utc(row["entry_time"]).isoformat(),
            "exit_time": _utc(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def load_schedules(
    *,
    clock_path: str | Path = EVALUATION_CLOCKS,
    expected_clock_sha256: str | None = None,
) -> dict[str, pd.DataFrame]:
    path = Path(clock_path)
    if not path.exists():
        raise FileNotFoundError("DLPD-12 evaluation clock is not frozen")
    if expected_clock_sha256 is not None and _sha256(path) != expected_clock_sha256:
        raise ValueError("DLPD-12 frozen evaluation clock hash changed")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != EVALUATION_CLOCK_COLUMNS:
        raise ValueError("DLPD-12 evaluation clock schema changed")
    frame["split"] = frame["split"].astype(str)
    for column in (
        "source_hour_start",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if set(frame["control"]) != set(ALL_CONTROLS):
        raise ValueError("DLPD-12 evaluation control family changed")
    schedules: dict[str, pd.DataFrame] = {}
    for control in ALL_CONTROLS:
        schedule = cast(
            pd.DataFrame,
            frame.loc[frame["control"].eq(control)].copy(),
        ).sort_values("entry_time", kind="mergesort").reset_index(drop=True)
        if not schedule["side"].isin((-1, 1)).all():
            raise ValueError(f"DLPD-12 {control} has invalid side")
        entries = cast(pd.Series, schedule["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, schedule["exit_time"]).reset_index(drop=True)
        if len(schedule) > 1 and not bool(
            entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()
        ):
            raise ValueError(f"DLPD-12 {control} schedule overlaps")
        if not bool((exits - entries).eq(pd.Timedelta(hours=12)).all()):
            raise ValueError(f"DLPD-12 {control} hold changed")
        expected_delay = pd.Timedelta(
            minutes=65 if control == "extra_latency_1h" else 5
        )
        if not bool(
            (entries - schedule["decision_time"].reset_index(drop=True))
            .eq(expected_delay)
            .all()
        ):
            raise ValueError(f"DLPD-12 {control} latency changed")
        if not bool(schedule["feature_available_time"].le(schedule["entry_time"]).all()):
            raise ValueError(f"DLPD-12 {control} uses unavailable features")
        for split, stage in (("2022", "train"), ("2023", "test")):
            start, end = STAGE_WINDOWS[stage]
            subset = cast(pd.DataFrame, schedule.loc[schedule["split"].eq(split)])
            if not (
                bool(subset["entry_time"].ge(start).all())
                and bool(subset["exit_time"].le(end).all())
            ):
                raise ValueError(f"DLPD-12 {control}/{split} containment changed")
        schedules[control] = schedule

    primary = schedules["primary"].reset_index(drop=True)
    for control in DERIVED_SAME_CLOCK_CONTROLS:
        other = schedules[control].reset_index(drop=True)
        if not other[["decision_time", "entry_time", "exit_time"]].equals(
            primary[["decision_time", "entry_time", "exit_time"]]
        ):
            raise ValueError(f"DLPD-12 {control} changed the primary clock")
    if not schedules["direction_flip"]["side"].reset_index(drop=True).eq(
        -primary["side"].reset_index(drop=True)
    ).all():
        raise ValueError("DLPD-12 direction-flip semantics changed")
    random_expected = (
        primary["decision_time"]
        .map(_deterministic_random_side)
        .reset_index(drop=True)
    )
    if not (
        schedules["deterministic_random_side"]["side"]
        .reset_index(drop=True)
        .eq(random_expected)
        .all()
    ):
        raise ValueError("DLPD-12 random-side semantics changed")
    delayed = schedules["extra_latency_1h"].reset_index(drop=True)
    if not delayed["entry_time"].eq(primary["entry_time"] + pd.Timedelta(hours=1)).all():
        raise ValueError("DLPD-12 latency-control entry changed")
    if not delayed["exit_time"].eq(primary["exit_time"] + pd.Timedelta(hours=1)).all():
        raise ValueError("DLPD-12 latency-control exit changed")
    if not delayed["side"].eq(primary["side"]).all():
        raise ValueError("DLPD-12 latency-control side changed")

    expected = derive_evaluation_clocks()
    semantic = ["control", "split", "decision_time", "entry_time", "exit_time", "side"]
    actual_semantics = (
        frame[semantic]
        .sort_values(["control", "entry_time"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_semantics = (
        expected[semantic]
        .sort_values(["control", "entry_time"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not actual_semantics.equals(expected_semantics):
        raise ValueError("DLPD-12 frozen evaluation-clock semantics changed")
    return schedules


def _window_schedule(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage not in STAGE_SPLITS:
        raise FileNotFoundError(f"DLPD-12 {stage} signal source remains sealed")
    start, end = STAGE_WINDOWS[stage]
    selected = cast(
        pd.DataFrame,
        frame.loc[frame["split"].eq(STAGE_SPLITS[stage])].copy(),
    )
    if not (
        bool(selected["entry_time"].ge(start).all())
        and bool(selected["exit_time"].le(end).all())
    ):
        raise ValueError(f"DLPD-12 {stage} schedule crosses its window")
    return selected.sort_values("entry_time", kind="mergesort").reset_index(drop=True)


def _stage_source_spec(stage: str) -> dict[str, Any]:
    start, end = STAGE_WINDOWS[stage]
    return {
        "stage": stage,
        "required_manifest": str(STAGE_SOURCE_MANIFESTS[stage]),
        "required_protocol_version": "btcdom_leverage_polarity_execution_source_v1",
        "physical_window": [start.isoformat(), end.isoformat()],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": False,
        "strategy_outcomes_calculated": False,
    }


def _future_signal_spec(stage: str) -> dict[str, Any]:
    start, end = STAGE_WINDOWS[stage]
    return {
        "stage": stage,
        "required_manifest": str(FUTURE_SIGNAL_MANIFESTS[stage]),
        "physical_window": [start.isoformat(), end.isoformat()],
        "policy_source": str(dlpd.PREREGISTRATION_SOURCE),
        "policy_source_sha256": STATIC_INPUT_SHA256[str(dlpd.PREREGISTRATION_SOURCE)],
        "outcomes_opened": False,
    }


def freeze_evaluator(
    output_path: str | Path = EVALUATOR_FREEZE,
    *,
    evaluation_clock_path: str | Path = EVALUATION_CLOCKS,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("DLPD-12 evaluator freeze is write-once")
    if any(path.exists() for path in STAGE_OUTPUTS.values()):
        raise RuntimeError("DLPD-12 cannot freeze after an outcome stage exists")
    prereg, support = _verify_static_inputs()
    clock_sha = _write_evaluation_clocks(derive_evaluation_clocks(), evaluation_clock_path)
    schedules = load_schedules(
        clock_path=evaluation_clock_path,
        expected_clock_sha256=clock_sha,
    )
    records = {
        name: {
            "events": len(schedule),
            "schedule_hash": _schedule_hash(schedule),
            "stage_counts": {
                stage: len(_window_schedule(schedule, stage))
                for stage in STAGE_SPLITS
            },
            "first_entry": _utc(schedule["entry_time"].min()).isoformat(),
            "last_exit": _utc(schedule["exit_time"].max()).isoformat(),
        }
        for name, schedule in schedules.items()
    }
    core = {
        "protocol_version": "btcdom_leverage_polarity_evaluator_v1",
        "candidate": POLICY_ID,
        "as_of_date": "2026-07-20",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "support_manifest_hash": support["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_dependency": str(strict_engine.EVALUATOR_SOURCE),
        "strict_engine_dependency_sha256": STATIC_INPUT_SHA256[str(strict_engine.EVALUATOR_SOURCE)],
        "strict_source_dependency": str(strict_source.EVALUATOR_SOURCE),
        "strict_source_dependency_sha256": STATIC_INPUT_SHA256[str(strict_source.EVALUATOR_SOURCE)],
        "evaluation_config": asdict(FROZEN_CONFIG),
        "static_inputs": STATIC_INPUT_SHA256,
        "evaluation_clock": str(evaluation_clock_path),
        "evaluation_clock_sha256": clock_sha,
        "schedule_records": records,
        "phase_scope": ["train", "test"],
        "execution_source_specs": {
            stage: _stage_source_spec(stage) for stage in ("train", "test")
        },
        "future_signal_source_specs": {
            stage: _future_signal_spec(stage) for stage in ("eval", "final")
        },
        "phase_two_requirement": (
            "after train and test pass, freeze a separate immutable eval/final "
            "signal-and-execution evaluator before opening any 2024+ outcome row"
        ),
        "legacy_container_contract": {
            "eligible_stages": ["train", "test"],
            "market": {
                "path": str(LEGACY_MARKET),
                "sha256": LEGACY_MARKET_SHA256,
                "manifest": str(LEGACY_MARKET_MANIFEST),
                "manifest_sha256": LEGACY_MARKET_MANIFEST_SHA256,
            },
            "funding": {
                "path": str(LEGACY_FUNDING),
                "sha256": LEGACY_FUNDING_SHA256,
                "manifest": str(LEGACY_FUNDING_MANIFEST),
                "manifest_sha256": LEGACY_FUNDING_MANIFEST_SHA256,
            },
        },
        "stage_windows": {
            name: [start.isoformat(), end.isoformat()]
            for name, (start, end) in STAGE_WINDOWS.items()
        },
        "half_windows": {
            stage: {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in windows.items()
            }
            for stage, windows in HALF_WINDOWS.items()
        },
        "outcome_gate": prereg["conditional_outcome_gate"],
        "source_only_controls": list(dlpd.SOURCE_ONLY_CONTROLS),
        "derived_controls": list(DERIVED_CONTROLS),
        "strict_accounting": {
            "funding_boundary": (
                "interior symmetric; exact entry/exit credits dropped and debits "
                "retained"
            ),
            "mdd": (
                "global/pre-entry HWM; costs; exact funding marks; every held 5m "
                "favorable-then-adverse OHLC path"
            ),
            "cagr": "full declared split calendar including idle cash",
            "entry_exit": (
                "next 5m open after completed-hour decision; exact 12h open-to-open hold"
            ),
        },
        "opened_windows": [],
        "sealed_windows": list(STAGE_ORDER),
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "execution_outcome_data_bytes_hashed_during_freeze": False,
        "simulation_run_during_freeze": False,
        "mutable_parameters": [],
    }
    report = _seal(core)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_manifest(payload, label="evaluator freeze")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("DLPD-12 evaluator source changed after freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("DLPD-12 evaluator froze another support commit")
    if payload.get("evaluation_config") != asdict(FROZEN_CONFIG):
        raise ValueError("DLPD-12 evaluator configuration changed")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("DLPD-12 evaluator freeze is not sealed")
    if payload.get("sealed_windows") != list(STAGE_ORDER):
        raise ValueError("DLPD-12 evaluator stage seal changed")
    for field in ("execution_ohlc_rows_parsed_during_freeze", "funding_rows_parsed_during_freeze"):
        if payload.get(field) != 0:
            raise ValueError(f"DLPD-12 evaluator freeze changed {field}")
    if payload.get("execution_outcome_data_bytes_hashed_during_freeze") is not False:
        raise ValueError("DLPD-12 evaluator freeze hashed outcomes")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("DLPD-12 evaluator freeze simulated outcomes")
    prereg, support = _verify_static_inputs()
    if payload.get("outcome_gate") != prereg["conditional_outcome_gate"]:
        raise ValueError("DLPD-12 evaluator outcome gate changed")
    if payload.get("support_manifest_hash") != support["manifest_hash"]:
        raise ValueError("DLPD-12 evaluator support binding changed")
    clock_path = Path(cast(str, payload["evaluation_clock"]))
    schedules = load_schedules(
        clock_path=clock_path,
        expected_clock_sha256=cast(str, payload["evaluation_clock_sha256"]),
    )
    for name, schedule in schedules.items():
        if payload["schedule_records"][name]["schedule_hash"] != _schedule_hash(schedule):
            raise ValueError(f"DLPD-12 {name} schedule changed after freeze")
    return payload


def _verified_prior_reports(stage: str, *, freeze_hash: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for prior in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        payload = _load_json(STAGE_OUTPUTS[prior])
        _verify_manifest(payload, label=f"stored {prior}")
        if payload.get("stage") != prior or payload.get("stage_passed") is not True:
            raise ValueError(f"DLPD-12 {prior} did not pass; {stage} remains sealed")
        index = STAGE_ORDER.index(prior)
        if payload.get("opened_windows") != list(STAGE_ORDER[: index + 1]):
            raise ValueError(f"DLPD-12 {prior} opened an unexpected window")
        if payload.get("sealed_windows") != list(STAGE_ORDER[index + 1 :]):
            raise ValueError(f"DLPD-12 {prior} stage seal changed")
        if payload.get("evaluator_freeze_manifest_hash") != freeze_hash:
            raise ValueError(f"DLPD-12 {prior} froze another evaluator")
        if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
            raise ValueError(f"DLPD-12 {prior} evaluator source changed")
        reports.append(payload)
    return reports


def _slice_gzip_csv(
    source: str | Path,
    output: str | Path,
    *,
    timestamp_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Copy only [start, end) rows while parsing timestamps, never values."""
    source_path = Path(source)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"DLPD-12 stage source is write-once: {output_path}")
    rows = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(source_path, "rt", encoding="utf-8", newline="") as input_handle:
            header_line = input_handle.readline()
            header = header_line.rstrip("\r\n").split(",")
            timestamp_index = header.index(timestamp_column)
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", mtime=0
                ) as compressed:
                    compressed.write(header_line.encode("utf-8"))
                    for line in input_handle:
                        fields = line.rstrip("\r\n").split(",")
                        timestamp = _utc(fields[timestamp_index])
                        if timestamp < start:
                            continue
                        if timestamp >= end:
                            break
                        if last is not None and timestamp <= last:
                            raise ValueError(
                                "DLPD-12 source timestamps are not strictly increasing"
                            )
                        compressed.write(line.encode("utf-8"))
                        first = timestamp if first is None else first
                        last = timestamp
                        rows += 1
        if rows == 0 or first is None or last is None:
            raise ValueError("DLPD-12 stage source slice is empty")
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"DLPD-12 stage source is write-once: {output_path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "rows": rows,
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
    }


def prepare_stage_source(stage: str) -> dict[str, Any]:
    if stage not in ("train", "test"):
        raise ValueError("DLPD-12 eval/final require separately frozen official sources")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    manifest_path = STAGE_SOURCE_MANIFESTS[stage]
    if manifest_path.exists():
        raise FileExistsError(f"DLPD-12 {stage} execution source is write-once")
    if _sha256(LEGACY_MARKET) != LEGACY_MARKET_SHA256:
        raise ValueError("DLPD-12 legacy market container bytes changed")
    if _sha256(LEGACY_FUNDING) != LEGACY_FUNDING_SHA256:
        raise ValueError("DLPD-12 legacy funding container bytes changed")
    start, end = STAGE_WINDOWS[stage]
    directory = STAGE_SOURCE_DIRS[stage]
    market_path = directory / "BTCUSDT_5m.csv.gz"
    funding_path = directory / "BTCUSDT_funding_marks.csv.gz"
    created: list[Path] = []
    try:
        market = _slice_gzip_csv(
            LEGACY_MARKET,
            market_path,
            timestamp_column="date",
            start=start,
            end=end,
        )
        created.append(market_path)
        funding = _slice_gzip_csv(
            LEGACY_FUNDING,
            funding_path,
            timestamp_column="funding_time_utc",
            start=start,
            end=end,
        )
        created.append(funding_path)
        parsed_market, market_diagnostics = strict_source._parse_market_window(
            market_path,
            start,
            end,
            require_exact_physical_window=True,
            include_end_boundary=False,
        )
        parsed_funding, funding_diagnostics = strict_source._parse_funding_window(
            funding_path,
            start,
            end,
            require_exact_physical_window=True,
            include_end_boundary=False,
        )
        if len(parsed_market) != market["rows"] or len(parsed_funding) != funding["rows"]:
            raise ValueError("DLPD-12 stage-source validation count changed")
        spec = _stage_source_spec(stage)
        core = {
            "protocol_version": spec["required_protocol_version"],
            "candidate": POLICY_ID,
            "stage": stage,
            "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": False,
            "strategy_outcomes_calculated": False,
            "official_checksums_verified": True,
            "full_parent_containers_hashed": True,
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": {
                "path": str(LEGACY_MARKET),
                "sha256": LEGACY_MARKET_SHA256,
                "manifest": str(LEGACY_MARKET_MANIFEST),
                "manifest_sha256": LEGACY_MARKET_MANIFEST_SHA256,
            },
            "parent_funding": {
                "path": str(LEGACY_FUNDING),
                "sha256": LEGACY_FUNDING_SHA256,
                "manifest": str(LEGACY_FUNDING_MANIFEST),
                "manifest_sha256": LEGACY_FUNDING_MANIFEST_SHA256,
            },
            "market": {**market, "diagnostics": market_diagnostics},
            "funding": {**funding, "diagnostics": funding_diagnostics},
        }
        report = _seal(core)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        return report
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _load_stage_source(stage: str) -> dict[str, Any]:
    spec = _stage_source_spec(stage)
    payload = _load_json(STAGE_SOURCE_MANIFESTS[stage])
    _verify_manifest(payload, label=f"{stage} execution source")
    if payload.get("protocol_version") != spec["required_protocol_version"]:
        raise ValueError(f"DLPD-12 {stage} source protocol changed")
    if payload.get("candidate") != POLICY_ID or payload.get("stage") != stage:
        raise ValueError(f"DLPD-12 {stage} source identity changed")
    for key in (
        "physical_window",
        "physical_rows_limited_to_window",
        "exit_boundary_required",
        "strategy_outcomes_calculated",
    ):
        if payload.get(key) != spec[key]:
            raise ValueError(f"DLPD-12 {stage} source {key} changed")
    if payload.get("official_checksums_verified") is not True:
        raise ValueError(f"DLPD-12 {stage} source lacks checksum verification")
    for name in ("market", "funding"):
        item = payload.get(name)
        if not isinstance(item, dict) or set(item) < {"path", "sha256"}:
            raise ValueError(f"DLPD-12 {stage} source lacks {name} identity")
    return payload


def load_execution_window(stage: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"DLPD-12 unknown stage: {stage}")
    if stage not in STAGE_SPLITS:
        raise RuntimeError(
            "DLPD-12 eval/final are phase-two sealed; no execution source may open"
        )
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    contract = _load_stage_source(stage)
    if contract.get("evaluator_freeze_manifest_hash") != freeze["manifest_hash"]:
        raise ValueError(f"DLPD-12 {stage} source froze another evaluator")
    start, end = STAGE_WINDOWS[stage]
    market_item = contract["market"]
    funding_item = contract["funding"]
    market, market_diagnostics = strict_source._parse_market_window(
        market_item["path"],
        start,
        end,
        require_exact_physical_window=True,
        include_end_boundary=False,
    )
    funding, funding_diagnostics = strict_source._parse_funding_window(
        funding_item["path"],
        start,
        end,
        require_exact_physical_window=True,
        include_end_boundary=False,
    )
    if _sha256(market_item["path"]) != market_item["sha256"]:
        raise ValueError(f"DLPD-12 {stage} market bytes changed")
    if _sha256(funding_item["path"]) != funding_item["sha256"]:
        raise ValueError(f"DLPD-12 {stage} funding bytes changed")
    return market, funding, {
        "stage": stage,
        "physical_window": [start.isoformat(), end.isoformat()],
        "market": market_diagnostics,
        "funding": funding_diagnostics,
        "market_sha256": market_item["sha256"],
        "funding_sha256": funding_item["sha256"],
    }


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate_per_side: float,
    cfg: EvaluationConfig = FROZEN_CONFIG,
) -> dict[str, Any]:
    if cfg != FROZEN_CONFIG:
        raise ValueError("DLPD-12 evaluation configuration is frozen")
    _verify_config()
    return strict_engine.simulate_strict(
        market,
        funding,
        clocks,
        start=start,
        end=end,
        cost_rate_per_side=cost_rate_per_side,
        cfg=strict_engine.EvaluationConfig(**asdict(cfg)),
    )


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    significance = metrics["weekly_cluster_signflip"]
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "longs": metrics["longs"],
        "shorts": metrics["shorts"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "weekly_cluster_signflip_p": significance["p_value_two_sided"],
        "weekly_clusters": significance["cluster_count"],
        "weekly_test_method": significance["method"],
    }


def _simulate_window(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost: float,
) -> dict[str, Any]:
    selected = cast(
        pd.DataFrame,
        schedule.loc[schedule["entry_time"].ge(start) & schedule["exit_time"].le(end)].copy(),
    )
    return simulate_strict(
        market,
        funding,
        selected,
        start=start,
        end=end,
        cost_rate_per_side=cost,
    )


def _stage_gates(
    stage: str,
    base: dict[str, Any],
    stress: dict[str, Any],
    halves: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, bool]:
    gate = prereg["conditional_outcome_gate"]
    each = gate["each_opened_stage"]
    checks = {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"]
        >= float(each["cagr_to_strict_mdd_at_least"]),
        "strict_mdd_at_most_15pct": base["strict_mdd_pct"]
        <= float(each["strict_mdd_at_most"]) * 100.0,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "contained_subperiods_positive": all(
            item["absolute_return_pct"] > 0.0 for item in halves.values()
        ),
        "weekly_cluster_signflip_p_at_most_10pct": base["weekly_cluster_signflip"][
            "p_value_two_sided"
        ]
        <= float(each["weekly_cluster_signflip_p_at_most"]),
        "direction_flip_inferior": controls["direction_flip"]["cagr_to_strict_mdd"]
        < base["cagr_to_strict_mdd"],
    }
    if stage in ("train", "test"):
        checks["minimum_120_trades"] = base["trades"] >= int(
            gate["minimum_train_and_test_trades_each"]
        )
    return checks


def _build_stage_report(stage: str) -> dict[str, Any]:
    if stage not in STAGE_SPLITS:
        raise RuntimeError(
            "DLPD-12 eval/final require a separately frozen phase-two evaluator"
        )
    freeze = verify_evaluator_freeze()
    prereg, _ = _verify_static_inputs()
    prior = _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    schedules = load_schedules(
        clock_path=cast(str, freeze["evaluation_clock"]),
        expected_clock_sha256=cast(str, freeze["evaluation_clock_sha256"]),
    )
    start, end = STAGE_WINDOWS[stage]
    primary_schedule = _window_schedule(schedules["primary"], stage)
    market, funding, diagnostics = load_execution_window(stage)
    base = simulate_strict(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
    )
    stress = simulate_strict(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate_per_side=FROZEN_CONFIG.stress_cost_notional_per_side,
    )
    halves = {
        name: _simulate_window(
            market,
            funding,
            primary_schedule,
            start=half_start,
            end=half_end,
            cost=FROZEN_CONFIG.base_cost_notional_per_side,
        )
        for name, (half_start, half_end) in HALF_WINDOWS[stage].items()
    }
    controls = {
        name: simulate_strict(
            market,
            funding,
            _window_schedule(schedule, stage),
            start=start,
            end=end,
            cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
        )
        for name, schedule in schedules.items()
        if name != "primary"
    }
    gates = _stage_gates(stage, base, stress, halves, controls, prereg)
    passed = all(gates.values())
    index = STAGE_ORDER.index(stage)
    core = {
        "protocol_version": "btcdom_leverage_polarity_stage_v1",
        "candidate": POLICY_ID,
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_prior_stage_manifest_hashes": {
            row["stage"]: row["manifest_hash"] for row in prior
        },
        "config": asdict(FROZEN_CONFIG),
        "execution_diagnostics": diagnostics,
        "primary": {
            "metrics": base,
            "headline": _headline(base),
            "stress_metrics": stress,
            "stress_headline": _headline(stress),
            "contained_subperiod_metrics": halves,
            "contained_subperiod_headlines": {
                name: _headline(item) for name, item in halves.items()
            },
        },
        "controls": {
            name: {"metrics": item, "headline": _headline(item)}
            for name, item in controls.items()
        },
        "component_controls_are_diagnostic_only": True,
        "component_controls_cannot_repair_primary": True,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "stage_passed": passed,
        "opened_windows": list(STAGE_ORDER[: index + 1]),
        "sealed_windows": list(STAGE_ORDER[index + 1 :]),
        "disposition": (
            f"ADVANCE_TO_{STAGE_ORDER[index + 1].upper()}"
            if passed and index + 1 < len(STAGE_ORDER)
            else "QUALIFIED_FOR_POST_PASS_AUDIT"
            if passed
            else "REJECT_NO_REPAIR"
        ),
    }
    return _seal(core)


def _metric_row(label: str, item: dict[str, Any]) -> str:
    return (
        f"| {label} | {item['absolute_return_pct']:.2f}% | {item['cagr_pct']:.2f}% | "
        f"{item['strict_mdd_pct']:.2f}% | {item['cagr_to_strict_mdd']:.2f} | "
        f"{item['trades']} | {item['longs']}/{item['shorts']} | "
        f"{item['mean_gross_underlying_bp']:.2f}bp | "
        f"{item['weekly_cluster_signflip_p']:.4f} |"
    )


def render_stage_doc(report: dict[str, Any]) -> str:
    lines = [
        f"# DLPD-12 {report['stage']} strict result — 2026-07-20",
        "",
        "Absolute return and CAGR use the full declared calendar. Strict MDD uses "
        "the global/pre-entry HWM, exact funding, costs, and every held five-minute "
        "favorable-then-adverse path.",
        "",
        "| Clock | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("primary", report["primary"]["headline"]),
        _metric_row("primary 10bp stress", report["primary"]["stress_headline"]),
    ]
    for name, item in report["controls"].items():
        lines.append(_metric_row(name, item["headline"]))
    lines.extend(
        [
            "",
            f"- Stage passed: **{report['stage_passed']}**",
            f"- Failed gates: `{report['failed_gates']}`",
            f"- Disposition: `{report['disposition']}`",
            "- Component controls are diagnostic only and cannot repair primary.",
            "",
            "## Contained subperiods",
            "",
            "| Window | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["primary"]["contained_subperiod_headlines"].items():
        lines.append(_metric_row(name, item))
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- evaluator SHA-256: `{report['evaluator_source_sha256']}`",
            f"- report manifest: `{report['manifest_hash']}`",
            f"- physical source window: `{report['execution_diagnostics']['physical_window']}`",
            f"- still sealed: `{report['sealed_windows']}`",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_stage(stage: str) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"DLPD-12 unknown stage: {stage}")
    output = STAGE_OUTPUTS[stage]
    document = STAGE_DOCS[stage]
    if output.exists() or document.exists():
        raise FileExistsError(f"DLPD-12 {stage} result is write-once")
    report = _build_stage_report(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with document.open("x", encoding="utf-8") as handle:
        handle.write(render_stage_doc(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--prepare-stage-source", choices=STAGE_ORDER)
    group.add_argument("--stage", choices=STAGE_ORDER)
    args = parser.parse_args()
    if args.freeze:
        report = freeze_evaluator()
        summary = {
            "manifest_hash": report["manifest_hash"],
            "evaluator_source_sha256": report["evaluator_source_sha256"],
            "evaluation_clock_sha256": report["evaluation_clock_sha256"],
            "opened_windows": report["opened_windows"],
            "sealed_windows": report["sealed_windows"],
        }
    elif args.prepare_stage_source is not None:
        report = prepare_stage_source(args.prepare_stage_source)
        summary = {
            "stage": report["stage"],
            "manifest_hash": report["manifest_hash"],
            "market": report["market"],
            "funding": report["funding"],
            "strategy_outcomes_calculated": report["strategy_outcomes_calculated"],
        }
    else:
        assert args.stage is not None
        report = evaluate_stage(args.stage)
        summary = {
            "stage": report["stage"],
            "passed": report["stage_passed"],
            "headline": report["primary"]["headline"],
            "failed_gates": report["failed_gates"],
            "disposition": report["disposition"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
