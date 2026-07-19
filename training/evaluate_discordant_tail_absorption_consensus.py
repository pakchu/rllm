"""Sequential hardened evaluator for frozen DTAC-8 clocks.

The evaluator freeze opens no BTC execution OHLC or funding row. Outcome
stages are physically and logically sequential: train -> test -> eval ->
final, with every later loader blocked unless all prior frozen gates passed.
"""

from __future__ import annotations

import argparse
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

from training import evaluate_stablecoin_quote_flow_diffusion as strict_source  # noqa: E402
from training import preregister_discordant_tail_absorption_consensus as prereg  # noqa: E402
from training.build_six_alt_price_free_flow_panel import (  # noqa: E402
    deterministic_gzip_csv,
)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("DTAC-8 timestamp is NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


POLICY_ID = prereg.POLICY_ID
SUPPORT_COMMIT = "60b889c9e50fd8a365bcbf1b398635303393bc6d"
SUPPORT_RESULT = Path(
    "results/discordant_tail_absorption_consensus_support_2026-07-19.json"
)
PRIMARY_CLOCKS = Path(
    "data/discordant_tail_absorption_consensus_clocks_2023_2026.csv.gz"
)
CONTROL_CLOCKS = Path(
    "data/discordant_tail_absorption_consensus_control_clocks_2023_2026.csv.gz"
)
EVALUATOR_SOURCE = Path("training/evaluate_discordant_tail_absorption_consensus.py")
EVALUATOR_FREEZE = Path(
    "results/discordant_tail_absorption_consensus_evaluator_freeze_2026-07-19.json"
)

TRAIN_MARKET = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
TRAIN_MARKET_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
TRAIN_MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
TRAIN_MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
TRAIN_FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
TRAIN_FUNDING_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
TRAIN_FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
TRAIN_FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

STATIC_INPUT_SHA256 = {
    str(prereg.PREREGISTRATION_SOURCE): (
        "092302ca733c9498a5472e55e6a9868fbaa0e7849be26900af4c48237248744b"
    ),
    str(SUPPORT_RESULT): (
        "b3b34619443b92a458f0babece588881bd2079d91828d0af034a3a988777fe9e"
    ),
    "docs/discordant-tail-absorption-consensus-preregistration-2026-07-19.md": (
        "3f8833f84429a343b5ddd6eb4a4fdc7192f87420c40d29eb5b9138c7f4fcd870"
    ),
    str(PRIMARY_CLOCKS): (
        "c71685bb6285c07e90e328ffe2f69a11de37445f8113a47d7ffee6ef16eece79"
    ),
    str(prereg.FLOW_PANEL): prereg.FLOW_PANEL_SHA256,
    str(prereg.FLOW_MANIFEST): prereg.FLOW_MANIFEST_SHA256,
    str(prereg.PREMIUM_SUMMARY): prereg.PREMIUM_SUMMARY_SHA256,
    **{
        str(prereg.premium_path(symbol)): digest
        for symbol, digest in prereg.PREMIUM_SHA256.items()
    },
    str(TRAIN_MARKET_MANIFEST): TRAIN_MARKET_MANIFEST_SHA256,
    str(TRAIN_FUNDING_MANIFEST): TRAIN_FUNDING_MANIFEST_SHA256,
    "training/evaluate_stablecoin_quote_flow_diffusion.py": (
        "0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b"
    ),
}

STAGE_ORDER = ("train", "test", "eval", "final")
STAGE_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    name: (_utc(start), _utc(end)) for name, (start, end) in prereg.SPLITS.items()
}
HALF_WINDOWS: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {
    "train": {
        "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
        "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
    },
    "test": {
        "2024_h1": (_utc("2024-01-01"), _utc("2024-07-01")),
        "2024_h2": (_utc("2024-07-01"), _utc("2025-01-01")),
    },
    "eval": {
        "2025_h1": (_utc("2025-01-01"), _utc("2025-07-01")),
        "2025_h2": (_utc("2025-07-01"), _utc("2026-01-01")),
    },
    "final": {
        "2026_q1": (_utc("2026-01-01"), _utc("2026-04-01")),
        "2026_q2": (_utc("2026-04-01"), _utc("2026-06-01")),
    },
}
STAGE_OUTPUTS = {
    "train": Path(
        "results/discordant_tail_absorption_consensus_train_2023_2026-07-19.json"
    ),
    "test": Path(
        "results/discordant_tail_absorption_consensus_test_2024_2026-07-19.json"
    ),
    "eval": Path(
        "results/discordant_tail_absorption_consensus_eval_2025_2026-07-19.json"
    ),
    "final": Path(
        "results/discordant_tail_absorption_consensus_final_2026h1_2026-07-19.json"
    ),
}
STAGE_DOCS = {
    stage: Path(
        f"docs/discordant-tail-absorption-consensus-{stage}-result-2026-07-19.md"
    )
    for stage in STAGE_ORDER
}
FUTURE_SOURCE_MANIFESTS = {
    stage: Path(
        f"results/discordant_tail_absorption_consensus_{stage}_execution_source_2026-07-19.json"
    )
    for stage in STAGE_ORDER[1:]
}

SAME_CLOCK_CONTROLS = (
    "direction_flip",
    "all_six_premium_side",
    "all_six_flow_fade_side",
    "deterministic_random_side",
)
INDEPENDENT_CLOCK_CONTROLS = (
    "symbol_permuted_premium_pairing",
    "stale_premium_pairing_24h",
)
LATENCY_CONTROLS = ("extra_latency_1h",)
MECHANISM_CONTROLS = (
    *SAME_CLOCK_CONTROLS,
    *INDEPENDENT_CLOCK_CONTROLS,
    *LATENCY_CONTROLS,
)
ALL_CONTROLS = ("primary", *MECHANISM_CONTROLS)
CONTROL_CLOCK_COLUMNS = ("control", *prereg.EVENT_COLUMNS)
BAR = pd.Timedelta(minutes=5)
YEAR_SECONDS = 365.25 * 86_400.0


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 96
    exact_cluster_max: int = 20
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_719


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
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"DTAC-8 expected a JSON object: {path}")
    return payload


def _verify_manifest(payload: dict[str, Any], *, label: str) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError(f"DTAC-8 {label} manifest hash changed")


def _verify_static_inputs() -> dict[str, Any]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"DTAC-8 frozen input changed: {path}")
    support = _load_json(SUPPORT_RESULT)
    _verify_manifest(support, label="support")
    if support.get("candidate") != POLICY_ID:
        raise ValueError("DTAC-8 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("DTAC-8 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("DTAC-8 support opened an outcome source")
    if support.get("support_passed") is not True:
        raise ValueError("DTAC-8 source support did not pass")
    if support.get("clock_sha256") != STATIC_INPUT_SHA256[str(PRIMARY_CLOCKS)]:
        raise ValueError("DTAC-8 support no longer binds its clock")
    gate = support["protocol"]["outcome_gate"]
    cfg = EvaluationConfig()
    statistical = gate["statistical_test"]
    expected = {
        "leverage": support["protocol"]["eventual_execution"]["leverage"],
        "base_cost_notional_per_side": 0.0006,
        "stress_cost_notional_per_side": gate["stress_cost_notional_per_side"],
        "hold_bars": 8 * 12,
        "exact_cluster_max": statistical["exact_cluster_max"],
        "cluster_draws": statistical["monte_carlo_draws"],
        "cluster_seed": statistical["seed"],
    }
    if asdict(cfg) != expected:
        raise ValueError("DTAC-8 evaluation configuration drifted")
    return support


def _deterministic_random_side(decision_time: Any) -> int:
    timestamp = _utc(decision_time).strftime("%Y-%m-%dT%H:%M:%SZ")
    first_nibble = int(
        hashlib.sha256(f"{POLICY_ID}|{timestamp}".encode()).hexdigest()[0], 16
    )
    return 1 if first_nibble % 2 == 0 else -1


def _score_side(score: float, primary_side: int) -> int:
    if not np.isfinite(score):
        raise ValueError("DTAC-8 control score is non-finite")
    if score > 0.0:
        return 1
    if score < 0.0:
        return -1
    return primary_side


def _primary_clock() -> pd.DataFrame:
    support = _verify_static_inputs()
    frame = pd.read_csv(PRIMARY_CLOCKS)
    if tuple(frame.columns) != prereg.EVENT_COLUMNS:
        raise ValueError("DTAC-8 primary clock schema changed")
    for column in (
        "source_hour_open_utc",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if len(frame) != int(support["clock_rows"]):
        raise ValueError("DTAC-8 primary clock count changed")
    if set(frame["candidate"]) != {POLICY_ID}:
        raise ValueError("DTAC-8 primary clock identity changed")
    return frame


def derive_control_clocks() -> pd.DataFrame:
    """Derive all source-only controls without opening BTC outcomes."""
    primary = _primary_clock()
    support = _verify_static_inputs()
    selected = support["selected"]
    cfg = prereg.Config()
    flow = prereg.flow_matrix(prereg.load_flow_prefix(end_exclusive=None))
    premium = prereg.load_premium_prefix(end_exclusive=None)
    records: list[dict[str, Any]] = []
    for raw in primary.to_dict(orient="records"):
        decision = _utc(raw["decision_time"])
        naive = decision.tz_localize(None)
        primary_side = int(raw["side"])
        current_flow = cast(pd.Series, flow.loc[naive])
        current_premium = cast(pd.Series, premium.loc[naive])
        sides = {
            "primary": primary_side,
            "direction_flip": -primary_side,
            "all_six_premium_side": _score_side(
                float(current_premium.mean()), primary_side
            ),
            "all_six_flow_fade_side": _score_side(
                -float(current_flow.mean()), primary_side
            ),
            "deterministic_random_side": _deterministic_random_side(decision),
            "extra_latency_1h": primary_side,
        }
        for control in ("primary", *SAME_CLOCK_CONTROLS, *LATENCY_CONTROLS):
            row = dict(raw)
            row["control"] = control
            row["side"] = sides[control]
            if control == "extra_latency_1h":
                row["entry_time"] = _utc(row["entry_time"]) + pd.Timedelta(hours=1)
                row["exit_time"] = _utc(row["exit_time"]) + pd.Timedelta(hours=1)
            records.append(row)

    permuted_premium = pd.DataFrame(
        np.roll(premium.to_numpy(float), 1, axis=1),
        index=premium.index,
        columns=premium.columns,
    )
    source_controls = {
        "symbol_permuted_premium_pairing": permuted_premium,
        "stale_premium_pairing_24h": premium.shift(24),
    }
    for control, control_premium in source_controls.items():
        features = prereg.feature_panel(
            flow,
            control_premium,
            flow_tail_quantile=float(selected["flow_tail_quantile"]),
            premium_tail_quantile=float(selected["premium_tail_quantile"]),
            consensus_count=int(selected["consensus_count"]),
            cfg=cfg,
        )
        schedule = prereg.schedule_events(features, cfg)
        for raw in schedule.to_dict(orient="records"):
            records.append({"control": control, **raw})
    frame = pd.DataFrame(records, columns=pd.Index(CONTROL_CLOCK_COLUMNS))
    for column in (
        "source_hour_open_utc",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return frame.sort_values(["control", "entry_time"], kind="mergesort").reset_index(
        drop=True
    )


def _write_control_clock(frame: pd.DataFrame) -> str:
    CONTROL_CLOCKS.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=CONTROL_CLOCKS.parent,
        prefix=f".{CONTROL_CLOCKS.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        deterministic_gzip_csv(frame, temporary)
        rebuilt = temporary.read_bytes()
        try:
            os.link(temporary, CONTROL_CLOCKS)
        except FileExistsError:
            if CONTROL_CLOCKS.read_bytes() != rebuilt:
                raise RuntimeError("refusing to overwrite frozen DTAC control clock")
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(CONTROL_CLOCKS)


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
    *, expected_clock_sha256: str | None = None
) -> dict[str, pd.DataFrame]:
    support = _verify_static_inputs()
    if not CONTROL_CLOCKS.exists():
        raise FileNotFoundError("DTAC-8 control clock is not frozen")
    if (
        expected_clock_sha256 is not None
        and _sha256(CONTROL_CLOCKS) != expected_clock_sha256
    ):
        raise ValueError("DTAC-8 frozen control clock hash changed")
    frame = pd.read_csv(CONTROL_CLOCKS)
    if tuple(frame.columns) != CONTROL_CLOCK_COLUMNS:
        raise ValueError("DTAC-8 control clock schema changed")
    for column in (
        "source_hour_open_utc",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if set(frame["control"]) != set(ALL_CONTROLS):
        raise ValueError("DTAC-8 control family changed")
    schedules: dict[str, pd.DataFrame] = {}
    for control in ALL_CONTROLS:
        schedule = (
            cast(pd.DataFrame, frame.loc[frame["control"].eq(control)].copy())
            .sort_values("entry_time", kind="mergesort")
            .reset_index(drop=True)
        )
        if control not in INDEPENDENT_CLOCK_CONTROLS and len(schedule) != int(
            support["clock_rows"]
        ):
            raise ValueError(f"DTAC-8 {control} count changed")
        if control in INDEPENDENT_CLOCK_CONTROLS and schedule.empty:
            raise ValueError(f"DTAC-8 {control} is empty")
        entries = cast(pd.Series, schedule["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, schedule["exit_time"]).reset_index(drop=True)
        if not bool(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()):
            raise ValueError(f"DTAC-8 {control} schedule overlaps")
        expected_delay = pd.Timedelta(
            minutes=65 if control == "extra_latency_1h" else 5
        )
        if not bool(
            (entries - schedule["decision_time"].reset_index(drop=True))
            .eq(expected_delay)
            .all()
        ):
            raise ValueError(f"DTAC-8 {control} latency changed")
        for split, (start, end) in STAGE_WINDOWS.items():
            window = cast(pd.DataFrame, schedule.loc[schedule["split"].eq(split)])
            if not (
                bool(window["entry_time"].ge(start).all())
                and bool(window["exit_time"].le(end).all())
            ):
                raise ValueError(f"DTAC-8 {control}/{split} containment changed")
        schedules[control] = schedule
    primary = schedules["primary"]
    for control in SAME_CLOCK_CONTROLS:
        if not schedules[control][["decision_time", "entry_time", "exit_time"]].equals(
            primary[["decision_time", "entry_time", "exit_time"]]
        ):
            raise ValueError(f"DTAC-8 {control} changed the primary clock")
    latency = schedules["extra_latency_1h"]
    if not latency["decision_time"].equals(primary["decision_time"]):
        raise ValueError("DTAC-8 latency control changed decisions")
    if not bool(
        latency["entry_time"].eq(primary["entry_time"] + pd.Timedelta(hours=1)).all()
        and latency["exit_time"].eq(primary["exit_time"] + pd.Timedelta(hours=1)).all()
        and latency["side"].eq(primary["side"]).all()
    ):
        raise ValueError("DTAC-8 latency control semantics changed")
    for control in INDEPENDENT_CLOCK_CONTROLS:
        if schedules[control]["decision_time"].equals(primary["decision_time"]):
            raise ValueError(f"DTAC-8 {control} duplicated the primary clock")
    expected = derive_control_clocks()
    semantic_columns = [
        "control",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    actual_semantics = (
        cast(pd.DataFrame, frame[semantic_columns])
        .sort_values(["control", "entry_time"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_semantics = (
        cast(pd.DataFrame, expected[semantic_columns])
        .sort_values(["control", "entry_time"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not actual_semantics.equals(expected_semantics):
        raise ValueError("DTAC-8 frozen control semantics changed")
    return schedules


def _window_schedule(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    start, end = STAGE_WINDOWS[stage]
    selected = cast(pd.DataFrame, frame.loc[frame["split"].eq(stage)].copy())
    if not (
        bool(selected["entry_time"].ge(start).all())
        and bool(selected["exit_time"].le(end).all())
    ):
        raise ValueError(f"DTAC-8 {stage} schedule crosses its window")
    return selected.sort_values("entry_time", kind="mergesort").reset_index(drop=True)


def _stage_exit_boundary_required(
    stage: str, schedules: dict[str, pd.DataFrame]
) -> bool:
    end = STAGE_WINDOWS[stage][1]
    return any(
        bool(_window_schedule(schedule, stage)["exit_time"].eq(end).any())
        for schedule in schedules.values()
    )


def _train_source_contract(schedules: dict[str, pd.DataFrame]) -> dict[str, Any]:
    market_manifest = _load_json(TRAIN_MARKET_MANIFEST)
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("DTAC-8 train market manifest provenance changed")
    if market_manifest.get("combined_sha256") != TRAIN_MARKET_SHA256:
        raise ValueError("DTAC-8 train market manifest output changed")
    funding_manifest = _load_json(TRAIN_FUNDING_MANIFEST)
    funding_core = {
        key: value
        for key, value in funding_manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if funding_manifest.get("manifest_hash") != _canonical_hash(funding_core):
        raise ValueError("DTAC-8 train funding manifest hash changed")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("DTAC-8 train funding manifest opened outcomes")
    if funding_manifest.get("data", {}).get("sha256") != TRAIN_FUNDING_SHA256:
        raise ValueError("DTAC-8 train funding manifest output changed")
    start, end = STAGE_WINDOWS["train"]
    return {
        "stage": "train",
        "physical_window": [start.isoformat(), end.isoformat()],
        "exit_boundary_required": _stage_exit_boundary_required("train", schedules),
        "market": {
            "path": str(TRAIN_MARKET),
            "sha256": TRAIN_MARKET_SHA256,
            "manifest": str(TRAIN_MARKET_MANIFEST),
            "manifest_sha256": TRAIN_MARKET_MANIFEST_SHA256,
        },
        "funding": {
            "path": str(TRAIN_FUNDING),
            "sha256": TRAIN_FUNDING_SHA256,
            "manifest": str(TRAIN_FUNDING_MANIFEST),
            "manifest_sha256": TRAIN_FUNDING_MANIFEST_SHA256,
        },
    }


def _future_source_spec(
    stage: str, schedules: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    start, end = STAGE_WINDOWS[stage]
    return {
        "stage": stage,
        "required_manifest": str(FUTURE_SOURCE_MANIFESTS[stage]),
        "required_protocol_version": "discordant_tail_absorption_consensus_execution_source_v1",
        "physical_window": [start.isoformat(), end.isoformat()],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": _stage_exit_boundary_required(stage, schedules),
        "strategy_outcomes_calculated": False,
    }


def _load_future_source_contract(
    stage: str, schedules: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    spec = _future_source_spec(stage, schedules)
    payload = _load_json(FUTURE_SOURCE_MANIFESTS[stage])
    _verify_manifest(payload, label=f"{stage} execution source")
    if payload.get("protocol_version") != spec["required_protocol_version"]:
        raise ValueError(f"DTAC-8 {stage} source protocol changed")
    if payload.get("candidate") != POLICY_ID or payload.get("stage") != stage:
        raise ValueError(f"DTAC-8 {stage} source identity changed")
    for key in (
        "physical_window",
        "physical_rows_limited_to_window",
        "exit_boundary_required",
        "strategy_outcomes_calculated",
    ):
        if payload.get(key) != spec[key]:
            raise ValueError(f"DTAC-8 {stage} source {key} changed")
    if payload.get("official_checksums_verified") is not True:
        raise ValueError(f"DTAC-8 {stage} source lacks official checksums")
    for name in ("market", "funding"):
        item = payload.get(name)
        if not isinstance(item, dict) or set(item) < {"path", "sha256"}:
            raise ValueError(f"DTAC-8 {stage} source lacks {name} identity")
    return payload


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("DTAC-8 evaluator freeze is write-once")
    if any(path.exists() for path in STAGE_OUTPUTS.values()):
        raise RuntimeError("DTAC-8 cannot freeze after an outcome stage exists")
    support = _verify_static_inputs()
    control_frame = derive_control_clocks()
    control_clock_sha = _write_control_clock(control_frame)
    schedules = load_schedules(expected_clock_sha256=control_clock_sha)
    records = {
        name: {
            "events": len(schedule),
            "schedule_hash": _schedule_hash(schedule),
            "stage_counts": {
                stage: len(_window_schedule(schedule, stage)) for stage in STAGE_ORDER
            },
            "first_entry": _utc(schedule["entry_time"].min()).isoformat(),
            "last_exit": _utc(schedule["exit_time"].max()).isoformat(),
        }
        for name, schedule in schedules.items()
    }
    core = {
        "protocol_version": "discordant_tail_absorption_consensus_evaluator_v1",
        "candidate": POLICY_ID,
        "as_of_date": "2026-07-19",
        "support_commit": SUPPORT_COMMIT,
        "support_manifest_hash": support["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_source_dependency": str(strict_source.EVALUATOR_SOURCE),
        "strict_source_dependency_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_stablecoin_quote_flow_diffusion.py"
        ],
        "evaluation_config": asdict(EvaluationConfig()),
        "static_inputs": STATIC_INPUT_SHA256,
        "control_clock": str(CONTROL_CLOCKS),
        "control_clock_sha256": control_clock_sha,
        "schedule_records": records,
        "train_execution_source": _train_source_contract(schedules),
        "sealed_future_source_specs": {
            stage: _future_source_spec(stage, schedules) for stage in STAGE_ORDER[1:]
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
        "outcome_gate": support["protocol"]["outcome_gate"],
        "mechanism_controls": list(MECHANISM_CONTROLS),
        "strict_accounting": {
            "funding_boundary": (
                "interior symmetric; exact entry/exit credits dropped and debits "
                "retained; every settlement mark visited; event at entry+offset is "
                "interior and event after exit is outside"
            ),
            "mdd": (
                "global/pre-entry HWM; entry cost; funding settlement marks; each "
                "held 5m favorable then adverse OHLC; virtual adverse-mark exit "
                "cost; actual exit cost"
            ),
            "cagr": "full declared split calendar including warm-up and idle cash",
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
        raise ValueError("DTAC-8 evaluator source changed after freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("DTAC-8 evaluator froze another support commit")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("DTAC-8 evaluator config changed after freeze")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("DTAC-8 evaluator freeze is not sealed")
    if payload.get("sealed_windows") != list(STAGE_ORDER):
        raise ValueError("DTAC-8 evaluator freeze stage seal changed")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("DTAC-8 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("DTAC-8 evaluator freeze parsed funding")
    if payload.get("execution_outcome_data_bytes_hashed_during_freeze") is not False:
        raise ValueError("DTAC-8 evaluator freeze hashed execution outcomes")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("DTAC-8 evaluator freeze simulated outcomes")
    support = _verify_static_inputs()
    if payload.get("outcome_gate") != support["protocol"]["outcome_gate"]:
        raise ValueError("DTAC-8 evaluator outcome gate changed")
    schedules = load_schedules(
        expected_clock_sha256=cast(str, payload["control_clock_sha256"])
    )
    for name, schedule in schedules.items():
        if payload["schedule_records"][name]["schedule_hash"] != _schedule_hash(
            schedule
        ):
            raise ValueError(f"DTAC-8 {name} schedule changed after freeze")
    return payload


def _verified_prior_reports(stage: str, *, freeze_hash: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for prior in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        payload = _load_json(STAGE_OUTPUTS[prior])
        _verify_manifest(payload, label=f"stored {prior}")
        if payload.get("stage") != prior or payload.get("stage_passed") is not True:
            raise ValueError(f"DTAC-8 {prior} did not pass; {stage} remains sealed")
        index = STAGE_ORDER.index(prior)
        if payload.get("opened_windows") != list(STAGE_ORDER[: index + 1]):
            raise ValueError(f"DTAC-8 {prior} opened an unexpected window")
        if payload.get("sealed_windows") != list(STAGE_ORDER[index + 1 :]):
            raise ValueError(f"DTAC-8 {prior} stage seal changed")
        if payload.get("evaluator_freeze_manifest_hash") != freeze_hash:
            raise ValueError(f"DTAC-8 {prior} froze another evaluator")
        if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
            raise ValueError(f"DTAC-8 {prior} evaluator source changed")
        reports.append(payload)
    return reports


def load_execution_window(
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"DTAC-8 unknown stage: {stage}")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    schedules = load_schedules(
        expected_clock_sha256=cast(str, freeze["control_clock_sha256"])
    )
    if stage == "train":
        contract = _train_source_contract(schedules)
    else:
        contract = _load_future_source_contract(stage, schedules)
    start, end = STAGE_WINDOWS[stage]
    exact_physical = stage != "train"
    include_boundary = bool(contract["exit_boundary_required"])
    market_item = contract["market"]
    funding_item = contract["funding"]
    market, market_diagnostics = strict_source._parse_market_window(
        market_item["path"],
        start,
        end,
        require_exact_physical_window=exact_physical,
        include_end_boundary=include_boundary,
    )
    funding, funding_diagnostics = strict_source._parse_funding_window(
        funding_item["path"],
        start,
        end,
        require_exact_physical_window=exact_physical,
        include_end_boundary=include_boundary,
    )
    if _sha256(market_item["path"]) != market_item["sha256"]:
        raise ValueError(f"DTAC-8 {stage} market bytes changed")
    if _sha256(funding_item["path"]) != funding_item["sha256"]:
        raise ValueError(f"DTAC-8 {stage} funding bytes changed")
    return (
        market,
        funding,
        {
            "stage": stage,
            "physical_window": [start.isoformat(), end.isoformat()],
            "market": market_diagnostics,
            "funding": funding_diagnostics,
            "market_sha256": market_item["sha256"],
            "funding_sha256": funding_item["sha256"],
        },
    )


def _ratio(cagr: float, strict_mdd: float) -> float:
    if strict_mdd > 0.0:
        return cagr / strict_mdd
    if cagr > 0.0:
        return float("inf")
    if cagr < 0.0:
        return float("-inf")
    return 0.0


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate_per_side: float,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    if cfg != EvaluationConfig():
        raise ValueError("DTAC-8 evaluation configuration is frozen")
    if not 0.0 <= cost_rate_per_side < 0.1 or end <= start:
        raise ValueError("DTAC-8 simulation window or cost is invalid")
    positions = {
        _utc(value): index
        for index, value in enumerate(cast(pd.Series, market["date"]))
    }
    funding_times = cast(pd.Series, funding["funding_time"])
    realized_equity = 1.0
    high_water_mark = 1.0
    maximum_drawdown = 0.0
    records: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None

    def update_path(value: float) -> None:
        nonlocal high_water_mark, maximum_drawdown
        if not np.isfinite(value):
            raise ValueError("DTAC-8 strict equity path is non-finite")
        high_water_mark = max(high_water_mark, value)
        maximum_drawdown = max(
            maximum_drawdown, 1.0 - value / max(high_water_mark, 1e-15)
        )

    for clock in clocks.to_dict(orient="records"):
        entry_time = _utc(clock["entry_time"])
        exit_time = _utc(clock["exit_time"])
        if entry_time < start or exit_time > end:
            raise ValueError("DTAC-8 clock crosses the simulation window")
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("DTAC-8 simulation schedule overlaps")
        previous_exit = exit_time
        entry_position = positions.get(entry_time)
        exit_position = positions.get(exit_time)
        if entry_position is None or exit_position is None:
            raise ValueError("DTAC-8 clock is absent from the market grid")
        if exit_position - entry_position != cfg.hold_bars:
            raise ValueError("DTAC-8 hold is not exactly 96 bars / 8 hours")
        side = int(clock["side"])
        if side not in (-1, 1):
            raise ValueError("DTAC-8 side must be -1 or 1")

        entry_price = float(market.iloc[entry_position]["open"])
        exit_price = float(market.iloc[exit_position]["open"])
        pre_entry_equity = realized_equity
        quantity = pre_entry_equity * cfg.leverage / entry_price
        entry_fee = quantity * entry_price * cost_rate_per_side
        cash = pre_entry_equity - entry_fee
        update_path(cash)
        included_funding = cast(
            pd.DataFrame,
            funding.loc[
                funding_times.ge(entry_time) & funding_times.le(exit_time)
            ].copy(),
        )
        next_funding = 0
        funding_cash = 0.0
        applied_funding_events = 0
        dropped_boundary_credits = 0
        visited_funding_events = 0

        def apply_funding_through(upper: pd.Timestamp) -> None:
            nonlocal cash, funding_cash, next_funding
            nonlocal applied_funding_events, dropped_boundary_credits
            nonlocal visited_funding_events
            while next_funding < len(included_funding):
                event = included_funding.iloc[next_funding]
                event_time = _utc(event["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(event["settlement_mark_price"])
                visited_funding_events += 1
                cash_flow = (
                    -side * quantity * settlement_mark * float(event["funding_rate"])
                )
                boundary = event_time in (entry_time, exit_time)
                if boundary and cash_flow > 0.0:
                    dropped_boundary_credits += 1
                else:
                    cash += cash_flow
                    funding_cash += cash_flow
                    applied_funding_events += 1
                marked = cash + side * quantity * (settlement_mark - entry_price)
                virtual_exit_fee = quantity * settlement_mark * cost_rate_per_side
                update_path(marked - virtual_exit_fee)
                next_funding += 1

        for position in range(entry_position, exit_position):
            bar = market.iloc[position]
            bar_time = _utc(bar["date"])
            bar_end = _utc(bar_time + BAR - pd.Timedelta(1, unit="ns"))
            apply_funding_through(bar_end)
            favorable = float(bar["high"] if side > 0 else bar["low"])
            update_path(cash + side * quantity * (favorable - entry_price))
            adverse = float(bar["low"] if side > 0 else bar["high"])
            adverse_equity = cash + side * quantity * (adverse - entry_price)
            update_path(adverse_equity - quantity * adverse * cost_rate_per_side)
        apply_funding_through(exit_time)
        if next_funding != len(included_funding):
            raise ValueError("DTAC-8 funding event was not visited before exit")
        gross_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * cost_rate_per_side
        realized_equity = cash + gross_pnl - exit_fee
        update_path(realized_equity)
        records.append(
            {
                "control": str(clock.get("control", "primary")),
                "split": str(clock.get("split", "synthetic")),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars_held": cfg.hold_bars,
                "pre_entry_equity": pre_entry_equity,
                "quantity_btc": quantity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "funding_cash": funding_cash,
                "funding_events": applied_funding_events,
                "visited_funding_events": visited_funding_events,
                "dropped_boundary_funding_credits": dropped_boundary_credits,
                "gross_underlying_bp": side
                * (exit_price / entry_price - 1.0)
                * 10_000.0,
                "gross_pnl": gross_pnl,
                "net_return": realized_equity / pre_entry_equity - 1.0,
                "post_exit_equity": realized_equity,
            }
        )

    years = (end - start).total_seconds() / YEAR_SECONDS
    absolute_return = realized_equity - 1.0
    cagr = realized_equity ** (1.0 / years) - 1.0 if realized_equity > 0 else -1.0
    trades = pd.DataFrame(records)
    significance = strict_source.weekly_cluster_signflip_two_sided(trades)
    mean_gross = (
        float(cast(pd.Series, trades["gross_underlying_bp"]).mean())
        if len(trades)
        else 0.0
    )
    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "calendar_years": years,
        "absolute_return_pct": absolute_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "strict_mdd_pct": maximum_drawdown * 100.0,
        "cagr_to_strict_mdd": _ratio(cagr, maximum_drawdown),
        "trades": len(records),
        "longs": sum(int(row["side"]) == 1 for row in records),
        "shorts": sum(int(row["side"]) == -1 for row in records),
        "ending_equity": realized_equity,
        "mean_gross_underlying_bp": mean_gross,
        "weekly_cluster_signflip": significance,
        "trade_details": records,
    }


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
        schedule.loc[
            schedule["entry_time"].ge(start) & schedule["exit_time"].le(end)
        ].copy(),
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
    support: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float], float]:
    gate = support["protocol"]["outcome_gate"]
    primary_ratio = float(base["cagr_to_strict_mdd"])
    control_ratios = {
        name: float(controls[name]["cagr_to_strict_mdd"]) for name in MECHANISM_CONTROLS
    }
    if any(np.isnan(value) or np.isposinf(value) for value in control_ratios.values()):
        minimum_margin = float("-inf")
    else:
        minimum_margin = min(primary_ratio - value for value in control_ratios.values())
    checks = {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": primary_ratio
        >= float(gate["cagr_to_strict_mdd_min"]),
        "strict_mdd_at_most_15pct": base["strict_mdd_pct"]
        <= float(gate["strict_mdd_max_pct"]),
        "minimum_trades": base["trades"] >= int(gate["minimum_trades"][stage]),
        "weekly_cluster_signflip_p_at_most_10pct": base["weekly_cluster_signflip"][
            "p_value_two_sided"
        ]
        <= float(gate["weekly_cluster_signflip_p_max"]),
        "mean_gross_underlying_at_least_20bp": base["mean_gross_underlying_bp"]
        >= float(gate["mean_gross_underlying_move_bp_min"]),
        "each_contained_half_absolute_return_positive": all(
            item["absolute_return_pct"] > 0.0 for item in halves.values()
        ),
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "stress_cagr_to_strict_mdd_at_least_2_5": stress["cagr_to_strict_mdd"]
        >= float(gate["stress_cagr_to_strict_mdd_min"]),
        "mechanism_control_margin_at_least_0_25": minimum_margin
        >= float(gate["mechanism_control_margin_min"]),
    }
    return checks, control_ratios, minimum_margin


def _build_stage_report(stage: str) -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    support = _verify_static_inputs()
    prior = _verified_prior_reports(
        stage, freeze_hash=cast(str, freeze["manifest_hash"])
    )
    schedules = load_schedules(
        expected_clock_sha256=cast(str, freeze["control_clock_sha256"])
    )
    market, funding, diagnostics = load_execution_window(stage)
    start, end = STAGE_WINDOWS[stage]
    primary_schedule = _window_schedule(schedules["primary"], stage)
    base = simulate_strict(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate_per_side=EvaluationConfig.base_cost_notional_per_side,
    )
    stress = simulate_strict(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate_per_side=EvaluationConfig.stress_cost_notional_per_side,
    )
    halves = {
        name: _simulate_window(
            market,
            funding,
            primary_schedule,
            start=half_start,
            end=half_end,
            cost=EvaluationConfig.base_cost_notional_per_side,
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
            cost_rate_per_side=EvaluationConfig.base_cost_notional_per_side,
        )
        for name, schedule in schedules.items()
        if name != "primary"
    }
    gates, ratios, margin = _stage_gates(stage, base, stress, halves, controls, support)
    passed = all(gates.values())
    index = STAGE_ORDER.index(stage)
    core = {
        "protocol_version": "discordant_tail_absorption_consensus_stage_v1",
        "candidate": POLICY_ID,
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_prior_stage_manifest_hashes": {
            row["stage"]: row["manifest_hash"] for row in prior
        },
        "config": asdict(EvaluationConfig()),
        "execution_diagnostics": diagnostics,
        "primary": {
            "metrics": base,
            "headline": _headline(base),
            "stress_metrics": stress,
            "stress_headline": _headline(stress),
            "contained_half_metrics": halves,
            "contained_half_headlines": {
                name: _headline(item) for name, item in halves.items()
            },
        },
        "controls": {
            name: {"metrics": item, "headline": _headline(item)}
            for name, item in controls.items()
        },
        "mechanism_control_ratios": ratios,
        "minimum_mechanism_control_margin": margin,
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
        f"# DTAC-8 {report['stage']} strict result — 2026-07-19",
        "",
        "Absolute return and CAGR use the full declared calendar. Strict MDD uses "
        "the global/pre-entry HWM, costs, exact funding boundaries and every held "
        "five-minute favorable-then-adverse path.",
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
            "",
            "## Contained halves",
            "",
            "| Window | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["primary"]["contained_half_headlines"].items():
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
        raise ValueError(f"DTAC-8 unknown stage: {stage}")
    output = STAGE_OUTPUTS[stage]
    document = STAGE_DOCS[stage]
    if output.exists() or document.exists():
        raise FileExistsError(f"DTAC-8 {stage} result is write-once")
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
    group.add_argument("--stage", choices=STAGE_ORDER)
    args = parser.parse_args()
    if args.freeze:
        report = freeze_evaluator()
        print(
            json.dumps(
                {
                    "manifest_hash": report["manifest_hash"],
                    "evaluator_source_sha256": report["evaluator_source_sha256"],
                    "control_clock_sha256": report["control_clock_sha256"],
                    "opened_windows": report["opened_windows"],
                    "sealed_windows": report["sealed_windows"],
                },
                indent=2,
            )
        )
    else:
        assert args.stage is not None
        report = evaluate_stage(args.stage)
        print(
            json.dumps(
                {
                    "stage": report["stage"],
                    "passed": report["stage_passed"],
                    "headline": report["primary"]["headline"],
                    "failed_gates": report["failed_gates"],
                    "disposition": report["disposition"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
