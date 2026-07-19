"""Sequential hardened evaluator for frozen SQFD-6 clocks.

The freeze path opens no execution OHLC or funding row. Outcome stages are
strictly sequential: train -> test -> eval -> final, with later data loaders
blocked until every prior frozen gate passes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_stablecoin_quote_flow_diffusion_support as support_builder


POLICY_ID = "SQFD-6"
SUPPORT_COMMIT = "107ddbddc233e9316392cc15ba69588f497e06db"
PREREGISTRATION = Path(
    "results/stablecoin_quote_flow_diffusion_preregistration_2026-07-19.json"
)
SUPPORT_RESULT = Path("results/stablecoin_quote_flow_diffusion_support_2026-07-19.json")
CLOCKS = Path("data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz")
EVALUATOR_SOURCE = Path("training/evaluate_stablecoin_quote_flow_diffusion.py")
EVALUATOR_FREEZE = Path(
    "results/stablecoin_quote_flow_diffusion_evaluator_freeze_2026-07-19.json"
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
    str(PREREGISTRATION): (
        "3fed620146b98e920175445a12e2a8684c2a3431e42b1a784ea0e3076577aee3"
    ),
    "docs/stablecoin-quote-flow-diffusion-preregistration-2026-07-19.md": (
        "ce1a284cabfd800cc8093cc166ae7fe1decec3423b7721433942500b84ca7781"
    ),
    "training/build_stablecoin_quote_flow_diffusion_support.py": (
        "15f2b6cc34ddd6331be61aeeabed3c878ba8cb8d1091f42ca1ebf006ad242d17"
    ),
    "docs/stablecoin-quote-flow-diffusion-support-freeze-2026-07-19.md": (
        "e029e01fa663ec6b30e2fb16aa491d78e509572fcbfa5f56a0c4f2558ec3111b"
    ),
    str(SUPPORT_RESULT): (
        "07230e9e579f1b16e07712a022e572026b4fbfa17070e998970b3fd8ee21d4b5"
    ),
    str(CLOCKS): ("a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b"),
    str(TRAIN_MARKET_MANIFEST): TRAIN_MARKET_MANIFEST_SHA256,
    str(TRAIN_FUNDING_MANIFEST): TRAIN_FUNDING_MANIFEST_SHA256,
}

STAGE_ORDER = ("train", "test", "eval", "final")
STAGE_OUTPUTS = {
    "train": Path(
        "results/stablecoin_quote_flow_diffusion_train_2023h2_2026-07-19.json"
    ),
    "test": Path("results/stablecoin_quote_flow_diffusion_test_2024_2026-07-19.json"),
    "eval": Path("results/stablecoin_quote_flow_diffusion_eval_2025_2026-07-19.json"),
    "final": Path(
        "results/stablecoin_quote_flow_diffusion_final_2026h1_2026-07-19.json"
    ),
}
STAGE_DOCS = {
    stage: Path(f"docs/stablecoin-quote-flow-diffusion-{stage}-result-2026-07-19.md")
    for stage in STAGE_ORDER
}
FUTURE_SOURCE_MANIFESTS = {
    stage: Path(
        f"results/stablecoin_quote_flow_diffusion_{stage}_execution_source_2026-07-19.json"
    )
    for stage in STAGE_ORDER[1:]
}
STAGE_WINDOWS = dict(support_builder.SPLITS)
HALF_WINDOWS = {
    "train": {
        "2023_q3": (
            support_builder._utc_timestamp("2023-07-01T00:00:00Z"),
            support_builder._utc_timestamp("2023-10-01T00:00:00Z"),
        ),
        "2023_q4": (
            support_builder._utc_timestamp("2023-10-01T00:00:00Z"),
            support_builder._utc_timestamp("2024-01-01T00:00:00Z"),
        ),
    },
    "test": {
        "2024_h1": (
            support_builder._utc_timestamp("2024-01-01T00:00:00Z"),
            support_builder._utc_timestamp("2024-07-01T00:00:00Z"),
        ),
        "2024_h2": (
            support_builder._utc_timestamp("2024-07-01T00:00:00Z"),
            support_builder._utc_timestamp("2025-01-01T00:00:00Z"),
        ),
    },
    "eval": {
        "2025_h1": (
            support_builder._utc_timestamp("2025-01-01T00:00:00Z"),
            support_builder._utc_timestamp("2025-07-01T00:00:00Z"),
        ),
        "2025_h2": (
            support_builder._utc_timestamp("2025-07-01T00:00:00Z"),
            support_builder._utc_timestamp("2026-01-01T00:00:00Z"),
        ),
    },
    "final": {
        "2026_q1": (
            support_builder._utc_timestamp("2026-01-01T00:00:00Z"),
            support_builder._utc_timestamp("2026-04-01T00:00:00Z"),
        ),
        "2026_q2": (
            support_builder._utc_timestamp("2026-04-01T00:00:00Z"),
            support_builder._utc_timestamp("2026-07-01T00:00:00Z"),
        ),
    },
}

ABLATION_CONTROLS = (
    "no_alt_breadth",
    "no_usdt_lag",
    "no_participation",
    "usdt_only",
)
FALSIFICATION_CONTROLS = (
    "direction_flip",
    "extra_latency_1h",
    "deterministic_random_side",
)
MECHANISM_CONTROLS = (*ABLATION_CONTROLS, *FALSIFICATION_CONTROLS)
ALL_CONTROLS = ("primary", *MECHANISM_CONTROLS)
BAR = pd.Timedelta(minutes=5)
YEAR_SECONDS = 365.25 * 86_400.0


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 72
    exact_cluster_max: int = 20
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_719


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("SQFD-6 timestamp is NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"SQFD-6 expected JSON object: {path}")
    return payload


def _verify_manifest(
    payload: dict[str, Any],
    *,
    label: str,
    ignore_created_at: bool = False,
) -> None:
    excluded = {"manifest_hash"}
    if ignore_created_at:
        excluded.add("created_at")
    core = {key: value for key, value in payload.items() if key not in excluded}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError(f"SQFD-6 {label} manifest hash changed")


def _verify_evaluation_contract(prereg: dict[str, Any]) -> EvaluationConfig:
    policy = prereg["policy"]
    statistical = prereg["outcome_gate"]["statistical_test_contract"]
    cfg = EvaluationConfig()
    expected = {
        "leverage": policy["leverage"],
        "base_cost_notional_per_side": policy["base_cost_notional_per_side"],
        "stress_cost_notional_per_side": policy["stress_cost_notional_per_side"],
        "hold_bars": int(policy["hold_hours"] * 12),
        "exact_cluster_max": 20,
        "cluster_draws": statistical["draws"],
        "cluster_seed": statistical["seed"],
    }
    if asdict(cfg) != expected:
        raise ValueError("SQFD-6 evaluator configuration drifted")
    if statistical["cluster_key"] != "UTC entry timestamp ISO year/week":
        raise ValueError("SQFD-6 weekly cluster key drifted")
    return cfg


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"SQFD-6 frozen input changed: {path}")
    prereg = _load_json(PREREGISTRATION)
    _verify_manifest(prereg, label="preregistration")
    if prereg.get("outcomes_opened") is not False:
        raise ValueError("SQFD-6 preregistration opened outcomes")
    support = _load_json(SUPPORT_RESULT)
    _verify_manifest(support, label="support")
    if support.get("candidate") != POLICY_ID:
        raise ValueError("SQFD-6 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("SQFD-6 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("SQFD-6 support opened an outcome source")
    if support.get("support_passed") is not True:
        raise ValueError("SQFD-6 support did not pass")
    if support.get("clock_sha256") != STATIC_INPUT_SHA256[str(CLOCKS)]:
        raise ValueError("SQFD-6 support no longer binds clocks")
    _verify_evaluation_contract(prereg)
    return prereg, support


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "control": str(row["control"]),
            "split": str(row["split"]),
            "source_hour_start": _timestamp(row["source_hour_start"]).isoformat(),
            "decision_time": _timestamp(row["decision_time"]).isoformat(),
            "feature_available_time": _timestamp(
                row["feature_available_time"]
            ).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def _deterministic_random_side(decision_time: Any) -> int:
    timestamp = _timestamp(decision_time).strftime("%Y-%m-%dT%H:%M:%SZ")
    first_nibble = int(
        hashlib.sha256(f"{POLICY_ID}|{timestamp}".encode("ascii")).hexdigest()[0],
        16,
    )
    return 1 if first_nibble % 2 == 0 else -1


def load_schedules() -> dict[str, pd.DataFrame]:
    _, support = _verify_static_inputs()
    frame = pd.read_csv(CLOCKS)
    if tuple(frame.columns) != support_builder.CLOCK_COLUMNS:
        raise ValueError("SQFD-6 clock schema changed")
    for column in (
        "source_hour_start",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if set(frame["candidate"]) != {POLICY_ID}:
        raise ValueError("SQFD-6 clock candidate changed")
    if set(frame["control"]) != set(ALL_CONTROLS):
        raise ValueError("SQFD-6 control family changed")
    if not bool(frame["side"].isin((-1, 1)).all()):
        raise ValueError("SQFD-6 clock side changed")
    if not bool(
        frame["decision_time"]
        .eq(frame["source_hour_start"] + pd.Timedelta(hours=1))
        .all()
    ):
        raise ValueError("SQFD-6 decision timing changed")
    if not bool(frame["feature_available_time"].eq(frame["decision_time"]).all()):
        raise ValueError("SQFD-6 feature availability changed")
    if not bool(
        frame["exit_time"].eq(frame["entry_time"] + pd.Timedelta(hours=6)).all()
    ):
        raise ValueError("SQFD-6 hold changed")

    schedules: dict[str, pd.DataFrame] = {}
    for control in ALL_CONTROLS:
        schedule = (
            cast(
                pd.DataFrame,
                frame.loc[frame["control"].eq(control)].copy(),
            )
            .sort_values("entry_time", kind="mergesort")
            .reset_index(drop=True)
        )
        expected_count = sum(
            int(support["control_support"][control][split]["events"])
            for split in STAGE_ORDER
        )
        if len(schedule) != expected_count:
            raise ValueError(f"SQFD-6 {control} count changed")
        entries = cast(pd.Series, schedule["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, schedule["exit_time"]).reset_index(drop=True)
        if len(schedule) > 1 and not bool(
            entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()
        ):
            raise ValueError(f"SQFD-6 {control} schedule overlaps")
        delay = entries - cast(pd.Series, schedule["decision_time"]).reset_index(
            drop=True
        )
        expected_delay = pd.Timedelta(
            minutes=65 if control == "extra_latency_1h" else 5
        )
        if not bool(delay.eq(expected_delay).all()):
            raise ValueError(f"SQFD-6 {control} latency changed")
        for split, (start, end) in STAGE_WINDOWS.items():
            window = cast(pd.DataFrame, schedule.loc[schedule["split"].eq(split)])
            contained = (
                bool(window["source_hour_start"].ge(start).all())
                and bool(window["entry_time"].ge(start).all())
                and bool(window["exit_time"].le(end).all())
            )
            if not contained:
                raise ValueError(f"SQFD-6 {control}/{split} containment changed")
        schedules[control] = schedule

    primary = schedules["primary"]
    for name in ("direction_flip", "deterministic_random_side"):
        control = schedules[name]
        if not control[["decision_time", "entry_time", "exit_time"]].equals(
            primary[["decision_time", "entry_time", "exit_time"]]
        ):
            raise ValueError(f"SQFD-6 {name} does not reuse the primary clock")
    if not bool(
        schedules["direction_flip"]["side"]
        .reset_index(drop=True)
        .eq(-primary["side"].reset_index(drop=True))
        .all()
    ):
        raise ValueError("SQFD-6 direction flip changed")
    random_control = schedules["deterministic_random_side"]
    expected_random_sides = random_control["decision_time"].map(
        _deterministic_random_side
    )
    if not bool(random_control["side"].eq(expected_random_sides).all()):
        raise ValueError("SQFD-6 deterministic random-side contract changed")
    delayed = schedules["extra_latency_1h"]
    if not delayed[["decision_time", "source_hour_start"]].equals(
        primary[["decision_time", "source_hour_start"]]
    ):
        raise ValueError("SQFD-6 latency control changed source clocks")
    if not bool(
        delayed["entry_time"]
        .reset_index(drop=True)
        .eq(primary["entry_time"].reset_index(drop=True) + pd.Timedelta(hours=1))
        .all()
    ):
        raise ValueError("SQFD-6 latency control changed shift")
    return schedules


def _window_schedule(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage not in STAGE_WINDOWS:
        raise ValueError(f"SQFD-6 unknown stage: {stage}")
    start, end = STAGE_WINDOWS[stage]
    selected = cast(pd.DataFrame, frame.loc[frame["split"].eq(stage)].copy())
    contained = (
        bool(selected["source_hour_start"].ge(start).all())
        and bool(selected["entry_time"].ge(start).all())
        and bool(selected["exit_time"].le(end).all())
    )
    if not contained:
        raise ValueError(f"SQFD-6 {stage} schedule crosses its window")
    return selected.sort_values("entry_time", kind="mergesort").reset_index(drop=True)


def _stage_exit_boundary_required(stage: str) -> bool:
    if stage not in STAGE_WINDOWS:
        raise ValueError(f"SQFD-6 unknown stage: {stage}")
    end = STAGE_WINDOWS[stage][1]
    return any(
        bool(_window_schedule(schedule, stage)["exit_time"].eq(end).any())
        for schedule in load_schedules().values()
    )


def _train_source_contract() -> dict[str, Any]:
    market_manifest = _load_json(TRAIN_MARKET_MANIFEST)
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("SQFD-6 train market manifest provenance changed")
    if market_manifest.get("combined_sha256") != TRAIN_MARKET_SHA256:
        raise ValueError("SQFD-6 train market manifest changed output")
    funding_manifest = _load_json(TRAIN_FUNDING_MANIFEST)
    _verify_manifest(
        funding_manifest,
        label="train funding source",
        ignore_created_at=True,
    )
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("SQFD-6 train funding manifest opened outcomes")
    if funding_manifest.get("data", {}).get("sha256") != TRAIN_FUNDING_SHA256:
        raise ValueError("SQFD-6 train funding manifest changed output")
    start, end = STAGE_WINDOWS["train"]
    return {
        "stage": "train",
        "physical_window": [start.isoformat(), end.isoformat()],
        "exit_boundary_required": _stage_exit_boundary_required("train"),
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
            "mark_contract": (
                "exact fundingTime/rate; official containing 8h mark-price kline open "
                "as the frozen settlement-mark proxy"
            ),
        },
    }


def _future_source_spec(stage: str) -> dict[str, Any]:
    start, end = STAGE_WINDOWS[stage]
    boundary_required = _stage_exit_boundary_required(stage)
    return {
        "stage": stage,
        "required_manifest": str(FUTURE_SOURCE_MANIFESTS[stage]),
        "required_protocol_version": (
            "stablecoin_quote_flow_diffusion_execution_source_v1"
        ),
        "physical_window": [start.isoformat(), end.isoformat()],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": boundary_required,
        "market": (
            "official checksum-verified Binance BTCUSDT USD-M 5m OHLC exact grid; "
            "[start,end) unless a frozen position exits exactly at end, in which case "
            "the end-boundary open is included; no fill or other rows outside the stage"
        ),
        "funding": (
            "exact Binance fundingTime/rate plus official settlement-mark field when "
            "present, otherwise the same containing 8h mark-price-open proxy contract; "
            "exact half-open event grid and no rows outside the stage"
        ),
        "strategy_outcomes_calculated": False,
    }


def _load_future_source_contract(stage: str) -> dict[str, Any]:
    spec = _future_source_spec(stage)
    payload = _load_json(FUTURE_SOURCE_MANIFESTS[stage])
    _verify_manifest(payload, label=f"{stage} execution source")
    if payload.get("protocol_version") != spec["required_protocol_version"]:
        raise ValueError(f"SQFD-6 {stage} source protocol changed")
    if payload.get("candidate") != POLICY_ID or payload.get("stage") != stage:
        raise ValueError(f"SQFD-6 {stage} source identity changed")
    if payload.get("physical_window") != spec["physical_window"]:
        raise ValueError(f"SQFD-6 {stage} source window changed")
    if payload.get("strategy_outcomes_calculated") is not False:
        raise ValueError(f"SQFD-6 {stage} source calculated outcomes")
    if payload.get("physical_rows_limited_to_window") is not True:
        raise ValueError(f"SQFD-6 {stage} source is not physically stage-limited")
    if payload.get("exit_boundary_required") is not spec["exit_boundary_required"]:
        raise ValueError(f"SQFD-6 {stage} source boundary contract changed")
    if payload.get("official_checksums_verified") is not True:
        raise ValueError(f"SQFD-6 {stage} source lacks official checksums")
    for name in ("market", "funding"):
        item = payload.get(name)
        if not isinstance(item, dict) or set(item) < {"path", "sha256"}:
            raise ValueError(f"SQFD-6 {stage} source lacks {name} identity")
    return payload


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    prereg, support = _verify_static_inputs()
    schedules = load_schedules()
    train_source = _train_source_contract()
    records = {
        name: {
            "events": int(len(schedule)),
            "schedule_hash": _schedule_hash(schedule),
            "stage_counts": {
                stage: int(len(_window_schedule(schedule, stage)))
                for stage in STAGE_ORDER
            },
            "first_entry": _timestamp(schedule["entry_time"].min()).isoformat(),
            "last_exit": _timestamp(schedule["exit_time"].max()).isoformat(),
        }
        for name, schedule in schedules.items()
    }
    core: dict[str, Any] = {
        "protocol_version": "stablecoin_quote_flow_diffusion_evaluator_v1",
        "candidate": POLICY_ID,
        "as_of_date": "2026-07-19",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "support_manifest_hash": support["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "evaluation_config": asdict(_verify_evaluation_contract(prereg)),
        "static_inputs": STATIC_INPUT_SHA256,
        "schedule_records": records,
        "train_execution_source": train_source,
        "sealed_future_source_specs": {
            stage: _future_source_spec(stage) for stage in STAGE_ORDER[1:]
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
        "outcome_gate": prereg["outcome_gate"],
        "ablation_controls": list(ABLATION_CONTROLS),
        "mechanism_controls": list(MECHANISM_CONTROLS),
        "falsification_controls": list(FALSIFICATION_CONTROLS),
        "strict_accounting": {
            "funding_boundary": (
                "interior symmetric; exact entry/exit credits dropped and debits retained; "
                "every settlement mark visited even when credit is dropped"
            ),
            "mdd": (
                "global/pre-entry HWM; entry cost; funding settlement marks; each held "
                "5m favorable then adverse OHLC; virtual adverse-mark exit cost; actual exit cost"
            ),
            "cagr": "full declared split calendar including warm-up and idle cash",
        },
        "opened_windows": [],
        "sealed_windows": list(STAGE_ORDER),
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "execution_data_bytes_hashed_during_freeze": False,
        "simulation_run_during_freeze": False,
        "mutable_parameters": [],
    }
    report = _seal(core)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_manifest(payload, label="evaluator freeze")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("SQFD-6 evaluator source changed after freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("SQFD-6 evaluator froze another support commit")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("SQFD-6 evaluator freeze is not sealed")
    if payload.get("sealed_windows") != list(STAGE_ORDER):
        raise ValueError("SQFD-6 evaluator freeze stage seal changed")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("SQFD-6 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("SQFD-6 evaluator freeze parsed funding")
    if payload.get("execution_data_bytes_hashed_during_freeze") is not False:
        raise ValueError("SQFD-6 evaluator freeze hashed execution data")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("SQFD-6 evaluator freeze simulated outcomes")
    prereg, _ = _verify_static_inputs()
    if payload.get("evaluation_config") != asdict(_verify_evaluation_contract(prereg)):
        raise ValueError("SQFD-6 evaluator config changed after freeze")
    schedules = load_schedules()
    for name, schedule in schedules.items():
        if payload["schedule_records"][name]["schedule_hash"] != _schedule_hash(
            schedule
        ):
            raise ValueError(f"SQFD-6 {name} schedule changed after freeze")
    return payload


def _parse_market_window(
    path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    require_exact_physical_window: bool = False,
    include_end_boundary: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wanted = ("date", "open", "high", "low", "close")
    rows: list[tuple[Any, ...]] = []
    line_hash = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["date"] != 0:
            raise ValueError("SQFD-6 market timestamp is not first")
        for line in handle:
            timestamp_text = line.split(",", 1)[0]
            timestamp = _timestamp(timestamp_text)
            if timestamp < start:
                if require_exact_physical_window:
                    raise ValueError("SQFD-6 market source opened a prior-stage row")
                continue
            if timestamp > end or (timestamp == end and not include_end_boundary):
                raise ValueError("SQFD-6 market source opened an unneeded end boundary")
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    timestamp,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
            line_hash.update(line.encode("utf-8"))
    frame = pd.DataFrame(rows, columns=pd.Index(wanted))
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    actual = pd.DatetimeIndex(frame["date"])
    expected = pd.date_range(
        start,
        end,
        freq="5min",
        inclusive="both" if include_end_boundary else "left",
    )
    if not actual.equals(expected):
        raise ValueError("SQFD-6 market window is not the exact 5m grid")
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("SQFD-6 market contains invalid prices")
    opening, high, low, close = values.T
    if (
        (high < np.maximum(opening, close)).any()
        or (low > np.minimum(opening, close)).any()
        or (high < low).any()
    ):
        raise ValueError("SQFD-6 market violates OHLC invariants")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": _timestamp(frame["date"].min()).isoformat(),
        "last_timestamp": _timestamp(frame["date"].max()).isoformat(),
        "window_line_sha256": line_hash.hexdigest(),
    }


def _parse_funding_window(
    path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    require_exact_physical_window: bool = False,
    include_end_boundary: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wanted = ("funding_time_utc", "symbol", "funding_rate", "settlement_mark_price")
    rows: list[tuple[Any, ...]] = []
    line_hash = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        for line in handle:
            fields = line.rstrip("\r\n").split(",")
            timestamp = _timestamp(fields[positions["funding_time_utc"]])
            if timestamp < start:
                if require_exact_physical_window:
                    raise ValueError("SQFD-6 funding source opened a prior-stage row")
                continue
            if timestamp > end or (timestamp == end and not include_end_boundary):
                raise ValueError(
                    "SQFD-6 funding source opened an unneeded end boundary"
                )
            rows.append(
                (
                    timestamp,
                    fields[positions["symbol"]],
                    float(fields[positions["funding_rate"]]),
                    float(fields[positions["settlement_mark_price"]]),
                )
            )
            line_hash.update(line.encode("utf-8"))
    frame = pd.DataFrame(
        rows,
        columns=pd.Index(
            ["funding_time", "symbol", "funding_rate", "settlement_mark_price"]
        ),
    )
    frame["funding_time"] = pd.to_datetime(
        frame["funding_time"], utc=True, errors="raise"
    )
    actual = pd.DatetimeIndex(frame["funding_time"])
    if not actual.is_unique or not actual.is_monotonic_increasing:
        raise ValueError("SQFD-6 funding timestamps are invalid")
    interior = actual[actual < end]
    expected = pd.date_range(start, end, freq="8h", inclusive="left")
    step_ns = int(pd.Timedelta(hours=8).value)
    floor_ns = (np.asarray(interior.asi8, dtype=np.int64) // step_ns) * step_ns
    expected_ns = np.asarray(expected.asi8, dtype=np.int64)
    if not np.array_equal(floor_ns, expected_ns):
        raise ValueError("SQFD-6 funding window is not the exact 8h event grid")
    offsets_ms = (np.asarray(interior.asi8, dtype=np.int64) - expected_ns) / 1_000_000.0
    if np.abs(offsets_ms).max(initial=0.0) > 60_000.0:
        raise ValueError("SQFD-6 funding timestamp offset exceeds one minute")
    if len(actual[actual == end]) > 1:
        raise ValueError("SQFD-6 has duplicate exit-boundary funding")
    if not bool(frame["symbol"].eq("BTCUSDT").all()):
        raise ValueError("SQFD-6 funding symbol changed")
    values = frame[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or bool(
        frame["settlement_mark_price"].le(0.0).any()
    ):
        raise ValueError("SQFD-6 funding contains invalid values")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": _timestamp(frame["funding_time"].min()).isoformat(),
        "last_timestamp": _timestamp(frame["funding_time"].max()).isoformat(),
        "window_line_sha256": line_hash.hexdigest(),
        "maximum_absolute_grid_offset_ms": float(np.abs(offsets_ms).max(initial=0.0)),
        "exit_boundary_events": int(len(actual[actual == end])),
    }


def load_execution_window(
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"SQFD-6 unknown execution stage: {stage}")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=str(freeze["manifest_hash"]))
    if stage == "train":
        contract = _train_source_contract()
    elif stage in STAGE_ORDER[1:]:
        contract = _load_future_source_contract(stage)
    start, end = STAGE_WINDOWS[stage]
    market_item = contract["market"]
    funding_item = contract["funding"]
    exact_physical_window = stage != "train"
    include_end_boundary = bool(contract["exit_boundary_required"])
    market, market_diagnostics = _parse_market_window(
        market_item["path"],
        start,
        end,
        require_exact_physical_window=exact_physical_window,
        include_end_boundary=include_end_boundary,
    )
    funding, funding_diagnostics = _parse_funding_window(
        funding_item["path"],
        start,
        end,
        require_exact_physical_window=exact_physical_window,
        include_end_boundary=include_end_boundary,
    )
    if _sha256(market_item["path"]) != market_item["sha256"]:
        raise ValueError(f"SQFD-6 {stage} market bytes changed")
    if _sha256(funding_item["path"]) != funding_item["sha256"]:
        raise ValueError(f"SQFD-6 {stage} funding bytes changed")
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


def weekly_cluster_signflip_two_sided(
    trades: pd.DataFrame,
    *,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    if trades.empty:
        return {
            "p_value_two_sided": 1.0,
            "cluster_count": 0,
            "method": "empty",
            "draws": 0,
            "seed": cfg.cluster_seed,
            "observed_abs_mean_net_return": 0.0,
            "weekly_net_return_sums": {},
        }
    entry = pd.to_datetime(trades["entry_time"], utc=True, errors="raise")
    iso = entry.dt.isocalendar()
    keys = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    weekly = cast(pd.Series, trades["net_return"]).groupby(keys).sum()
    values = weekly.to_numpy(float)
    observed = abs(float(cast(pd.Series, trades["net_return"]).mean()))
    clusters = len(values)
    exceed = 0
    if clusters <= cfg.exact_cluster_max:
        total = 1 << clusters
        bit_positions = np.arange(clusters, dtype=np.uint64)
        for begin in range(0, total, 50_000):
            indices = np.arange(begin, min(total, begin + 50_000), dtype=np.uint64)
            bits = (indices[:, None] >> bit_positions[None, :]) & 1
            signs = 1.0 - 2.0 * bits.astype(float)
            null = np.abs((signs @ values) / float(len(trades)))
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
        p_value = float(exceed / total)
        method = "exact"
        draws = total
    else:
        generator = np.random.default_rng(cfg.cluster_seed)
        remaining = cfg.cluster_draws
        while remaining:
            batch = min(10_000, remaining)
            signs = generator.choice((-1.0, 1.0), size=(batch, clusters))
            null = np.abs((signs @ values) / float(len(trades)))
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
            remaining -= batch
        p_value = float((1 + exceed) / (cfg.cluster_draws + 1))
        method = "monte_carlo"
        draws = cfg.cluster_draws
    return {
        "p_value_two_sided": p_value,
        "cluster_count": int(clusters),
        "method": method,
        "draws": int(draws),
        "seed": int(cfg.cluster_seed),
        "observed_abs_mean_net_return": observed,
        "weekly_net_return_sums": {
            str(label): float(value) for label, value in weekly.items()
        },
    }


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
        raise ValueError("SQFD-6 evaluation configuration is frozen")
    if not 0.0 <= cost_rate_per_side < 0.1 or end <= start:
        raise ValueError("SQFD-6 simulation window or cost is invalid")
    positions = {
        _timestamp(value): index
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
            raise ValueError("SQFD-6 strict equity path is non-finite")
        high_water_mark = max(high_water_mark, value)
        maximum_drawdown = max(
            maximum_drawdown,
            1.0 - value / max(high_water_mark, 1e-15),
        )

    for clock in clocks.to_dict(orient="records"):
        entry_time = _timestamp(clock["entry_time"])
        exit_time = _timestamp(clock["exit_time"])
        if entry_time < start or exit_time > end:
            raise ValueError("SQFD-6 clock crosses the simulation window")
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("SQFD-6 simulation schedule overlaps")
        previous_exit = exit_time
        entry_position = positions.get(entry_time)
        exit_position = positions.get(exit_time)
        if entry_position is None or exit_position is None:
            raise ValueError("SQFD-6 clock is absent from the market grid")
        if exit_position - entry_position != cfg.hold_bars:
            raise ValueError("SQFD-6 hold is not exactly 72 bars / 6 hours")
        side = int(clock["side"])
        if side not in (-1, 1):
            raise ValueError("SQFD-6 side must be -1 or 1")

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
            nonlocal cash
            nonlocal funding_cash
            nonlocal next_funding
            nonlocal applied_funding_events
            nonlocal dropped_boundary_credits
            nonlocal visited_funding_events
            while next_funding < len(included_funding):
                event = included_funding.iloc[next_funding]
                event_time = _timestamp(event["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(event["settlement_mark_price"])
                visited_funding_events += 1
                cash_flow = (
                    -side * quantity * settlement_mark * float(event["funding_rate"])
                )
                is_boundary = event_time in (entry_time, exit_time)
                if is_boundary and cash_flow > 0.0:
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
            bar_time = _timestamp(bar["date"])
            bar_end = _timestamp(bar_time + BAR - pd.Timedelta(1, unit="ns"))
            apply_funding_through(bar_end)
            favorable_price = float(bar["high"] if side > 0 else bar["low"])
            favorable_equity = cash + side * quantity * (favorable_price - entry_price)
            update_path(favorable_equity)
            adverse_price = float(bar["low"] if side > 0 else bar["high"])
            adverse_equity = cash + side * quantity * (adverse_price - entry_price)
            adverse_exit_fee = quantity * adverse_price * cost_rate_per_side
            update_path(adverse_equity - adverse_exit_fee)

        apply_funding_through(exit_time)
        if next_funding != len(included_funding):
            raise ValueError("SQFD-6 funding event was not applied before exit")
        gross_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * cost_rate_per_side
        realized_equity = cash + gross_pnl - exit_fee
        update_path(realized_equity)
        net_return = realized_equity / pre_entry_equity - 1.0
        records.append(
            {
                "control": str(clock["control"]),
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
                "net_return": net_return,
                "post_exit_equity": realized_equity,
            }
        )

    years = (end - start).total_seconds() / YEAR_SECONDS
    if years <= 0.0:
        raise ValueError("SQFD-6 evaluation window has no duration")
    absolute_return = realized_equity - 1.0
    cagr = realized_equity ** (1.0 / years) - 1.0 if realized_equity > 0.0 else -1.0
    trade_frame = pd.DataFrame(records)
    significance = weekly_cluster_signflip_two_sided(trade_frame, cfg=cfg)
    mean_gross = (
        float(cast(pd.Series, trade_frame["gross_underlying_bp"]).mean())
        if len(trade_frame)
        else 0.0
    )
    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "calendar_years": float(years),
        "absolute_return_pct": float(absolute_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "strict_mdd_pct": float(maximum_drawdown * 100.0),
        "cagr_to_strict_mdd": float(_ratio(cagr, maximum_drawdown)),
        "trades": int(len(records)),
        "longs": int(sum(int(row["side"]) == 1 for row in records)),
        "shorts": int(sum(int(row["side"]) == -1 for row in records)),
        "ending_equity": float(realized_equity),
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
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    # A position owns [entry_time, exit_time), so exit_time == end is fully
    # contained and uses only the boundary open to settle the prior half.
    selected = cast(
        pd.DataFrame,
        schedule.loc[
            schedule["source_hour_start"].ge(start)
            & schedule["entry_time"].ge(start)
            & schedule["exit_time"].le(end)
        ].copy(),
    )
    return simulate_strict(
        market,
        funding,
        selected,
        start=start,
        end=end,
        cost_rate_per_side=cost,
        cfg=cfg,
    )


def _stage_gates(
    stage: str,
    base: dict[str, Any],
    stress: dict[str, Any],
    halves: dict[str, dict[str, Any]],
    control_metrics: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float], float]:
    gate = prereg["outcome_gate"]
    primary_ratio = float(base["cagr_to_strict_mdd"])
    control_ratios = {
        name: float(control_metrics[name]["cagr_to_strict_mdd"])
        for name in MECHANISM_CONTROLS
    }
    if any(np.isnan(ratio) or np.isposinf(ratio) for ratio in control_ratios.values()):
        minimum_margin = float("-inf")
    else:
        margins = [primary_ratio - ratio for ratio in control_ratios.values()]
        minimum_margin = min(margins)
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


def _verified_prior_reports(
    stage: str,
    *,
    freeze_hash: str,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for prior in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        payload = _load_json(STAGE_OUTPUTS[prior])
        _verify_manifest(payload, label=f"stored {prior}")
        if payload.get("stage") != prior or payload.get("stage_passed") is not True:
            raise ValueError(f"SQFD-6 {prior} did not pass; {stage} remains sealed")
        prior_index = STAGE_ORDER.index(prior)
        if payload.get("opened_windows") != list(STAGE_ORDER[: prior_index + 1]):
            raise ValueError(f"SQFD-6 {prior} opened an unexpected window")
        if payload.get("sealed_windows") != list(STAGE_ORDER[prior_index + 1 :]):
            raise ValueError(f"SQFD-6 {prior} stage seal changed")
        if payload.get("evaluator_freeze_manifest_hash") != freeze_hash:
            raise ValueError(f"SQFD-6 {prior} froze another evaluator")
        if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
            raise ValueError(f"SQFD-6 {prior} evaluator source changed")
        reports.append(payload)
    return reports


def _build_stage_report(stage: str) -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    prereg, _ = _verify_static_inputs()
    prior = _verified_prior_reports(stage, freeze_hash=freeze["manifest_hash"])
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window(stage)
    cfg = _verify_evaluation_contract(prereg)
    start, end = STAGE_WINDOWS[stage]
    primary_schedule = _window_schedule(schedules["primary"], stage)
    base = simulate_strict(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate_per_side=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    stress = simulate_strict(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate_per_side=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    halves = {
        name: _simulate_window(
            market,
            funding,
            primary_schedule,
            start=half_start,
            end=half_end,
            cost=cfg.base_cost_notional_per_side,
            cfg=cfg,
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
            cost_rate_per_side=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in schedules.items()
        if name != "primary"
    }
    gates, mechanism_ratios, margin = _stage_gates(
        stage, base, stress, halves, controls, prereg
    )
    passed = bool(all(gates.values()))
    index = STAGE_ORDER.index(stage)
    core: dict[str, Any] = {
        "protocol_version": "stablecoin_quote_flow_diffusion_stage_v1",
        "candidate": POLICY_ID,
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_prior_stage_manifest_hashes": {
            row["stage"]: row["manifest_hash"] for row in prior
        },
        "config": asdict(cfg),
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
        "mechanism_control_ratios": mechanism_ratios,
        "minimum_mechanism_control_margin": margin,
        "gates": gates,
        "failed_gates": [
            name for name, passed_gate in gates.items() if not passed_gate
        ],
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


def _metric_row(label: str, headline: dict[str, Any]) -> str:
    return (
        f"| {label} | {headline['absolute_return_pct']:.2f}% | "
        f"{headline['cagr_pct']:.2f}% | {headline['strict_mdd_pct']:.2f}% | "
        f"{headline['cagr_to_strict_mdd']:.2f} | {headline['trades']} | "
        f"{headline['longs']}/{headline['shorts']} | "
        f"{headline['mean_gross_underlying_bp']:.2f}bp | "
        f"{headline['weekly_cluster_signflip_p']:.4f} |"
    )


def render_stage_doc(report: dict[str, Any]) -> str:
    lines = [
        f"# SQFD-6 {report['stage']} strict result — 2026-07-19",
        "",
        "Absolute return uses the full declared calendar. Strict MDD includes the "
        "global/pre-entry HWM, entry and virtual/actual exit costs, conservative "
        "funding boundaries and every held 5m favorable-then-adverse path.",
        "",
        "| Clock | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p(two-sided) |",
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
            "| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | p(two-sided) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, headline in report["primary"]["contained_half_headlines"].items():
        lines.append(_metric_row(name, headline))
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
        raise ValueError(f"SQFD-6 unknown stage: {stage}")
    output = STAGE_OUTPUTS[stage]
    document = STAGE_DOCS[stage]
    if output.exists() or document.exists():
        raise FileExistsError(f"SQFD-6 {stage} result is write-once")
    report = _build_stage_report(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
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
