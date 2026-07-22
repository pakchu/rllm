"""Sequential strict economic evaluator for frozen BFMWD-144 clocks.

The evaluator freeze is outcome-blind.  It binds the source-supported family,
control derivation, strict accounting, statistical correction, and parent data
identities before any BTC OHLC or realized-funding byte is opened or hashed.
Train (2021-2022) must pass before the 2023 selection source can be prepared.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import evaluate_flow_centrality_incubation_relay as strict_engine  # noqa: E402
from training import evaluate_stablecoin_quote_flow_diffusion as strict_source  # noqa: E402
from training import preregister_bitfinex_margin_warehouse_deployment as prereg  # noqa: E402


def _frozen_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("BFMWD economic frozen timestamp is NaT")
    return cast(pd.Timestamp, timestamp)


PROTOCOL_VERSION = "bitfinex_margin_warehouse_deployment_evaluator_v1"
STAGE_PROTOCOL_VERSION = "bitfinex_margin_warehouse_deployment_stage_v1"
SOURCE_PROTOCOL_VERSION = "bitfinex_margin_warehouse_execution_source_v1"
POLICY_ID = prereg.CANDIDATE_FAMILY
AS_OF_DATE = "2026-07-20"
SUPPORT_COMMIT = "dfa1c15cd04acdfe624d28509c8d8876e6e6dc95"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PREREGISTRATION = Path(
    "results/bitfinex_margin_warehouse_deployment_preregistration_2026-07-20.json"
)
SUPPORT_RESULT = Path(
    "results/bitfinex_margin_warehouse_deployment_support_2026-07-20.json"
)
SOURCE_CLOCKS = Path(
    "data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz"
)
EVALUATOR_SOURCE = Path("training/evaluate_bitfinex_margin_warehouse_deployment.py")
EVALUATOR_FREEZE = Path(
    "results/bitfinex_margin_warehouse_deployment_evaluator_freeze_2026-07-20.json"
)
PROTOCOL_DOCUMENT = Path(
    "docs/bitfinex-margin-warehouse-deployment-economic-protocol-2026-07-20.md"
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
    str(prereg.PREREGISTRATION_SOURCE): (
        "67cfa51549ecadcd474ffe81f8037335b09ef761200deebd3870bdb6bbf8405c"
    ),
    "training/evaluate_bitfinex_margin_warehouse_deployment_support.py": (
        "6983246760bfd49a02e059b8dbebea9ca445778f8c700a6d488ed4bf5fffb630"
    ),
    str(PREREGISTRATION): (
        "6e478bac6becb58d282867f4ee612d9d13e803d01985474477d6e3073cd49e58"
    ),
    str(SUPPORT_RESULT): (
        "c857e070f4cb157a005f4a95bee0bff9c7b30daf97128832a690a68d05bfb79c"
    ),
    str(SOURCE_CLOCKS): (
        "02b4fcc462a5a48be7673649f4cf4b2f9bb210baca4294eed1696d479820cccc"
    ),
    "docs/bitfinex-margin-warehouse-deployment-preregistration-2026-07-20.md": (
        "180140619dabba0760033b83a87cfd4f4ca80b09987f7dc2c87e849bc4623fdc"
    ),
    "docs/bitfinex-margin-warehouse-deployment-support-protocol-2026-07-20.md": (
        "e117472e600643e4b082e447814d9f486716494cdaa92be32f1723e05cd459d3"
    ),
    str(PROTOCOL_DOCUMENT): (
        "7aafcaf5a405b504f9f0e83c6a890fa2f3b21956fd5194d8673c5eecd01e88b3"
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

FAMILY_IDS = tuple(variant.variant_id for variant in prereg.VARIANTS)
STAGE_ORDER = ("train", "selection")
STAGE_WINDOWS = {
    "train": (
        _frozen_timestamp(prereg.FROZEN_POLICY.train_start),
        _frozen_timestamp(prereg.FROZEN_POLICY.train_end_exclusive),
    ),
    "selection": (
        _frozen_timestamp(prereg.FROZEN_POLICY.selection_start),
        _frozen_timestamp(prereg.FROZEN_POLICY.selection_end_exclusive),
    ),
}
HALF_WINDOWS = {
    "train": {
        "2021_h1": (
            _frozen_timestamp("2021-01-01T00:00:00Z"),
            _frozen_timestamp("2021-07-01T00:00:00Z"),
        ),
        "2021_h2": (
            _frozen_timestamp("2021-07-01T00:00:00Z"),
            _frozen_timestamp("2022-01-01T00:00:00Z"),
        ),
        "2022_h1": (
            _frozen_timestamp("2022-01-01T00:00:00Z"),
            _frozen_timestamp("2022-07-01T00:00:00Z"),
        ),
        "2022_h2": (
            _frozen_timestamp("2022-07-01T00:00:00Z"),
            _frozen_timestamp("2023-01-01T00:00:00Z"),
        ),
    },
    "selection": {
        "2023_h1": (
            _frozen_timestamp("2023-01-01T00:00:00Z"),
            _frozen_timestamp("2023-07-01T00:00:00Z"),
        ),
        "2023_h2": (
            _frozen_timestamp("2023-07-01T00:00:00Z"),
            _frozen_timestamp("2024-01-01T00:00:00Z"),
        ),
    },
}
STAGE_OUTPUTS = {
    "train": Path(
        "results/bitfinex_margin_warehouse_deployment_train_2021_2022_2026-07-20.json"
    ),
    "selection": Path(
        "results/bitfinex_margin_warehouse_deployment_selection_2023_2026-07-20.json"
    ),
}
STAGE_DOCS = {
    stage: Path(
        f"docs/bitfinex-margin-warehouse-deployment-{stage}-result-2026-07-20.md"
    )
    for stage in STAGE_ORDER
}
STAGE_SOURCE_MANIFESTS = {
    stage: Path(
        "results/"
        f"bitfinex_margin_warehouse_deployment_{stage}_execution_source_2026-07-20.json"
    )
    for stage in STAGE_ORDER
}
STAGE_SOURCE_DIRS = {
    stage: Path("data/bitfinex_margin_warehouse_deployment_execution") / stage
    for stage in STAGE_ORDER
}

PRIMARY_CONTROL = "primary"
CONTROL_ORDER = (
    PRIMARY_CONTROL,
    "direction_flip",
    "fUSD_only",
    "fBTC_only",
    "deterministic_random_side",
    "extra_latency_one_bar",
)
CLOCK_COLUMNS = (
    "candidate",
    "variant_id",
    "control",
    "split",
    "symbol",
    "side",
    "observation_time",
    "source_available_at",
    "decision_available_at",
    "entry_time",
    "exit_time",
)
TIMESTAMP_COLUMNS = (
    "observation_time",
    "source_available_at",
    "decision_available_at",
    "entry_time",
    "exit_time",
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 144
    romano_wolf_draws: int = 100_000
    romano_wolf_block_days: int = 7
    romano_wolf_seed: int = 20_260_720
    bootstrap_batch_draws: int = 2_000


FROZEN_CONFIG = EvaluationConfig()


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("BFMWD economic timestamp is NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
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
    payload = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BFMWD economic JSON must be an object: {path}")
    return payload


def _verify_manifest(payload: Mapping[str, Any], *, label: str) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError(f"BFMWD economic {label} manifest hash changed")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _write_once_bytes(path: str | Path, payload: bytes) -> None:
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"BFMWD economic artifact is write-once: {output}")
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(
            f"BFMWD economic artifact is write-once: {output}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"BFMWD economic frozen input changed: {path}")

    preregistration = _load_json(PREREGISTRATION)
    support = _load_json(SUPPORT_RESULT)
    _verify_manifest(preregistration, label="preregistration")
    _verify_manifest(support, label="source support")
    if preregistration.get("candidate_family") != POLICY_ID:
        raise ValueError("BFMWD economic preregistration identity changed")
    if preregistration.get("economic_gates") != prereg.ECONOMIC_GATES:
        raise ValueError("BFMWD economic gates changed after preregistration")
    if support.get("candidate_family") != POLICY_ID:
        raise ValueError("BFMWD economic support identity changed")
    if support.get("family_support_passed") is not True:
        raise ValueError("BFMWD economic source-support gate did not pass")
    if tuple(support.get("passing_variants", ())) != FAMILY_IDS:
        raise ValueError("BFMWD economic source-supported family changed")
    clock = support.get("clock_artifact", {})
    if clock.get("path") != str(SOURCE_CLOCKS):
        raise ValueError("BFMWD economic source clock path changed")
    if clock.get("sha256") != STATIC_INPUT_SHA256[str(SOURCE_CLOCKS)]:
        raise ValueError("BFMWD economic source clock binding changed")
    boundary = support.get("outcome_boundary", {})
    expected_boundary = {
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "btc_market_rows_read": 0,
        "funding_paid_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "post_2023_rows_read": 0,
    }
    if boundary != expected_boundary:
        raise ValueError("BFMWD economic support outcome boundary changed")

    market_manifest = _load_json(LEGACY_MARKET_MANIFEST)
    if (
        market_manifest.get("combined_output") != str(LEGACY_MARKET)
        or market_manifest.get("combined_sha256") != LEGACY_MARKET_SHA256
        or market_manifest.get("config", {}).get("end") != "2024-01-01"
    ):
        raise ValueError("BFMWD economic market-manifest contract changed")
    funding_manifest = _load_json(LEGACY_FUNDING_MANIFEST)
    funding_data = funding_manifest.get("data", {})
    if (
        funding_data.get("path") != str(LEGACY_FUNDING)
        or funding_data.get("sha256") != LEGACY_FUNDING_SHA256
        or funding_manifest.get("selection_end_exclusive") != "2024-01-01 00:00:00"
    ):
        raise ValueError("BFMWD economic funding-manifest contract changed")
    return preregistration, support


def _deterministic_random_side(variant_id: str, entry_time: Any) -> int:
    material = f"{POLICY_ID}|{variant_id}|{_utc(entry_time).isoformat()}".encode(
        "ascii"
    )
    return 1 if hashlib.sha256(material).digest()[0] & 1 == 0 else -1


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "variant_id": str(row["variant_id"]),
            "control": str(row["control"]),
            "split": str(row["split"]),
            "symbol": str(row["symbol"]),
            "side": int(row["side"]),
            "entry_time": _utc(row["entry_time"]).isoformat(),
            "exit_time": _utc(row["exit_time"]).isoformat(),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def _validate_nonoverlap(frame: pd.DataFrame, *, label: str) -> None:
    ordered = frame.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    if len(ordered) > 1:
        entries = cast(pd.Series, ordered["entry_time"]).iloc[1:].reset_index(drop=True)
        exits = cast(pd.Series, ordered["exit_time"]).iloc[:-1].reset_index(drop=True)
        if not bool(entries.ge(exits).all()):
            raise ValueError(f"BFMWD economic {label} schedule overlaps")


def load_primary_clocks() -> tuple[pd.DataFrame, dict[str, Any]]:
    _, support = _verify_static_inputs()
    frame = cast(pd.DataFrame, pd.read_csv(_path(SOURCE_CLOCKS)))
    if tuple(str(column) for column in frame.columns) != CLOCK_COLUMNS:
        raise ValueError("BFMWD economic source clock schema changed")
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    expected_controls = {PRIMARY_CONTROL, *prereg.SOURCE_ONLY_CONTROLS}
    if set(frame["control"]) != expected_controls:
        raise ValueError("BFMWD economic source clock control family changed")
    primary = cast(
        pd.DataFrame,
        frame.loc[frame["control"].eq(PRIMARY_CONTROL)].copy(),
    )
    if set(primary["candidate"]) != {POLICY_ID}:
        raise ValueError("BFMWD economic clock candidate changed")
    if set(primary["variant_id"]) != set(FAMILY_IDS):
        raise ValueError("BFMWD economic clock family changed")
    if set(primary["split"]) != set(STAGE_ORDER):
        raise ValueError("BFMWD economic clock stages changed")
    if not bool(primary["side"].isin((-1, 1)).all()):
        raise ValueError("BFMWD economic primary side changed")
    expected_side = cast(pd.Series, primary["symbol"]).apply(
        lambda value: {"fUSD": 1, "fBTC": -1}.get(str(value))
    )
    if bool(expected_side.isna().any()) or not bool(
        primary["side"].eq(expected_side).all()
    ):
        raise ValueError("BFMWD economic symbol-direction mapping changed")
    if not bool(
        (primary["entry_time"] - primary["decision_available_at"])
        .eq(pd.Timedelta(minutes=5))
        .all()
    ):
        raise ValueError("BFMWD economic next-open delay changed")
    if not bool(
        (primary["exit_time"] - primary["entry_time"]).eq(pd.Timedelta(hours=12)).all()
    ):
        raise ValueError("BFMWD economic hold changed")
    if not bool(
        primary["source_available_at"].le(primary["decision_available_at"]).all()
    ):
        raise ValueError("BFMWD economic decision precedes source availability")

    expected_counts: dict[tuple[str, str], int] = {}
    for variant_id in FAMILY_IDS:
        frozen = support["variants"][variant_id]
        if frozen.get("support_passed") is not True:
            raise ValueError(
                f"BFMWD economic unsupported variant advanced: {variant_id}"
            )
        for stage in STAGE_ORDER:
            expected_counts[(variant_id, stage)] = int(
                frozen["support"][stage]["events"]
            )
    observed_counts = primary.groupby(["variant_id", "split"]).size().to_dict()
    if observed_counts != expected_counts:
        raise ValueError("BFMWD economic primary counts changed from source support")

    for variant_id in FAMILY_IDS:
        for stage in STAGE_ORDER:
            start, end = STAGE_WINDOWS[stage]
            selected = cast(
                pd.DataFrame,
                primary.loc[
                    primary["variant_id"].eq(variant_id) & primary["split"].eq(stage)
                ],
            )
            if not (
                bool(selected["entry_time"].ge(start).all())
                and bool(selected["exit_time"].lt(end).all())
            ):
                raise ValueError(
                    f"BFMWD economic {variant_id}/{stage} crosses exclusive boundary"
                )
            _validate_nonoverlap(selected, label=f"{variant_id}/{stage}/primary")
    primary = primary.sort_values(
        ["variant_id", "split", "entry_time"], kind="mergesort"
    ).reset_index(drop=True)
    return primary, support


def derive_schedules(primary: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    schedules: dict[str, dict[str, pd.DataFrame]] = {}
    for variant_id in FAMILY_IDS:
        source = cast(
            pd.DataFrame,
            primary.loc[primary["variant_id"].eq(variant_id)].copy(),
        ).sort_values(["split", "entry_time"], kind="mergesort")
        controls: dict[str, pd.DataFrame] = {}
        controls[PRIMARY_CONTROL] = source.copy()

        direction_flip = source.copy()
        direction_flip["side"] = -direction_flip["side"]
        controls["direction_flip"] = direction_flip
        controls["fUSD_only"] = cast(
            pd.DataFrame, source.loc[source["symbol"].eq("fUSD")].copy()
        )
        controls["fBTC_only"] = cast(
            pd.DataFrame, source.loc[source["symbol"].eq("fBTC")].copy()
        )
        random_side = source.copy()
        random_side["side"] = [
            _deterministic_random_side(variant_id, value)
            for value in random_side["entry_time"]
        ]
        controls["deterministic_random_side"] = random_side
        delayed = source.copy()
        delayed["entry_time"] = delayed["entry_time"] + pd.Timedelta(minutes=5)
        delayed["exit_time"] = delayed["exit_time"] + pd.Timedelta(minutes=5)
        contained = pd.Series(False, index=delayed.index)
        for stage, (start, end) in STAGE_WINDOWS.items():
            contained |= (
                delayed["split"].eq(stage)
                & delayed["entry_time"].ge(start)
                & delayed["exit_time"].lt(end)
            )
        controls["extra_latency_one_bar"] = cast(
            pd.DataFrame, delayed.loc[contained].copy()
        )

        for control, schedule in controls.items():
            schedule["control"] = control
            schedule = schedule.sort_values(
                ["split", "entry_time"], kind="mergesort"
            ).reset_index(drop=True)
            if not bool(schedule["side"].isin((-1, 1)).all()):
                raise ValueError(f"BFMWD economic {variant_id}/{control} side changed")
            for stage in STAGE_ORDER:
                stage_schedule = cast(
                    pd.DataFrame,
                    schedule.loc[schedule["split"].eq(stage)],
                )
                _validate_nonoverlap(
                    stage_schedule, label=f"{variant_id}/{stage}/{control}"
                )
                if len(stage_schedule):
                    start, end = STAGE_WINDOWS[stage]
                    if not (
                        bool(stage_schedule["entry_time"].ge(start).all())
                        and bool(stage_schedule["exit_time"].lt(end).all())
                    ):
                        raise ValueError(
                            f"BFMWD economic {variant_id}/{stage}/{control} "
                            "crosses exclusive boundary"
                        )
            controls[control] = schedule
        if tuple(controls) != CONTROL_ORDER:
            raise ValueError("BFMWD economic control order changed")
        schedules[variant_id] = controls
    return schedules


def _window_schedule(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    start, end = STAGE_WINDOWS[stage]
    selected = (
        cast(
            pd.DataFrame,
            frame.loc[
                frame["split"].eq(stage)
                & frame["entry_time"].ge(start)
                & frame["exit_time"].lt(end)
            ].copy(),
        )
        .sort_values("entry_time", kind="mergesort")
        .reset_index(drop=True)
    )
    return selected


def _stage_source_spec(stage: str) -> dict[str, Any]:
    start, end = STAGE_WINDOWS[stage]
    return {
        "stage": stage,
        "required_manifest": str(STAGE_SOURCE_MANIFESTS[stage]),
        "required_protocol_version": SOURCE_PROTOCOL_VERSION,
        "physical_window": [start.isoformat(), end.isoformat()],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": False,
        "strategy_outcomes_calculated": False,
        "full_parent_compressed_bytes_hashed": stage == "selection",
        "parent_digest_deferred_until_selection": stage == "train",
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    output = _path(output_path)
    if output.exists():
        raise FileExistsError("BFMWD economic evaluator freeze is write-once")
    if any(_path(path).exists() for path in STAGE_OUTPUTS.values()):
        raise RuntimeError(
            "BFMWD economic evaluator cannot freeze after a stage result"
        )
    preexisting_stage_sources = [
        path
        for stage in STAGE_ORDER
        for path in (
            STAGE_SOURCE_MANIFESTS[stage],
            STAGE_SOURCE_DIRS[stage] / "BTCUSDT_5m.csv.gz",
            STAGE_SOURCE_DIRS[stage] / "BTCUSDT_funding_marks.csv.gz",
        )
        if _path(path).exists()
    ]
    if preexisting_stage_sources:
        raise RuntimeError(
            "BFMWD economic evaluator cannot freeze after a stage source exists"
        )
    preregistration, support = _verify_static_inputs()
    primary, _ = load_primary_clocks()
    schedules = derive_schedules(primary)
    schedule_records = {
        variant_id: {
            control: {
                "events": len(schedule),
                "schedule_hash": _schedule_hash(schedule),
                "stage_counts": {
                    stage: len(_window_schedule(schedule, stage))
                    for stage in STAGE_ORDER
                },
            }
            for control, schedule in controls.items()
        }
        for variant_id, controls in schedules.items()
    }
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_family": POLICY_ID,
        "family_ids": list(FAMILY_IDS),
        "as_of_date": AS_OF_DATE,
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "support_manifest_hash": support["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "protocol_document": str(PROTOCOL_DOCUMENT),
        "protocol_document_sha256": STATIC_INPUT_SHA256[str(PROTOCOL_DOCUMENT)],
        "strict_engine_dependency": str(strict_engine.EVALUATOR_SOURCE),
        "strict_engine_dependency_sha256": STATIC_INPUT_SHA256[
            str(strict_engine.EVALUATOR_SOURCE)
        ],
        "strict_source_dependency": str(strict_source.EVALUATOR_SOURCE),
        "strict_source_dependency_sha256": STATIC_INPUT_SHA256[
            str(strict_source.EVALUATOR_SOURCE)
        ],
        "evaluation_config": asdict(FROZEN_CONFIG),
        "economic_gates": dict(prereg.ECONOMIC_GATES),
        "static_inputs": dict(STATIC_INPUT_SHA256),
        "source_clock": str(SOURCE_CLOCKS),
        "source_clock_sha256": STATIC_INPUT_SHA256[str(SOURCE_CLOCKS)],
        "schedule_records": schedule_records,
        "control_order": list(CONTROL_ORDER),
        "stage_windows": {
            stage: [start.isoformat(), end.isoformat()]
            for stage, (start, end) in STAGE_WINDOWS.items()
        },
        "half_windows": {
            stage: {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in windows.items()
            }
            for stage, windows in HALF_WINDOWS.items()
        },
        "execution_source_specs": {
            stage: _stage_source_spec(stage) for stage in STAGE_ORDER
        },
        "legacy_container_contract": {
            "eligible_stages": list(STAGE_ORDER),
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
        "strict_accounting": {
            "funding_boundary": (
                "interior symmetric; exact entry/exit credits dropped and debits retained"
            ),
            "mdd": (
                "global/pre-entry HWM; costs; exact funding marks; every held 5m "
                "favorable-then-adverse OHLC path"
            ),
            "cagr": "full declared stage calendar including idle cash",
            "entry_exit": "frozen next-5m-open clock and exact 144-bar hold",
            "stage_end": "exclusive; every admitted exit strictly before boundary",
        },
        "romano_wolf": {
            "method": "one-sided step-down max-t circular block bootstrap",
            "family": list(FAMILY_IDS),
            "realized_return_clock": "UTC exit calendar day; idle days are zero",
            "draws": FROZEN_CONFIG.romano_wolf_draws,
            "block_days": FROZEN_CONFIG.romano_wolf_block_days,
            "seed": FROZEN_CONFIG.romano_wolf_seed,
            "synchronized_indices": True,
        },
        "opened_windows": [],
        "sealed_windows": list(STAGE_ORDER),
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "execution_outcome_data_bytes_hashed_during_freeze": False,
        "simulation_run_during_freeze": False,
        "post_2023_access_supported": False,
        "mutable_parameters": [],
    }
    report = _seal(core)
    _write_once_bytes(output, _json_bytes(report))
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_manifest(payload, label="evaluator freeze")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("BFMWD economic evaluator protocol changed")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("BFMWD economic evaluator source changed after freeze")
    if payload.get("evaluation_config") != asdict(FROZEN_CONFIG):
        raise ValueError("BFMWD economic evaluator configuration changed")
    if payload.get("economic_gates") != prereg.ECONOMIC_GATES:
        raise ValueError("BFMWD economic evaluator gates changed")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("BFMWD economic evaluator is not sealed")
    if payload.get("sealed_windows") != list(STAGE_ORDER):
        raise ValueError("BFMWD economic evaluator stage seal changed")
    if payload.get("post_2023_access_supported") is not False:
        raise ValueError("BFMWD economic evaluator opened post-2023 access")
    for field in (
        "execution_ohlc_rows_parsed_during_freeze",
        "funding_rows_parsed_during_freeze",
    ):
        if payload.get(field) != 0:
            raise ValueError(f"BFMWD economic evaluator changed {field}")
    if payload.get("execution_outcome_data_bytes_hashed_during_freeze") is not False:
        raise ValueError("BFMWD economic evaluator freeze hashed outcomes")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("BFMWD economic evaluator freeze simulated outcomes")
    preregistration, support = _verify_static_inputs()
    if payload.get("preregistration_manifest_hash") != preregistration["manifest_hash"]:
        raise ValueError("BFMWD economic evaluator preregistration binding changed")
    if payload.get("support_manifest_hash") != support["manifest_hash"]:
        raise ValueError("BFMWD economic evaluator support binding changed")
    primary, _ = load_primary_clocks()
    schedules = derive_schedules(primary)
    for variant_id, controls in schedules.items():
        for control, schedule in controls.items():
            record = payload["schedule_records"][variant_id][control]
            if record["schedule_hash"] != _schedule_hash(schedule):
                raise ValueError(
                    f"BFMWD economic {variant_id}/{control} schedule changed"
                )
    return payload


def _verified_prior_reports(stage: str, *, freeze_hash: str) -> list[dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"BFMWD economic unknown stage: {stage}")
    reports: list[dict[str, Any]] = []
    for prior in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        payload = _load_json(STAGE_OUTPUTS[prior])
        _verify_manifest(payload, label=f"stored {prior}")
        if payload.get("stage") != prior or payload.get("stage_passed") is not True:
            raise ValueError(
                f"BFMWD economic {prior} did not pass; {stage} remains sealed"
            )
        if not payload.get("passing_variants"):
            raise ValueError(f"BFMWD economic {prior} has no advancing variant")
        index = STAGE_ORDER.index(prior)
        if payload.get("opened_windows") != list(STAGE_ORDER[: index + 1]):
            raise ValueError(f"BFMWD economic {prior} opened an unexpected window")
        if payload.get("sealed_windows") != list(STAGE_ORDER[index + 1 :]):
            raise ValueError(f"BFMWD economic {prior} stage seal changed")
        if payload.get("evaluator_freeze_manifest_hash") != freeze_hash:
            raise ValueError(f"BFMWD economic {prior} froze another evaluator")
        if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
            raise ValueError(f"BFMWD economic {prior} evaluator source changed")
        reports.append(payload)
    return reports


def _slice_gzip_csv(
    source: str | Path,
    output: str | Path,
    *,
    timestamp_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_rows: int,
) -> dict[str, Any]:
    """Copy only the stage interval while parsing no non-timestamp value."""
    source_path = _path(source)
    output_path = _path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"BFMWD economic stage source is write-once: {output_path}"
        )
    rows = 0
    prior_rows_skipped = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    if expected_rows < 1:
        raise ValueError("BFMWD economic expected stage-source rows are invalid")
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
            if timestamp_column not in header:
                raise ValueError("BFMWD economic source timestamp column is absent")
            timestamp_index = header.index(timestamp_column)
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", mtime=0
                ) as compressed:
                    compressed.write(header_line.encode("utf-8"))
                    while rows < expected_rows:
                        line = input_handle.readline()
                        if not line:
                            break
                        fields = line.rstrip("\r\n").split(",")
                        timestamp = _utc(fields[timestamp_index])
                        if timestamp < start:
                            prior_rows_skipped += 1
                            continue
                        if timestamp >= end:
                            raise ValueError(
                                "BFMWD economic stage source ended before its "
                                "expected row count"
                            )
                        if last is not None and timestamp <= last:
                            raise ValueError(
                                "BFMWD economic source timestamps are not strictly increasing"
                            )
                        compressed.write(line.encode("utf-8"))
                        first = timestamp if first is None else first
                        last = timestamp
                        rows += 1
        if rows != expected_rows or first is None or last is None:
            raise ValueError("BFMWD economic stage source row count changed")
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"BFMWD economic stage source is write-once: {output_path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(Path(output).as_posix()),
        "sha256": _sha256(output_path),
        "rows": rows,
        "expected_rows": expected_rows,
        "prior_rows_skipped_by_timestamp_only": prior_rows_skipped,
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
        "first_excluded_row_read": False,
        "post_stage_numeric_rows_parsed": 0,
    }


def prepare_stage_source(stage: str) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError("BFMWD economic supports only train and selection")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    manifest_path = STAGE_SOURCE_MANIFESTS[stage]
    if _path(manifest_path).exists():
        raise FileExistsError(f"BFMWD economic {stage} source is write-once")
    directory = STAGE_SOURCE_DIRS[stage]
    market_path = directory / "BTCUSDT_5m.csv.gz"
    funding_path = directory / "BTCUSDT_funding_marks.csv.gz"
    if any(_path(path).exists() for path in (market_path, funding_path)):
        raise FileExistsError(
            f"BFMWD economic {stage} has an orphaned write-once source"
        )
    start, end = STAGE_WINDOWS[stage]
    spec = cast(dict[str, Any], freeze["execution_source_specs"][stage])
    if stage == "selection":
        if _sha256(LEGACY_MARKET) != LEGACY_MARKET_SHA256:
            raise ValueError("BFMWD economic legacy market bytes changed")
        if _sha256(LEGACY_FUNDING) != LEGACY_FUNDING_SHA256:
            raise ValueError("BFMWD economic legacy funding bytes changed")
    market_rows = len(pd.date_range(start, end, freq="5min", inclusive="left"))
    funding_rows = len(pd.date_range(start, end, freq="8h", inclusive="left"))
    directory_path = _path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    if any(directory_path.glob(".prepare-*")):
        raise FileExistsError(
            f"BFMWD economic {stage} has an interrupted preparation artifact"
        )
    with tempfile.TemporaryDirectory(
        dir=directory_path, prefix=".prepare-"
    ) as temporary_directory:
        pending_market = Path(temporary_directory) / "BTCUSDT_5m.csv.gz"
        pending_funding = Path(temporary_directory) / "BTCUSDT_funding_marks.csv.gz"
        market = _slice_gzip_csv(
            LEGACY_MARKET,
            pending_market,
            timestamp_column="date",
            start=start,
            end=end,
            expected_rows=market_rows,
        )
        funding = _slice_gzip_csv(
            LEGACY_FUNDING,
            pending_funding,
            timestamp_column="funding_time_utc",
            start=start,
            end=end,
            expected_rows=funding_rows,
        )
        parsed_market, market_diagnostics = strict_source._parse_market_window(
            pending_market,
            start,
            end,
            require_exact_physical_window=True,
            include_end_boundary=False,
        )
        parsed_funding, funding_diagnostics = strict_source._parse_funding_window(
            pending_funding,
            start,
            end,
            require_exact_physical_window=True,
            include_end_boundary=False,
        )
        if (
            len(parsed_market) != market["rows"]
            or len(parsed_funding) != funding["rows"]
        ):
            raise ValueError("BFMWD economic stage-source validation count changed")
        market["path"] = str(market_path)
        funding["path"] = str(funding_path)
        core = {
            "protocol_version": SOURCE_PROTOCOL_VERSION,
            "candidate_family": POLICY_ID,
            "stage": stage,
            "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": False,
            "strategy_outcomes_calculated": False,
            "official_manifest_hashes_verified": True,
            "full_parent_compressed_bytes_hashed": spec[
                "full_parent_compressed_bytes_hashed"
            ],
            "parent_digest_deferred_until_selection": spec[
                "parent_digest_deferred_until_selection"
            ],
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": freeze["legacy_container_contract"]["market"],
            "parent_funding": freeze["legacy_container_contract"]["funding"],
            "market": {**market, "diagnostics": market_diagnostics},
            "funding": {**funding, "diagnostics": funding_diagnostics},
        }
        report = _seal(core)
        os.link(pending_market, _path(market_path))
        os.link(pending_funding, _path(funding_path))
        _write_once_bytes(manifest_path, _json_bytes(report))
        return report


def _load_stage_source(stage: str, *, freeze: Mapping[str, Any]) -> dict[str, Any]:
    spec = cast(dict[str, Any], freeze["execution_source_specs"][stage])
    payload = _load_json(STAGE_SOURCE_MANIFESTS[stage])
    _verify_manifest(payload, label=f"{stage} execution source")
    if payload.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise ValueError(f"BFMWD economic {stage} source protocol changed")
    if payload.get("candidate_family") != POLICY_ID or payload.get("stage") != stage:
        raise ValueError(f"BFMWD economic {stage} source identity changed")
    for key in (
        "physical_window",
        "physical_rows_limited_to_window",
        "exit_boundary_required",
        "strategy_outcomes_calculated",
        "full_parent_compressed_bytes_hashed",
        "parent_digest_deferred_until_selection",
    ):
        if payload.get(key) != spec[key]:
            raise ValueError(f"BFMWD economic {stage} source {key} changed")
    if payload.get("evaluator_freeze_manifest_hash") != freeze["manifest_hash"]:
        raise ValueError(f"BFMWD economic {stage} source froze another evaluator")
    if payload.get("official_manifest_hashes_verified") is not True:
        raise ValueError(f"BFMWD economic {stage} source lacks manifest verification")
    if payload.get("post_stage_numeric_rows_parsed") != 0:
        raise ValueError(f"BFMWD economic {stage} parsed a future numeric row")
    if payload.get("parent_market") != freeze["legacy_container_contract"]["market"]:
        raise ValueError(f"BFMWD economic {stage} parent market binding changed")
    if payload.get("parent_funding") != freeze["legacy_container_contract"]["funding"]:
        raise ValueError(f"BFMWD economic {stage} parent funding binding changed")
    expected_paths = {
        "market": STAGE_SOURCE_DIRS[stage] / "BTCUSDT_5m.csv.gz",
        "funding": STAGE_SOURCE_DIRS[stage] / "BTCUSDT_funding_marks.csv.gz",
    }
    for name, expected in expected_paths.items():
        item = payload.get(name)
        if not isinstance(item, dict) or item.get("path") != str(expected):
            raise ValueError(f"BFMWD economic {stage} {name} path changed")
        if _sha256(expected) != item.get("sha256"):
            raise ValueError(f"BFMWD economic {stage} {name} bytes changed")
    return payload


def load_execution_window(
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"BFMWD economic unknown stage: {stage}")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    contract = _load_stage_source(stage, freeze=freeze)
    source_manifest_path = STAGE_SOURCE_MANIFESTS[stage]
    start, end = STAGE_WINDOWS[stage]
    market_item = contract["market"]
    funding_item = contract["funding"]
    market, market_diagnostics = strict_source._parse_market_window(
        _path(market_item["path"]),
        start,
        end,
        require_exact_physical_window=True,
        include_end_boundary=False,
    )
    funding, funding_diagnostics = strict_source._parse_funding_window(
        _path(funding_item["path"]),
        start,
        end,
        require_exact_physical_window=True,
        include_end_boundary=False,
    )
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
            "execution_source_manifest": {
                "path": str(source_manifest_path),
                "sha256": _sha256(source_manifest_path),
                "manifest_hash": contract["manifest_hash"],
            },
            "stage_source_paths": {
                "market": market_item["path"],
                "funding": funding_item["path"],
            },
            "parent_contract": {
                "market": contract["parent_market"],
                "funding": contract["parent_funding"],
            },
        },
    )


def _realized_daily_log_returns(
    metrics: Mapping[str, Any], *, start: pd.Timestamp, end: pd.Timestamp
) -> np.ndarray:
    days = (end - start).days
    if end - start != pd.Timedelta(days=days) or days < 1:
        raise ValueError("BFMWD economic stage is not a whole-day calendar")
    daily = np.zeros(days, dtype=np.float64)
    for trade in metrics["trade_details"]:
        net_return = float(trade["net_return"])
        if not math.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("BFMWD economic trade return is invalid")
        exit_time = _utc(trade["exit_time"])
        index = (exit_time.floor("D") - start).days
        if not 0 <= index < days:
            raise ValueError("BFMWD economic realized return escaped the stage")
        daily[index] += math.log1p(net_return)
    ending_equity = float(metrics["ending_equity"])
    if ending_equity <= 0.0 or not math.isclose(
        float(daily.sum()), math.log(ending_equity), abs_tol=1e-12
    ):
        raise ValueError("BFMWD economic realized daily returns are inconsistent")
    return daily


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate_per_side: float,
) -> tuple[dict[str, Any], np.ndarray]:
    metrics = strict_engine.simulate_strict(
        market,
        funding,
        clocks,
        start=start,
        end=end,
        cost_rate_per_side=cost_rate_per_side,
    )
    daily = _realized_daily_log_returns(metrics, start=start, end=end)
    metrics["realized_daily_log_return_observations"] = int(len(daily))
    metrics["realized_daily_log_return_sha256"] = hashlib.sha256(
        daily.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    return metrics, daily


def _slice_execution_frames(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_window = cast(
        pd.DataFrame,
        market.loc[market["date"].ge(start) & market["date"].lt(end)].copy(),
    )
    funding_window = cast(
        pd.DataFrame,
        funding.loc[
            funding["funding_time"].ge(start) & funding["funding_time"].lt(end)
        ].copy(),
    )
    return market_window, funding_window


def _simulate_window(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost: float,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = cast(
        pd.DataFrame,
        schedule.loc[
            schedule["entry_time"].ge(start) & schedule["exit_time"].lt(end)
        ].copy(),
    )
    market_window, funding_window = _slice_execution_frames(
        market, funding, start=start, end=end
    )
    return simulate_strict(
        market_window,
        funding_window,
        selected,
        start=start,
        end=end,
        cost_rate_per_side=cost,
    )


def _studentized_mean(values: np.ndarray) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError("BFMWD economic Romano-Wolf input is invalid")
    if float(np.ptp(sample)) == 0.0:
        return 0.0
    deviation = float(sample.std(ddof=1))
    return math.sqrt(len(sample)) * float(sample.mean()) / deviation


def romano_wolf_stepdown(
    daily_returns: Mapping[str, np.ndarray],
    tested_ids: Sequence[str],
    *,
    draws: int = FROZEN_CONFIG.romano_wolf_draws,
    block_days: int = FROZEN_CONFIG.romano_wolf_block_days,
    seed: int = FROZEN_CONFIG.romano_wolf_seed,
    batch_draws: int = FROZEN_CONFIG.bootstrap_batch_draws,
) -> dict[str, Any]:
    if tuple(daily_returns) != FAMILY_IDS:
        raise ValueError("BFMWD economic Romano-Wolf family order changed")
    matrix = np.vstack(
        [
            np.asarray(daily_returns[variant_id], dtype=np.float64)
            for variant_id in FAMILY_IDS
        ]
    )
    if (
        matrix.ndim != 2
        or matrix.shape[1] < 2
        or not np.isfinite(matrix).all()
        or draws < 1
        or block_days < 1
        or batch_draws < 1
    ):
        raise ValueError("BFMWD economic Romano-Wolf configuration changed")
    tested = set(tested_ids)
    if not tested <= set(FAMILY_IDS):
        raise ValueError("BFMWD economic Romano-Wolf tested family escaped")
    observed = {
        variant_id: _studentized_mean(matrix[index])
        for index, variant_id in enumerate(FAMILY_IDS)
    }
    variance_positive = {
        variant_id: float(np.ptp(matrix[index])) > 0.0
        for index, variant_id in enumerate(FAMILY_IDS)
    }
    eligible = tuple(
        variant_id
        for variant_id in FAMILY_IDS
        if variant_id in tested and variance_positive[variant_id]
    )
    order = tuple(sorted(eligible, key=lambda item: (-observed[item], item)))
    adjusted = {variant_id: 1.0 for variant_id in FAMILY_IDS}
    raw_stepdown = {variant_id: 1.0 for variant_id in FAMILY_IDS}
    if order:
        centered = matrix - matrix.mean(axis=1, keepdims=True)
        family_index = {
            variant_id: index for index, variant_id in enumerate(FAMILY_IDS)
        }
        ordered_indices = [family_index[variant_id] for variant_id in order]
        exceedances = np.zeros(len(order), dtype=np.int64)
        generator = np.random.default_rng(seed)
        blocks_per_draw = math.ceil(matrix.shape[1] / block_days)
        block_offsets = np.arange(block_days, dtype=np.int64)
        completed = 0
        while completed < draws:
            current = min(batch_draws, draws - completed)
            starts = generator.integers(
                0,
                matrix.shape[1],
                size=(current, blocks_per_draw),
                dtype=np.int64,
            )
            indices = (
                starts[:, :, None] + block_offsets[None, None, :]
            ) % matrix.shape[1]
            indices = indices.reshape(current, -1)[:, : matrix.shape[1]]
            bootstrap_t = np.empty((current, len(order)), dtype=np.float64)
            for column, row_index in enumerate(ordered_indices):
                samples = centered[row_index][indices]
                deviations = samples.std(axis=1, ddof=1)
                statistics = np.zeros(current, dtype=np.float64)
                np.divide(
                    math.sqrt(matrix.shape[1]) * samples.mean(axis=1),
                    deviations,
                    out=statistics,
                    where=deviations > 0.0,
                )
                bootstrap_t[:, column] = statistics
            suffix_max = np.maximum.accumulate(bootstrap_t[:, ::-1], axis=1)[:, ::-1]
            tie_group_start = 0
            for column, variant_id in enumerate(order):
                if column > 0 and observed[variant_id] != observed[order[column - 1]]:
                    tie_group_start = column
                exceedances[column] += int(
                    np.count_nonzero(
                        suffix_max[:, tie_group_start] >= observed[variant_id]
                    )
                )
            completed += current
        monotone = 0.0
        for column, variant_id in enumerate(order):
            raw = (int(exceedances[column]) + 1.0) / (draws + 1.0)
            monotone = max(monotone, raw)
            raw_stepdown[variant_id] = raw
            adjusted[variant_id] = monotone
    return {
        "method": "one-sided Romano-Wolf step-down max-t circular block bootstrap",
        "draws": draws,
        "block_days": block_days,
        "seed": seed,
        "batch_draws": batch_draws,
        "daily_observations": int(matrix.shape[1]),
        "synchronized_indices": True,
        "equal_observed_t_removed_as_one_group": True,
        "ordered_tested_variant_ids": list(order),
        "observed_t": observed,
        "variance_positive": variance_positive,
        "raw_stepdown_p": raw_stepdown,
        "adjusted_p": adjusted,
    }


def _headline(metrics: Mapping[str, Any]) -> dict[str, Any]:
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
        "weekly_cluster_signflip_p_two_sided_unadjusted": significance[
            "p_value_two_sided"
        ],
        "weekly_clusters": significance["cluster_count"],
        "realized_daily_log_return_observations": metrics[
            "realized_daily_log_return_observations"
        ],
        "realized_daily_log_return_sha256": metrics["realized_daily_log_return_sha256"],
    }


def _gate_checks(
    *,
    base: Mapping[str, Any],
    stress: Mapping[str, Any],
    halves: Mapping[str, Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    adjusted_p: float,
) -> dict[str, bool]:
    gates = prereg.ECONOMIC_GATES
    return {
        "absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "cagr_to_strict_mdd_at_least_3": float(base["cagr_to_strict_mdd"])
        >= float(gates["minimum_cagr_to_strict_mdd"]),
        "strict_mdd_at_most_15pct": float(base["strict_mdd_pct"])
        <= float(gates["maximum_strict_mdd"]) * 100.0,
        "each_contained_calendar_half_positive": all(
            float(item["absolute_return_pct"]) > 0.0 for item in halves.values()
        ),
        "fUSD_long_contribution_positive": float(
            controls["fUSD_only"]["absolute_return_pct"]
        )
        > 0.0,
        "fBTC_short_contribution_positive": float(
            controls["fBTC_only"]["absolute_return_pct"]
        )
        > 0.0,
        "mean_gross_side_adjusted_move_at_least_30bp": float(
            base["mean_gross_underlying_bp"]
        )
        >= float(gates["minimum_mean_gross_side_adjusted_bp"]),
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "stress_cagr_to_strict_mdd_at_least_2_5": float(stress["cagr_to_strict_mdd"])
        >= float(gates["minimum_stress_cagr_to_strict_mdd"]),
        "one_bar_delay_absolute_return_positive": float(
            controls["extra_latency_one_bar"]["absolute_return_pct"]
        )
        > 0.0,
        "romano_wolf_adjusted_p_at_most_10pct": adjusted_p
        <= float(gates["weekly_cluster_pvalue_maximum"]),
    }


def _active_variants(stage: str, prior: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if stage == "train":
        return FAMILY_IDS
    if stage == "selection" and len(prior) == 1:
        active = tuple(str(item) for item in prior[0]["passing_variants"])
        if not active or not set(active) <= set(FAMILY_IDS):
            raise ValueError("BFMWD economic selection active family changed")
        return tuple(item for item in FAMILY_IDS if item in set(active))
    raise ValueError("BFMWD economic stage prior-report sequence changed")


def _build_stage_report(stage: str) -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    prior = _verified_prior_reports(
        stage, freeze_hash=cast(str, freeze["manifest_hash"])
    )
    primary, _ = load_primary_clocks()
    schedules = derive_schedules(primary)
    active = _active_variants(stage, prior)
    market, funding, diagnostics = load_execution_window(stage)
    start, end = STAGE_WINDOWS[stage]
    expected_days = int((end - start).days)
    variant_results: dict[str, dict[str, Any]] = {}
    daily_returns = {
        variant_id: np.zeros(expected_days, dtype=np.float64)
        for variant_id in FAMILY_IDS
    }
    for variant_id in active:
        controls = {
            name: _window_schedule(schedule, stage)
            for name, schedule in schedules[variant_id].items()
        }
        base, daily = simulate_strict(
            market,
            funding,
            controls[PRIMARY_CONTROL],
            start=start,
            end=end,
            cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
        )
        daily_returns[variant_id] = daily
        stress, _ = simulate_strict(
            market,
            funding,
            controls[PRIMARY_CONTROL],
            start=start,
            end=end,
            cost_rate_per_side=FROZEN_CONFIG.stress_cost_notional_per_side,
        )
        halves = {
            name: _simulate_window(
                market,
                funding,
                controls[PRIMARY_CONTROL],
                start=half_start,
                end=half_end,
                cost=FROZEN_CONFIG.base_cost_notional_per_side,
            )[0]
            for name, (half_start, half_end) in HALF_WINDOWS[stage].items()
        }
        control_metrics = {
            name: simulate_strict(
                market,
                funding,
                schedule,
                start=start,
                end=end,
                cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
            )[0]
            for name, schedule in controls.items()
            if name != PRIMARY_CONTROL
        }
        variant_results[variant_id] = {
            "primary_metrics": base,
            "primary_headline": _headline(base),
            "stress_headline": _headline(stress),
            "contained_half_headlines": {
                name: _headline(item) for name, item in halves.items()
            },
            "control_headlines": {
                name: _headline(item) for name, item in control_metrics.items()
            },
        }

    inference = romano_wolf_stepdown(daily_returns, active)
    passing: list[str] = []
    for variant_id in active:
        result = variant_results[variant_id]
        adjusted_p = float(inference["adjusted_p"][variant_id])
        checks = _gate_checks(
            base=result["primary_headline"],
            stress=result["stress_headline"],
            halves=result["contained_half_headlines"],
            controls=result["control_headlines"],
            adjusted_p=adjusted_p,
        )
        result["romano_wolf_adjusted_p"] = adjusted_p
        result["gates"] = checks
        result["failed_gates"] = [name for name, passed in checks.items() if not passed]
        result["variant_passed"] = all(checks.values())
        if result["variant_passed"]:
            passing.append(variant_id)
    for variant_id in FAMILY_IDS:
        if variant_id not in active:
            variant_results[variant_id] = {
                "evaluated": False,
                "reason": "rejected by prior train stage",
                "romano_wolf_adjusted_p": 1.0,
                "variant_passed": False,
            }

    stage_passed = bool(passing)
    index = STAGE_ORDER.index(stage)
    core = {
        "protocol_version": STAGE_PROTOCOL_VERSION,
        "candidate_family": POLICY_ID,
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_prior_stage_manifest_hashes": {
            report["stage"]: report["manifest_hash"] for report in prior
        },
        "config": asdict(FROZEN_CONFIG),
        "execution_diagnostics": diagnostics,
        "active_variants": list(active),
        "multiple_testing_family": list(FAMILY_IDS),
        "romano_wolf": inference,
        "variant_results": variant_results,
        "passing_variants": passing,
        "rejected_variants": [item for item in FAMILY_IDS if item not in passing],
        "stage_passed": stage_passed,
        "opened_windows": list(STAGE_ORDER[: index + 1]),
        "sealed_windows": list(STAGE_ORDER[index + 1 :]),
        "post_2023_rows_read": 0,
        "disposition": (
            "ADVANCE_TO_SELECTION"
            if stage == "train" and stage_passed
            else "QUALIFIED_FOR_POST_PASS_AUDIT"
            if stage == "selection" and stage_passed
            else "REJECT_NO_REPAIR"
        ),
    }
    return _seal(core)


def _format_metric(value: Any) -> str:
    number = float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.2f}"


def _metric_row(label: str, item: Mapping[str, Any], adjusted_p: Any = "-") -> str:
    p_text = adjusted_p if isinstance(adjusted_p, str) else f"{float(adjusted_p):.4f}"
    return (
        f"| {label} | {_format_metric(item['absolute_return_pct'])}% | "
        f"{_format_metric(item['cagr_pct'])}% | "
        f"{_format_metric(item['strict_mdd_pct'])}% | "
        f"{_format_metric(item['cagr_to_strict_mdd'])} | {item['trades']} | "
        f"{item['longs']}/{item['shorts']} | "
        f"{_format_metric(item['mean_gross_underlying_bp'])}bp | {p_text} |"
    )


def render_stage_doc(report: Mapping[str, Any]) -> str:
    lines = [
        f"# BFMWD-144 {report['stage']} strict result — 2026-07-20",
        "",
        "Absolute return and CAGR use the full declared calendar. Strict MDD uses "
        "the global/pre-entry HWM, costs, exact realized funding and every held "
        "five-minute favorable-then-adverse path.",
        "",
        "| Variant | Absolute | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross | RW p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id in FAMILY_IDS:
        result = report["variant_results"][variant_id]
        if result.get("evaluated") is False:
            lines.append(
                f"| {variant_id} | not advanced | - | - | - | - | - | - | 1.0000 |"
            )
            continue
        lines.append(
            _metric_row(
                variant_id,
                result["primary_headline"],
                result["romano_wolf_adjusted_p"],
            )
        )
    lines.extend(
        [
            "",
            f"- Stage passed: **{report['stage_passed']}**",
            f"- Passing variants: `{report['passing_variants']}`",
            f"- Disposition: `{report['disposition']}`",
            "",
            "## Per-variant gates",
            "",
        ]
    )
    for variant_id in report["active_variants"]:
        result = report["variant_results"][variant_id]
        lines.extend(
            [
                f"### `{variant_id}`",
                "",
                f"- passed: **{result['variant_passed']}**",
                f"- failed gates: `{result['failed_gates']}`",
                _metric_row("10bp stress", result["stress_headline"]),
                _metric_row(
                    "one-bar delay",
                    result["control_headlines"]["extra_latency_one_bar"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Integrity",
            "",
            f"- evaluator SHA-256: `{report['evaluator_source_sha256']}`",
            f"- report manifest: `{report['manifest_hash']}`",
            f"- physical source window: `{report['execution_diagnostics']['physical_window']}`",
            f"- still sealed: `{report['sealed_windows']}`",
            f"- post-2023 rows read: `{report['post_2023_rows_read']}`",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_stage(stage: str) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"BFMWD economic unknown stage: {stage}")
    output = _path(STAGE_OUTPUTS[stage])
    document = _path(STAGE_DOCS[stage])
    if output.exists() or document.exists():
        raise FileExistsError(f"BFMWD economic {stage} result is write-once")
    report = _build_stage_report(stage)
    try:
        _write_once_bytes(output, _json_bytes(report))
        _write_once_bytes(document, render_stage_doc(report).encode())
    except Exception:
        output.unlink(missing_ok=True)
        document.unlink(missing_ok=True)
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--prepare-stage-source", choices=STAGE_ORDER)
    group.add_argument("--stage", choices=STAGE_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze:
        report = freeze_evaluator()
    elif args.prepare_stage_source is not None:
        report = prepare_stage_source(args.prepare_stage_source)
    else:
        report = evaluate_stage(cast(str, args.stage))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
