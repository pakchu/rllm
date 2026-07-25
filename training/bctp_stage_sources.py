"""Sequential write-once annual source isolation for BCTP outcome stages.

The freezer binds the evaluator contract and parent manifests only.  This module
copies exactly one sealed calendar-year stage from the frozen chronological gzip
parents into deterministic stage-local gzip files, then validates the physical
copies.  It deliberately never hashes the full parent market/funding payloads.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Final, Literal

import pandas as pd

from training import freeze_block_clearing_target_position_evaluator as freeze


Stage = Literal["2020", "2021", "2022", "2023"]

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
EVALUATOR_FREEZE: Final = freeze.DEFAULT_OUTPUT
EXPECTED_EVALUATOR_MANIFEST_HASH: Final = (
    "25f80efcef9e8e5b08b9138dc17cb4c18edcc0b47187ad9b43f999acee0afd87"
)
DEFAULT_OUTPUT_ROOT: Final = Path("data/bctp_stage_sources")
STAGE_ORDER: Final = ("2020", "2021", "2022", "2023")
MARKET_COLUMNS: Final = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
FUNDING_COLUMNS: Final = (
    "funding_time_ms",
    "funding_time_utc",
    "symbol",
    "funding_rate",
    "settlement_mark_price",
    "mark_open_time_ms",
    "mark_open_time_utc",
    "funding_time_offset_ms",
    "mark_source",
)
SCHEDULE_MANIFEST_PROTOCOL: Final = "bctp_target_schedule_seal_v1"
SCHEDULE_COLUMNS: Final = (
    "policy_id",
    "sequence_id",
    "entry_time",
    "target",
)
PROMOTABLE_PRIMARY_IDS: Final = (
    "categorical_linear_fqi",
    "categorical_ridge_fqi",
    "extra_trees_fqi",
)


@dataclass(frozen=True)
class StageSpec:
    stage: Stage
    start: pd.Timestamp
    end: pd.Timestamp
    market_rows: int
    funding_rows: int


STAGE_SPECS: Final[dict[str, StageSpec]] = {
    stage: StageSpec(
        stage=stage,  # type: ignore[arg-type]
        start=pd.Timestamp(f"{stage}-01-01T00:00:00Z"),
        end=pd.Timestamp(f"{int(stage) + 1}-01-01T00:00:00Z"),
        market_rows=(
            int(
                (
                    pd.Timestamp(f"{int(stage) + 1}-01-01T00:00:00Z")
                    - pd.Timestamp(f"{stage}-01-01T00:00:00Z")
                ).total_seconds()
                // 300
            )
        ),
        funding_rows=(
            int(
                (
                    pd.Timestamp(f"{int(stage) + 1}-01-01T00:00:00Z")
                    - pd.Timestamp(f"{stage}-01-01T00:00:00Z")
                ).total_seconds()
                // (8 * 3600)
            )
        ),
    )
    for stage in STAGE_ORDER
}


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BCTP expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_spec(stage: str) -> StageSpec:
    if stage not in STAGE_SPECS:
        raise ValueError(f"BCTP unsupported stage: {stage!r}")
    return STAGE_SPECS[stage]


def _iso_z(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def _parse_aware_utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"BCTP {name} must be timezone aware")
    if pd.isna(parsed):
        raise ValueError(f"BCTP {name} is NaT")
    return parsed.tz_convert("UTC")


def _expected_grid(start: pd.Timestamp, periods: int, frequency: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=frequency, tz="UTC")


def verify_evaluator_freeze(
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen BCTP evaluator artifact without parent payload access."""

    path = _path(manifest_path or EVALUATOR_FREEZE)
    payload = _read_json(path)
    freeze.validate_manifest(payload)
    if payload.get("manifest_hash") != EXPECTED_EVALUATOR_MANIFEST_HASH:
        raise ValueError("BCTP evaluator manifest hash binding drifted")
    boundary = payload.get("outcome_boundary", {})
    if boundary.get("market_or_funding_payload_bytes_hashed") is not False:
        raise ValueError("BCTP freeze hashed outcome payload bytes")
    if boundary.get("market_rows_parsed") != 0 or boundary.get("funding_rows_parsed") != 0:
        raise ValueError("BCTP freeze parsed outcome rows")
    for source in payload.get("execution_sources", {}).values():
        if source.get("payload_bytes_hashed_during_freeze") is not False:
            raise ValueError("BCTP execution source hashed payload bytes")
    return payload


def _verify_parent_bindings(
    *,
    market_parent: str | Path,
    funding_parent: str | Path,
    evaluator_manifest_path: str | Path | None,
    allow_synthetic_parents: bool,
) -> dict[str, Any]:
    evaluator = verify_evaluator_freeze(evaluator_manifest_path)
    sources = evaluator["execution_sources"]
    requested = {
        "market": _path(market_parent).resolve(),
        "funding": _path(funding_parent).resolve(),
    }
    frozen = {
        "market": _path(sources["market"]["path"]).resolve(),
        "funding": _path(sources["funding"]["path"]).resolve(),
    }
    custom = requested != frozen
    if custom and not allow_synthetic_parents:
        raise ValueError(
            "BCTP production stage parents must match the frozen sources"
        )
    if allow_synthetic_parents and not custom:
        raise ValueError("BCTP synthetic-parent override was unnecessary")

    market_manifest = _read_json(freeze.MARKET_MANIFEST)
    funding_manifest = _read_json(freeze.FUNDING_MANIFEST)
    if _file_sha256(freeze.MARKET_MANIFEST) != sources["market"]["manifest_sha256"]:
        raise ValueError("BCTP market source manifest file hash drifted")
    if _file_sha256(freeze.FUNDING_MANIFEST) != sources["funding"]["manifest_sha256"]:
        raise ValueError("BCTP funding source manifest file hash drifted")
    if market_manifest.get("combined_output") != str(freeze.MARKET):
        raise ValueError("BCTP market manifest output binding drifted")
    if market_manifest.get("combined_sha256") != sources["market"]["expected_sha256"]:
        raise ValueError("BCTP market manifest payload hash binding drifted")
    funding_data = funding_manifest.get("data", {})
    if funding_data.get("path") != str(freeze.FUNDING):
        raise ValueError("BCTP funding manifest output binding drifted")
    if funding_data.get("sha256") != sources["funding"]["expected_sha256"]:
        raise ValueError("BCTP funding manifest payload hash binding drifted")
    if funding_manifest.get("strategy_outcomes_calculated") not in (False, []):
        raise ValueError("BCTP funding source manifest has outcomes calculated")
    return {
        "evaluator_manifest_hash": evaluator["manifest_hash"],
        "market_source_manifest_canonical_hash": _canonical_hash(market_manifest),
        "funding_source_manifest_canonical_hash": _canonical_hash(funding_manifest),
        "market_source_manifest_file_sha256": sources["market"]["manifest_sha256"],
        "funding_source_manifest_file_sha256": sources["funding"]["manifest_sha256"],
        "market_parent": str(requested["market"]),
        "funding_parent": str(requested["funding"]),
        "synthetic_parent_override": bool(custom),
        "parent_payload_bytes_hashed": False,
    }


def _schedule_frame_hash(frame: pd.DataFrame) -> str:
    records = [
        {
            column: str(row[column])
            for column in SCHEDULE_COLUMNS
        }
        for row in frame.to_dict(orient="records")
    ]
    return freeze.canonical_hash(records)


def _validate_schedule_artifact(
    binding: Any,
    *,
    stage: str,
    expected_policy_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError("BCTP schedule artifact binding is invalid")
    path = _path(str(binding.get("path", "")))
    if not path.is_file():
        raise ValueError("BCTP sealed schedule artifact is missing")
    digest = _file_sha256(path)
    if binding.get("sha256") != digest:
        raise ValueError("BCTP sealed schedule artifact hash mismatch")
    frame = pd.read_csv(path, compression="infer", dtype=str)
    if tuple(frame.columns) != SCHEDULE_COLUMNS:
        raise ValueError("BCTP sealed schedule schema mismatch")
    if len(frame) != int(binding.get("rows", -1)):
        raise ValueError("BCTP sealed schedule row count mismatch")
    if binding.get("columns") != list(SCHEDULE_COLUMNS):
        raise ValueError("BCTP sealed schedule columns binding mismatch")
    if frame.empty:
        raise ValueError("BCTP sealed schedule artifact is empty")
    observed_policy_order = tuple(dict.fromkeys(frame["policy_id"].tolist()))
    if observed_policy_order != expected_policy_ids:
        raise ValueError("BCTP sealed schedule policy order mismatch")
    if frame.duplicated(["policy_id", "sequence_id"]).any():
        raise ValueError("BCTP sealed schedule identities are duplicated")
    if not frame["target"].isin(freeze.MODEL_ACTION_NAMES).all():
        raise ValueError("BCTP sealed schedule target changed")
    spec = _stage_spec(stage)
    terminal = spec.end - pd.Timedelta(minutes=5)
    for policy_id in expected_policy_ids:
        policy_rows = frame.loc[frame["policy_id"].eq(policy_id)]
        times = [
            _parse_aware_utc_timestamp(value, name="schedule timestamp")
            for value in policy_rows["entry_time"]
        ]
        index = pd.DatetimeIndex(times)
        if (
            index.has_duplicates
            or not index.is_monotonic_increasing
            or not ((index >= spec.start) & (index < terminal)).all()
            or any(
                (timestamp - spec.start) % pd.Timedelta(minutes=5)
                != pd.Timedelta(0)
                for timestamp in index
            )
        ):
            raise ValueError("BCTP sealed schedule timestamps changed")
    frame_hash = _schedule_frame_hash(frame)
    if binding.get("frame_hash") != frame_hash:
        raise ValueError("BCTP sealed schedule frame hash mismatch")
    return {
        "path": str(path),
        "sha256": digest,
        "frame_hash": frame_hash,
        "rows": len(frame),
        "columns": list(SCHEDULE_COLUMNS),
    }


def _validate_required_schedule(
    stage: str,
    required_schedule_manifest: str | Path | None,
) -> dict[str, Any] | None:
    if stage == "2020":
        return None
    if required_schedule_manifest is None:
        raise RuntimeError("BCTP post-2020 stages require a pre-existing target-schedule manifest")
    path = _path(required_schedule_manifest)
    if not path.exists():
        raise RuntimeError("BCTP target-schedule manifest is missing")
    payload = _read_json(path)
    if payload.get("protocol_version") != SCHEDULE_MANIFEST_PROTOCOL:
        raise ValueError("BCTP target-schedule protocol mismatch")
    if payload.get("strategy_outcomes_calculated") is not False:
        raise ValueError("BCTP target-schedule manifest contains outcomes")
    if payload.get("outcome_payload_opened") is not False:
        raise ValueError("BCTP target-schedule manifest opened outcomes")
    if (
        payload.get("target_stage") != stage
        or payload.get("stage") != stage
    ):
        raise ValueError("BCTP target-schedule manifest stage mismatch")
    if payload.get("market_or_funding_payload_opened") is not False:
        raise ValueError("BCTP target-schedule manifest opened payloads")
    if (
        payload.get("market_or_funding_payload_bytes_hashed")
        is not False
    ):
        raise ValueError("BCTP target-schedule manifest hashed payloads")
    if payload.get("evaluator_manifest_hash") != EXPECTED_EVALUATOR_MANIFEST_HASH:
        raise ValueError("BCTP target-schedule evaluator binding mismatch")
    if payload.get("family_ids") != list(freeze.FAMILY_IDS):
        raise ValueError("BCTP target-schedule family changed")
    if payload.get("promotable_primary_ids") != list(PROMOTABLE_PRIMARY_IDS):
        raise ValueError("BCTP target-schedule primary family changed")
    if payload.get("stress_reuses_base_target_sequences") is not True:
        raise ValueError("BCTP target-schedule stress binding changed")
    without_hash = dict(payload)
    embedded_hash = without_hash.pop("manifest_hash", None)
    if embedded_hash != _canonical_hash(without_hash):
        raise ValueError("BCTP target-schedule manifest hash mismatch")
    base = _validate_schedule_artifact(
        payload.get("base_schedules"),
        stage=stage,
        expected_policy_ids=tuple(freeze.FAMILY_IDS),
    )
    delayed = _validate_schedule_artifact(
        payload.get("delayed_primary_schedules"),
        stage=stage,
        expected_policy_ids=PROMOTABLE_PRIMARY_IDS,
    )
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "canonical_hash": _canonical_hash(payload),
        "manifest_hash": embedded_hash,
        "base_schedules": base,
        "delayed_primary_schedules": delayed,
    }


def _verify_bound_schedule(stage: str, binding: Any) -> None:
    if stage == "2020":
        if binding is not None:
            raise ValueError("BCTP fit stage must not bind a target schedule")
        return
    if not isinstance(binding, dict) or "path" not in binding:
        raise ValueError("BCTP stage schedule binding is missing")
    current = _validate_required_schedule(stage, binding["path"])
    if current != binding:
        raise ValueError("BCTP stage schedule binding drifted")


def _stage_paths(output_root: str | Path, stage: str) -> dict[str, Path]:
    root = _path(output_root) / str(stage)
    return _paths_for_root(root, stage)


def _paths_for_root(root: Path, stage: str) -> dict[str, Path]:
    return {
        "root": root,
        "market": root / f"bctp_market_{stage}.csv.gz",
        "funding": root / f"bctp_funding_{stage}.csv.gz",
        "manifest": root / "source_manifest.json",
    }


def _assert_no_orphans(paths: dict[str, Path]) -> None:
    existing = {name for name in ("market", "funding", "manifest") if paths[name].exists()}
    if paths["root"].exists() and existing != {
        "market",
        "funding",
        "manifest",
    }:
        raise RuntimeError(f"BCTP orphaned stage source files: {sorted(existing)}")


def _gzip_payload_sha256(path: Path) -> str:
    return _file_sha256(path)


def _decoded_gzip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gzip_atomic(path: Path, text: str) -> dict[str, str]:
    encoded = text.encode("utf-8")
    decoded_digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as raw:
        tmp = Path(raw.name)
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(encoded)
    gzip_digest = _file_sha256(tmp)
    if path.exists():
        if (
            _file_sha256(path) != gzip_digest
            or _decoded_gzip_sha256(path) != decoded_digest
        ):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"BCTP write-once stage source drift: {path}")
        tmp.unlink(missing_ok=True)
        return {
            "gzip_sha256": gzip_digest,
            "decoded_lines_sha256": decoded_digest,
        }
    tmp.replace(path)
    return {
        "gzip_sha256": gzip_digest,
        "decoded_lines_sha256": decoded_digest,
    }


def _write_json_once(path: Path, payload: dict[str, Any]) -> str:
    encoded = _canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"BCTP write-once manifest drift: {path}")
        return hashlib.sha256(encoded).hexdigest()
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
        tmp = Path(handle.name)
    tmp.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def _timestamp_from_csv_line(line: str, timestamp_index: int) -> pd.Timestamp:
    # Timestamp-only prior scanning: split just enough to select the declared
    # timestamp field, but do not coerce or inspect any numeric outcome fields.
    fields = next(csv.reader([line.rstrip("\n\r")]))
    if timestamp_index >= len(fields):
        raise ValueError("BCTP source row is missing timestamp field")
    value = fields[timestamp_index].strip()
    if not value:
        raise ValueError("BCTP source row has empty timestamp")
    return _parse_utc_timestamp(value)


def _stream_stage_copy(
    *,
    parent: str | Path,
    output: Path,
    start: pd.Timestamp,
    expected_rows: int,
    expected_header: tuple[str, ...],
    timestamp_column: str,
) -> dict[str, Any]:
    rows = 0
    prior_timestamp_only_rows = 0
    lines: list[str] = []
    with gzip.open(_path(parent), "rt", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("BCTP source is empty")
        header = tuple(next(csv.reader([header_line.rstrip("\n\r")])) )
        if header != expected_header:
            raise ValueError("BCTP parent schema mismatch")
        timestamp_index = header.index(timestamp_column)
        lines.append(",".join(header) + "\n")
        for line in handle:
            ts = _timestamp_from_csv_line(line, timestamp_index)
            if ts < start:
                prior_timestamp_only_rows += 1
                continue
            lines.append(line if line.endswith("\n") else line + "\n")
            rows += 1
            if rows == expected_rows:
                break
    if rows != expected_rows:
        raise ValueError(f"BCTP expected {expected_rows} rows, copied {rows}")
    hashes = _write_gzip_atomic(output, "".join(lines))
    return {
        "path": str(output),
        **hashes,
        "rows_copied": rows,
        "prior_rows_timestamp_only": prior_timestamp_only_rows,
        "post_stage_numeric_rows_parsed": 0,
        "stopped_after_expected_count_without_first_future_row": True,
    }


def _validate_market_frame(df: pd.DataFrame, spec: StageSpec) -> None:
    if tuple(df.columns) != MARKET_COLUMNS:
        raise ValueError("BCTP market schema mismatch")
    if len(df) != spec.market_rows:
        raise ValueError("BCTP market row count mismatch")
    dates = pd.to_datetime(df["date"], utc=True, errors="raise")
    expected = _expected_grid(spec.start, spec.market_rows, "5min")
    if not (dates.to_numpy() == expected.to_numpy()).all():
        raise ValueError("BCTP market timestamp grid mismatch")
    numeric_columns = [column for column in MARKET_COLUMNS if column != "date"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.apply(lambda col: col.map(math.isfinite)).all().all():
        raise ValueError("BCTP market numeric values are invalid")
    if (numeric[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("BCTP market OHLC must be positive")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("BCTP market high violates OHLC")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("BCTP market low violates OHLC")
    nonnegative = ["volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"]
    if (numeric[nonnegative] < 0.0).any().any():
        raise ValueError("BCTP market nonnegative fields are invalid")


def _validate_funding_frame(df: pd.DataFrame, spec: StageSpec) -> None:
    if tuple(df.columns) != FUNDING_COLUMNS:
        raise ValueError("BCTP funding schema mismatch")
    if len(df) != spec.funding_rows:
        raise ValueError("BCTP funding row count mismatch")
    times = pd.to_datetime(
        df["funding_time_utc"],
        utc=True,
        errors="raise",
    )
    mark_times = pd.to_datetime(
        df["mark_open_time_utc"],
        utc=True,
        errors="raise",
    )
    expected_marks = _expected_grid(
        spec.start,
        spec.funding_rows,
        "8h",
    )
    if not (mark_times.to_numpy() == expected_marks.to_numpy()).all():
        raise ValueError("BCTP funding mark timestamp grid mismatch")
    funding_ms = pd.to_numeric(
        df["funding_time_ms"],
        errors="coerce",
    )
    mark_ms = pd.to_numeric(
        df["mark_open_time_ms"],
        errors="coerce",
    )
    expected_mark_ms = (
        expected_marks.view("int64") // 1_000_000
    ).astype("int64")
    if not (
        mark_ms.astype("int64").to_numpy() == expected_mark_ms
    ).all():
        raise ValueError("BCTP funding mark millisecond grid mismatch")
    parsed_funding_ms = (
        pd.DatetimeIndex(times).view("int64") // 1_000_000
    ).astype("int64")
    if not (
        funding_ms.astype("int64").to_numpy() == parsed_funding_ms
    ).all():
        raise ValueError("BCTP funding timestamp/millisecond mismatch")
    if not (df["symbol"].astype(str) == "BTCUSDT").all():
        raise ValueError("BCTP funding symbol mismatch")
    numeric_columns = [
        "funding_rate",
        "settlement_mark_price",
        "mark_open_time_ms",
        "funding_time_offset_ms",
    ]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.apply(lambda col: col.map(math.isfinite)).all().all():
        raise ValueError("BCTP funding numeric values are invalid")
    if (numeric["settlement_mark_price"] <= 0.0).any():
        raise ValueError("BCTP funding settlement marks must be positive")
    offsets = numeric["funding_time_offset_ms"].astype("int64")
    if (offsets < 0).any() or (offsets > 60_000).any():
        raise ValueError("BCTP funding timestamp offset exceeds contract")
    if not (
        offsets.to_numpy()
        == funding_ms.astype("int64").to_numpy()
        - mark_ms.astype("int64").to_numpy()
    ).all():
        raise ValueError("BCTP funding timestamp offset mismatch")
    if not (
        times.to_numpy()
        == (
            expected_marks
            + pd.to_timedelta(offsets.to_numpy(), unit="ms")
        ).to_numpy()
    ).all():
        raise ValueError("BCTP funding timestamp offset grid mismatch")


def _read_and_validate_stage(paths: dict[str, Path], spec: StageSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market = pd.read_csv(paths["market"])
    funding = pd.read_csv(paths["funding"])
    _validate_market_frame(market, spec)
    _validate_funding_frame(funding, spec)
    market["date"] = pd.to_datetime(
        market["date"],
        utc=True,
        errors="raise",
    )
    funding["funding_time_utc"] = pd.to_datetime(
        funding["funding_time_utc"],
        utc=True,
        errors="raise",
    )
    funding["mark_open_time_utc"] = pd.to_datetime(
        funding["mark_open_time_utc"],
        utc=True,
        errors="raise",
    )
    diagnostics = {
        "stage": spec.stage,
        "start": _iso_z(spec.start),
        "end": _iso_z(spec.end),
        "market_rows": len(market),
        "funding_rows": len(funding),
        "market_grid": "5min",
        "funding_grid": "8h",
    }
    return market, funding, diagnostics


def prepare_stage_source(
    stage: Stage,
    *,
    market_parent: str | Path = freeze.MARKET,
    funding_parent: str | Path = freeze.FUNDING,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path | None = None,
    required_schedule_manifest: str | Path | None = None,
    allow_synthetic_parents: bool = False,
) -> dict[str, Any]:
    """Copy exactly one BCTP calendar-year source stage, write-once.

    Parent payload bytes are streamed only until the exact expected in-stage
    row count is copied; the first future row is not read to discover the end.
    """

    spec = _stage_spec(stage)
    schedule_binding = _validate_required_schedule(stage, required_schedule_manifest)
    source_bindings = _verify_parent_bindings(
        market_parent=market_parent,
        funding_parent=funding_parent,
        evaluator_manifest_path=manifest_path,
        allow_synthetic_parents=allow_synthetic_parents,
    )
    paths = _stage_paths(output_root, stage)
    _assert_no_orphans(paths)
    if paths["manifest"].exists():
        existing = _read_json(paths["manifest"])
        if existing.get("stage") != stage:
            raise RuntimeError("BCTP existing manifest stage mismatch")
        if existing.get("target_schedule_binding") != schedule_binding:
            raise RuntimeError("BCTP existing stage schedule binding drifted")
        load_stage_source(stage, output_root=output_root, manifest_path=manifest_path)
        return existing

    paths["root"].parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            dir=paths["root"].parent,
            prefix=f".bctp-{stage}-",
        )
    )
    temporary_paths = _paths_for_root(temporary_root, stage)
    try:
        market_copy = _stream_stage_copy(
            parent=market_parent,
            output=temporary_paths["market"],
            start=spec.start,
            expected_rows=spec.market_rows,
            expected_header=MARKET_COLUMNS,
            timestamp_column="date",
        )
        funding_copy = _stream_stage_copy(
            parent=funding_parent,
            output=temporary_paths["funding"],
            start=spec.start,
            expected_rows=spec.funding_rows,
            expected_header=FUNDING_COLUMNS,
            timestamp_column="funding_time_utc",
        )
        market_copy["path"] = str(paths["market"])
        funding_copy["path"] = str(paths["funding"])
        _, _, validation = _read_and_validate_stage(
            temporary_paths,
            spec,
        )
        manifest_core: dict[str, Any] = {
            "protocol_version": (
                "bctp_sequential_stage_source_isolation_v1"
            ),
            "stage": stage,
            "stage_order": list(STAGE_ORDER),
            "start_inclusive": _iso_z(spec.start),
            "end_exclusive": _iso_z(spec.end),
            "expected_market_rows_5m": spec.market_rows,
            "expected_funding_rows_8h": spec.funding_rows,
            "market": market_copy,
            "funding": funding_copy,
            "source_bindings": source_bindings,
            "target_schedule_binding": schedule_binding,
            "strategy_outcomes_calculated": False,
            "post_stage_numeric_rows_parsed": 0,
            "market_or_funding_parent_payload_bytes_hashed": False,
            "validation": validation,
        }
        manifest = {
            **manifest_core,
            "manifest_hash": _canonical_hash(manifest_core),
        }
        _write_json_once(temporary_paths["manifest"], manifest)
        if paths["root"].exists():
            raise RuntimeError("BCTP stage destination appeared during publish")
        os.rename(temporary_root, paths["root"])
        return manifest
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def load_stage_source(
    stage: Stage,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load a prepared stage copy after re-verifying the frozen contract."""

    spec = _stage_spec(stage)
    verify_evaluator_freeze(manifest_path)
    paths = _stage_paths(output_root, stage)
    if not paths["manifest"].exists():
        raise FileNotFoundError(f"BCTP stage manifest missing: {paths['manifest']}")
    manifest = _read_json(paths["manifest"])
    embedded = dict(manifest)
    manifest_hash = embedded.pop("manifest_hash", None)
    if manifest_hash != _canonical_hash(embedded):
        raise ValueError("BCTP stage manifest hash mismatch")
    if manifest.get("stage") != stage:
        raise ValueError("BCTP stage manifest stage mismatch")
    if manifest.get("strategy_outcomes_calculated") is not False:
        raise ValueError("BCTP stage source manifest contains outcomes")
    if manifest.get("source_bindings", {}).get(
        "evaluator_manifest_hash"
    ) != EXPECTED_EVALUATOR_MANIFEST_HASH:
        raise ValueError("BCTP stage evaluator binding drifted")
    if manifest.get("market_or_funding_parent_payload_bytes_hashed") is not False:
        raise ValueError("BCTP stage source hashed parent payload")
    _verify_bound_schedule(
        stage,
        manifest.get("target_schedule_binding"),
    )
    if not paths["market"].exists() or not paths["funding"].exists():
        raise RuntimeError("BCTP stage source orphan manifest")
    if _gzip_payload_sha256(paths["market"]) != manifest["market"][
        "gzip_sha256"
    ]:
        raise ValueError("BCTP market stage gzip drift")
    if _decoded_gzip_sha256(paths["market"]) != manifest["market"][
        "decoded_lines_sha256"
    ]:
        raise ValueError("BCTP market decoded lines drift")
    if _gzip_payload_sha256(paths["funding"]) != manifest["funding"][
        "gzip_sha256"
    ]:
        raise ValueError("BCTP funding stage gzip drift")
    if _decoded_gzip_sha256(paths["funding"]) != manifest["funding"][
        "decoded_lines_sha256"
    ]:
        raise ValueError("BCTP funding decoded lines drift")
    market, funding, diagnostics = _read_and_validate_stage(paths, spec)
    diagnostics = {
        **diagnostics,
        "manifest_hash": manifest_hash,
        "strategy_outcomes_calculated": False,
        "post_stage_numeric_rows_parsed": 0,
        "parent_payload_bytes_hashed": False,
    }
    return market, funding, diagnostics
