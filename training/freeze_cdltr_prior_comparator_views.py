"""Freeze complete sanitized prior clocks for CDLTR-72A novelty checks.

The output contains every preregistered comparator identity and only its
capability plus decision/entry/exit timestamps, side, and source-clock label.
It never reads CDLTR sources, replays the chain strategy, or computes/retains a
prior return, PnL, equity path, CAGR, MDD, or forecast.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, cast

import pandas as pd


PROTOCOL_VERSION = "cdltr_prior_comparator_views_v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path("training/freeze_cdltr_prior_comparator_views.py")
AMENDMENT = Path("docs/cdltr72a-preincidence-comparator-amendment-2026-07-21.md")
DEFAULT_CLOCK = Path("results/cdltr_prior_comparator_views_2026-07-21.csv.gz")
DEFAULT_MANIFEST = Path("results/cdltr_prior_comparator_views_manifest_2026-07-21.json")

ORFR_CLOCK = Path(
    "results/overnight_rrp_flow_release_preregistered_clock_2026-07-17.csv.gz"
)
ORFR_SHA256 = "9f09bc88c9661441a33cee724e59524f57c0b021abff0fe81263e1a341b7b7b7"
ORFR_HEADER = (
    "operation_date",
    "decision_time",
    "entry_time",
    "scheduled_exit_time",
    "side",
    "clock_mode",
    "log_amount",
    "innovation",
    "innovation_rank",
)

CVTR_CLOCK = Path(
    "results/cboe_volatility_term_rotation_preregistered_clock_2026-07-17.csv.gz"
)
CVTR_SHA256 = "c0250d1f40c87049f6d7639ba43f5285835441399a62968434b65c7d46ed2a93"
CVTR_HEADER = (
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "clock_mode",
    "front_slope",
    "broad_slope",
    "front_rank",
    "broad_rank",
    "vix_level_rank",
    "score",
)

NTB_CLOCK = Path("results/network_topology_broadening_clock_2026-07-17.csv")
NTB_SHA256 = "6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f"
NTB_HEADER = (
    "policy_id",
    "side",
    "observation_date",
    "available_at",
    "earliest_tradable_open",
    "entry_date",
    "exit_date",
    "fanout",
    "breadth",
    "fanout_change",
    "breadth_change",
    "fanout_z",
    "breadth_z",
    "composite",
    "source_lag_days",
    "fanout_reference_count",
    "breadth_reference_count",
)

NWE7_CLOCK = Path("results/network_weak_signal_ensemble_feature_clock_2026-07-17.csv")
NWE7_SHA256 = "5e0fc6b99a3fefe5b13a3f6ad66cd40cde6fb4423e4d845a0ee5854496e1af67"
NWE7_HEADER = (
    "policy_id",
    "decision_date",
    "entry_date",
    "exit_date",
    "source_observation_date",
    "feature_available_at",
    "observation_age_days",
    "minimum_reference_count",
    "all_features_finite",
    "prediction_eligible",
    "fee_share_level_z",
    "fee_share_change_z",
    "transaction_density_level_z",
    "transaction_density_change_z",
    "breadth_level_z",
    "breadth_change_z",
    "fanout_level_z",
    "fanout_change_z",
)

NWE8_SELECTION = Path(
    "results/network_weak_signal_ensemble_v2_selection_2026-07-17.json"
)
NWE8_SHA256 = "64816351ace7af10fd78147018953d1cdda5b25c4dd4c451bfe448cb9b8aca1c"
NWE8_KEYS = {
    "decision_date",
    "entry_date",
    "exit_date",
    "side",
    "forecast",
    "abstain_threshold",
    "train_count",
    "latest_train_exit",
}

CHAIN_CLOCK = Path(
    "results/chain_activity_impulse_momentum_pre2024_comparator_clock_2026-07-21.csv.gz"
)
CHAIN_CLOCK_SHA256 = "e50cc154e23950a381aa456180970140882083734128bd7f902257738633f320"
CHAIN_HEADER = ("window", "decision_time", "entry_time", "exit_time", "side")
CHAIN_MANIFEST = Path(
    "results/chain_activity_impulse_momentum_pre2024_comparator_clock_manifest_2026-07-21.json"
)
CHAIN_MANIFEST_SHA256 = (
    "899704a0e998d818fd09735ca90af3c82aecfce94a288eec2bbc77c0c3df8441"
)

FLCC_CLOCK = Path(
    "results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz"
)
FLCC_SHA256 = "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c"
FLCC_HEADER = (
    "candidate_id",
    "clock_name",
    "feature_release_date",
    "signal_release_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "horizon_releases",
    "lower_rank_numerator",
    "upper_rank_numerator",
    "prior_lookback",
    "net_rank_numerator",
    "asset_rank_numerator",
    "tga_release_rank_numerator",
    "rrp_release_rank_numerator",
    "component_breadth",
    "component_tail_breadth",
)
FLCC_CANDIDATES = {
    "FLCC-H4-Q60",
    "FLCC-H4-Q65",
    "FLCC-H8-Q60",
    "FLCC-H8-Q65",
}

DFFB_CLOCK = Path(
    "results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz"
)
DFFB_SHA256 = "df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978"
DFFB_HEADER = (
    "policy_id",
    "clock",
    "window",
    "signal_record_date",
    "execution_record_date",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    "deposit_breadth",
    "withdrawal_breadth",
    "issue_breadth",
    "redemption_breadth",
    "deposit_eligible_categories",
    "withdrawal_eligible_categories",
    "issue_eligible_categories",
    "redemption_eligible_categories",
    "cash_impulse",
    "debt_impulse",
    "cash_rank126",
    "debt_rank126",
    "total_net_cash",
    "total_net_cash_rank126",
)

LIVE_ANCHOR = Path(
    "results/cross_collateral_basis_snapback_live_anchor_clock_2023.json"
)
LIVE_ANCHOR_SHA256 = "0d837e22f2f9c237baf8264332b424e707f73ef92c2169decf8b826442681f2f"
LIVE_KEYS = {"entry_time", "side", "signal_time", "sleeve", "split"}

MICROSTRUCTURE_BUNDLE = Path(
    "results/prior_microstructure_comparator_clock_bundle_2026-07-20.json"
)
MICROSTRUCTURE_SHA256 = (
    "c5584256140799b380973f9f376e5751ad754a81c9683473467b9d05af0bb9f0"
)
MICROSTRUCTURE_COUNTS = {
    "cbfr72": 144,
    "mfic_fast": 1_566,
    "mfic_slow": 1_635,
    "mfic_union": 3_019,
    "netf_fast": 319,
    "netf_slow": 267,
    "netf_union": 586,
    "terminal_absorption_wait72_h72": 100,
    "wfrs_l288_q90_h144": 278,
}

CLOCK_COLUMNS = (
    "comparator",
    "capability",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "source_clock",
)
DIRECTIONAL = "directional_interval"
TIMESTAMP_ONLY = "timestamp_only"
EXPECTED_DIRECTIONAL_COMPARATORS = {
    "ORFR-1",
    "CVTR-1",
    "NTB-7",
    "NWE-8",
    "chain_activity_impulse_momentum",
    *(f"FLCC-1:{name}" for name in FLCC_CANDIDATES),
    "DFFB-601",
}
EXPECTED_TIMESTAMP_COMPARATORS = {
    "NWE-7",
    "live_anchor_2023",
    *(f"prior_microstructure:{name}" for name in MICROSTRUCTURE_COUNTS),
}


@dataclass(frozen=True)
class Config:
    output_clock: str = str(DEFAULT_CLOCK)
    output_manifest: str = str(DEFAULT_MANIFEST)


def _repository_path(path: str | Path) -> Path:
    raw = str(path)
    candidate = Path(path)
    if raw.startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("CDLTR path must be repository-relative")
    return REPOSITORY_ROOT / candidate


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_path(_repository_path(path))


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: str | Path, expected_sha: str, label: str) -> Path:
    target = _repository_path(path)
    if target.is_symlink():
        raise RuntimeError(f"{label} is a symlink")
    if not target.is_file():
        raise RuntimeError(f"{label} is missing")
    if _sha256_path(target) != expected_sha:
        raise RuntimeError(f"{label} SHA drift")
    return target


def _stream_top_level_json_value(
    path: Path, key: str, *, maximum_value_bytes: int = 64 << 20
) -> Any:
    """Decode one exact top-level value and stop before later siblings."""
    marker = f"  {json.dumps(key, ensure_ascii=False)}:"
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.startswith(marker):
                continue
            buffer = line[len(marker) :].lstrip()
            scan_position = 0
            value_start: int | None = None
            depth = 0
            in_string = False
            escaped = False
            while True:
                for position in range(scan_position, len(buffer)):
                    character = buffer[position]
                    if value_start is None:
                        if character.isspace():
                            continue
                        if character not in "[{":
                            raise RuntimeError(
                                f"JSON top-level value is not a container: {key}"
                            )
                        value_start = position
                    if in_string:
                        if escaped:
                            escaped = False
                        elif character == "\\":
                            escaped = True
                        elif character == '"':
                            in_string = False
                        continue
                    if character == '"':
                        in_string = True
                    elif character in "[{":
                        depth += 1
                    elif character in "]}":
                        depth -= 1
                        if depth < 0:
                            raise RuntimeError(
                                f"JSON top-level container is invalid: {key}"
                            )
                        if depth == 0:
                            tail = buffer[position + 1 :].lstrip()
                            if not tail:
                                continuation = handle.readline()
                                if not continuation:
                                    raise RuntimeError(
                                        f"JSON top-level boundary is missing: {key}"
                                    )
                                tail = continuation.lstrip()
                                if not tail or tail[0] not in ",}":
                                    raise RuntimeError(
                                        f"JSON top-level boundary is invalid: {key}"
                                    )
                            if tail[0] not in ",}":
                                raise RuntimeError(
                                    f"JSON top-level boundary is invalid: {key}"
                                )
                            value_text = buffer[value_start : position + 1]
                            return json.loads(value_text)
                else:
                    scan_position = len(buffer)
                    continuation = handle.readline()
                    if not continuation:
                        raise RuntimeError(f"JSON top-level value is truncated: {key}")
                    buffer += continuation
                if len(buffer.encode("utf-8")) > maximum_value_bytes:
                    raise RuntimeError(f"JSON top-level value is too large: {key}")
    raise RuntimeError(f"JSON top-level key is missing: {key}")


def _read_header(path: Path) -> tuple[str, ...]:
    opener: Callable[..., Any] = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle = opener(path, "rt", encoding="utf-8", newline="")
    else:
        handle = opener(path, "rt", encoding="utf-8", newline="")
    with handle:
        return tuple(next(csv.reader(handle)))


def _read_columns(path: Path, columns: set[str]) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        pd.read_csv(path, usecols=lambda column: column in columns),
    )


def _iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if bool(pd.isna(timestamp)):
        raise RuntimeError("comparator timestamp is missing")
    timestamp = cast(pd.Timestamp, timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _side(value: Any) -> int:
    normalized = str(value).strip().upper()
    if normalized == "LONG":
        return 1
    if normalized == "SHORT":
        return -1
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("comparator side is invalid") from error
    if integer not in (-1, 1):
        raise RuntimeError("comparator side is invalid")
    return integer


def _timestamp_row(
    comparator: str, decision: Any, entry: Any, source_clock: str
) -> dict[str, Any]:
    decision_iso = _iso(decision)
    entry_iso = _iso(entry)
    if (
        cast(pd.Timestamp, pd.Timestamp(decision_iso)).value
        > cast(pd.Timestamp, pd.Timestamp(entry_iso)).value
    ):
        raise RuntimeError(f"{comparator} timestamp order is invalid")
    return {
        "comparator": comparator,
        "capability": TIMESTAMP_ONLY,
        "decision_time": decision_iso,
        "entry_time": entry_iso,
        "exit_time": "",
        "side": "",
        "source_clock": source_clock,
    }


def _directional_row(
    comparator: str,
    decision: Any,
    entry: Any,
    exit_time: Any,
    side: Any,
    source_clock: str,
) -> dict[str, Any]:
    decision_iso = _iso(decision)
    entry_iso = _iso(entry)
    exit_iso = _iso(exit_time)
    decision_ns = cast(pd.Timestamp, pd.Timestamp(decision_iso)).value
    entry_ns = cast(pd.Timestamp, pd.Timestamp(entry_iso)).value
    exit_ns = cast(pd.Timestamp, pd.Timestamp(exit_iso)).value
    if not decision_ns <= entry_ns < exit_ns:
        raise RuntimeError(f"{comparator} directional interval is invalid")
    return {
        "comparator": comparator,
        "capability": DIRECTIONAL,
        "decision_time": decision_iso,
        "entry_time": entry_iso,
        "exit_time": exit_iso,
        "side": _side(side),
        "source_clock": source_clock,
    }


def _require_header(path: Path, expected: tuple[str, ...], label: str) -> None:
    if _read_header(path) != expected:
        raise RuntimeError(f"{label} header drift")


def _orfr_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, ORFR_HEADER, "ORFR-1")
    frame = _read_columns(
        path,
        {"decision_time", "entry_time", "scheduled_exit_time", "side", "clock_mode"},
    )
    frame = frame.loc[frame["clock_mode"].eq("primary")]
    if len(frame) != 328:
        raise RuntimeError("ORFR-1 primary count drift")
    return [
        _directional_row(
            "ORFR-1",
            row.decision_time,
            row.entry_time,
            row.scheduled_exit_time,
            row.side,
            "ORFR-1:primary",
        )
        for row in frame.itertuples(index=False)
    ]


def _cvtr_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, CVTR_HEADER, "CVTR-1")
    frame = _read_columns(
        path, {"signal_time", "entry_time", "exit_time", "side", "clock_mode"}
    )
    frame = frame.loc[frame["clock_mode"].eq("primary")]
    if len(frame) != 661:
        raise RuntimeError("CVTR-1 primary count drift")
    return [
        _directional_row(
            "CVTR-1",
            row.signal_time,
            row.entry_time,
            row.exit_time,
            row.side,
            "CVTR-1:primary",
        )
        for row in frame.itertuples(index=False)
    ]


def _ntb_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, NTB_HEADER, "NTB-7")
    frame = _read_columns(
        path, {"policy_id", "available_at", "entry_date", "exit_date", "side"}
    )
    frame = frame.loc[frame["policy_id"].eq("NTB-7")]
    if len(frame) != 41:
        raise RuntimeError("NTB-7 count drift")
    return [
        _directional_row(
            "NTB-7",
            row.available_at,
            row.entry_date,
            row.exit_date,
            row.side,
            "NTB-7",
        )
        for row in frame.itertuples(index=False)
    ]


def _strict_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError("boolean comparator field is invalid")


def _nwe7_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, NWE7_HEADER, "NWE-7")
    frame = _read_columns(
        path, {"policy_id", "decision_date", "entry_date", "prediction_eligible"}
    )
    eligible = cast(pd.Series, frame["prediction_eligible"]).map(_strict_bool)
    frame = frame.loc[frame["policy_id"].eq("NWE-7") & eligible]
    if len(frame) != 147:
        raise RuntimeError("NWE-7 eligible count drift")
    return [
        _timestamp_row("NWE-7", row.decision_date, row.entry_date, "NWE-7")
        for row in frame.itertuples(index=False)
    ]


def _nwe8_rows(path: Path) -> list[dict[str, Any]]:
    schedule = _stream_top_level_json_value(path, "primary_schedule")
    if not isinstance(schedule, list) or len(schedule) != 81:
        raise RuntimeError("NWE-8 primary schedule count drift")
    output: list[dict[str, Any]] = []
    for row in schedule:
        if not isinstance(row, dict) or set(row) != NWE8_KEYS:
            raise RuntimeError("NWE-8 primary schedule schema drift")
        output.append(
            _directional_row(
                "NWE-8",
                row["decision_date"],
                row["entry_date"],
                row["exit_date"],
                row["side"],
                "NWE-8:primary_schedule",
            )
        )
    return output


def _chain_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, CHAIN_HEADER, "chain comparator")
    frame = _read_columns(path, set(CHAIN_HEADER))
    if len(frame) != 66 or set(frame["window"]) != {
        "fit_2021",
        "fit_2022",
        "select_2023",
    }:
        raise RuntimeError("chain comparator clock drift")
    output: list[dict[str, Any]] = []
    ordered = cast(pd.DataFrame, frame.loc[:, list(CHAIN_HEADER)])
    for window, decision, entry, exit_time, side in ordered.itertuples(
        index=False, name=None
    ):
        output.append(
            _directional_row(
                "chain_activity_impulse_momentum",
                decision,
                entry,
                exit_time,
                side,
                f"chain_activity_impulse_momentum:{window}",
            )
        )
    return output


def _flcc_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, FLCC_HEADER, "FLCC-1")
    frame = _read_columns(
        path,
        {
            "candidate_id",
            "clock_name",
            "signal_time",
            "entry_time",
            "exit_time",
            "side",
        },
    )
    frame = frame.loc[frame["clock_name"].eq("primary")]
    if len(frame) != 499 or set(frame["candidate_id"]) != FLCC_CANDIDATES:
        raise RuntimeError("FLCC-1 primary candidate drift")
    expected_counts = {
        "FLCC-H4-Q60": 136,
        "FLCC-H4-Q65": 122,
        "FLCC-H8-Q60": 125,
        "FLCC-H8-Q65": 116,
    }
    if (
        cast(pd.Series, frame["candidate_id"]).value_counts().to_dict()
        != expected_counts
    ):
        raise RuntimeError("FLCC-1 primary count drift")
    return [
        _directional_row(
            f"FLCC-1:{row.candidate_id}",
            row.signal_time,
            row.entry_time,
            row.exit_time,
            row.side,
            f"FLCC-1:{row.candidate_id}:primary",
        )
        for row in frame.itertuples(index=False)
    ]


def _dffb_rows(path: Path) -> list[dict[str, Any]]:
    _require_header(path, DFFB_HEADER, "DFFB-601")
    frame = _read_columns(
        path,
        {
            "policy_id",
            "clock",
            "decision_time_utc",
            "entry_time_utc",
            "exit_time_utc",
            "side",
        },
    )
    frame = frame.loc[frame["policy_id"].eq("DFFB-601") & frame["clock"].eq("primary")]
    if len(frame) != 112:
        raise RuntimeError("DFFB-601 primary count drift")
    return [
        _directional_row(
            "DFFB-601",
            row.decision_time_utc,
            row.entry_time_utc,
            row.exit_time_utc,
            row.side,
            "DFFB-601:primary",
        )
        for row in frame.itertuples(index=False)
    ]


def _live_anchor_rows(path: Path) -> list[dict[str, Any]]:
    events = _stream_top_level_json_value(path, "events")
    if not isinstance(events, list) or len(events) != 136:
        raise RuntimeError("live-anchor event count drift")
    output: list[dict[str, Any]] = []
    for event_index, row in enumerate(events):
        if not isinstance(row, dict) or set(row) != LIVE_KEYS:
            raise RuntimeError("live-anchor event schema drift")
        output.append(
            _timestamp_row(
                "live_anchor_2023",
                row["signal_time"],
                row["entry_time"],
                f"live_anchor_2023:event_{event_index:03d}",
            )
        )
    return output


def _microstructure_rows(path: Path) -> list[dict[str, Any]]:
    comparators = _stream_top_level_json_value(path, "comparators")
    if not isinstance(comparators, dict) or set(comparators) != set(
        MICROSTRUCTURE_COUNTS
    ):
        raise RuntimeError("prior-microstructure comparator set drift")
    output: list[dict[str, Any]] = []
    for name, expected_count in MICROSTRUCTURE_COUNTS.items():
        descriptor = comparators[name]
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"prior-microstructure descriptor drift: {name}")
        events = descriptor.get("events")
        if not isinstance(events, list) or len(events) != expected_count:
            raise RuntimeError(f"prior-microstructure event count drift: {name}")
        if descriptor.get("clock_rows") != expected_count:
            raise RuntimeError(f"prior-microstructure clock count drift: {name}")
        for event in events:
            if not isinstance(event, dict) or set(event) != {"signal_date", "side"}:
                raise RuntimeError(f"prior-microstructure event schema drift: {name}")
            decision = cast(pd.Timestamp, pd.Timestamp(event["signal_date"], tz="UTC"))
            output.append(
                _timestamp_row(
                    f"prior_microstructure:{name}",
                    decision,
                    decision + pd.Timedelta(minutes=5),
                    f"prior_microstructure:{name}",
                )
            )
    return output


def _validate_output(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or set(frame.columns) != set(CLOCK_COLUMNS):
        raise RuntimeError("sanitized comparator output schema drift")
    frame = cast(pd.DataFrame, frame.loc[:, pd.Index(CLOCK_COLUMNS)])
    if bool(
        frame.duplicated(
            ["comparator", "decision_time", "entry_time", "source_clock"]
        ).any()
    ):
        raise RuntimeError("sanitized comparator contains duplicate clocks")
    directional = frame["capability"].eq(DIRECTIONAL)
    directional_names = set(frame.loc[directional, "comparator"])
    timestamp_names = set(frame.loc[~directional, "comparator"])
    if directional_names != EXPECTED_DIRECTIONAL_COMPARATORS:
        raise RuntimeError("directional comparator identity set drift")
    if timestamp_names != EXPECTED_TIMESTAMP_COMPARATORS:
        raise RuntimeError("timestamp comparator identity set drift")
    if not bool(frame.loc[directional, "side"].astype(int).isin((-1, 1)).all()):
        raise RuntimeError("directional comparator has an invalid side")
    if not bool(frame.loc[directional, "exit_time"].ne("").all()):
        raise RuntimeError("directional comparator has an empty exit")
    if not bool(frame.loc[~directional, "side"].eq("").all()):
        raise RuntimeError("timestamp-only comparator retained a side")
    if not bool(frame.loc[~directional, "exit_time"].eq("").all()):
        raise RuntimeError("timestamp-only comparator retained an exit")
    return frame.sort_values(
        ["comparator", "decision_time", "entry_time", "source_clock"],
        kind="mergesort",
    ).reset_index(drop=True)


def _gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
        path.chmod(mode)
    finally:
        temporary.unlink(missing_ok=True)


def build_bundle(cfg: Config | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen_cfg = Config() if cfg is None else cfg
    input_specs = {
        "orfr": (ORFR_CLOCK, ORFR_SHA256, "ORFR-1 clock"),
        "cvtr": (CVTR_CLOCK, CVTR_SHA256, "CVTR-1 clock"),
        "ntb": (NTB_CLOCK, NTB_SHA256, "NTB-7 clock"),
        "nwe7": (NWE7_CLOCK, NWE7_SHA256, "NWE-7 clock"),
        "nwe8": (NWE8_SELECTION, NWE8_SHA256, "NWE-8 selection"),
        "chain_clock": (CHAIN_CLOCK, CHAIN_CLOCK_SHA256, "chain clock"),
        "chain_manifest": (
            CHAIN_MANIFEST,
            CHAIN_MANIFEST_SHA256,
            "chain clock manifest",
        ),
        "flcc": (FLCC_CLOCK, FLCC_SHA256, "FLCC-1 clock"),
        "dffb": (DFFB_CLOCK, DFFB_SHA256, "DFFB-601 clock"),
        "live_anchor": (LIVE_ANCHOR, LIVE_ANCHOR_SHA256, "live-anchor clock"),
        "prior_microstructure": (
            MICROSTRUCTURE_BUNDLE,
            MICROSTRUCTURE_SHA256,
            "prior-microstructure bundle",
        ),
    }
    inputs = {
        name: _require_file(path, expected_sha, label)
        for name, (path, expected_sha, label) in input_specs.items()
    }
    rows = [
        *_orfr_rows(inputs["orfr"]),
        *_cvtr_rows(inputs["cvtr"]),
        *_ntb_rows(inputs["ntb"]),
        *_nwe7_rows(inputs["nwe7"]),
        *_nwe8_rows(inputs["nwe8"]),
        *_chain_rows(inputs["chain_clock"]),
        *_flcc_rows(inputs["flcc"]),
        *_dffb_rows(inputs["dffb"]),
        *_live_anchor_rows(inputs["live_anchor"]),
        *_microstructure_rows(inputs["prior_microstructure"]),
    ]
    frame = _validate_output(rows)
    counts = {
        str(name): int(value)
        for name, value in frame["comparator"].value_counts().sort_index().items()
    }
    capability_counts = {
        str(name): int(value)
        for name, value in frame["capability"].value_counts().sort_index().items()
    }
    clock_bytes = _gzip_csv_bytes(frame)
    manifest_core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": "CDLTR-72A",
        "config": asdict(frozen_cfg),
        "amendment": {"path": str(AMENDMENT), "sha256": sha256_file(AMENDMENT)},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "inputs": {
            name: {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256_path(path),
            }
            for name, path in inputs.items()
        },
        "protocol": {
            "cdltr_source_rows_read": 0,
            "cdltr_incidence_rows_derived": 0,
            "chain_raw_market_network_funding_rows_read": 0,
            "chain_execution_or_research_code_imported": False,
            "json_sibling_outcomes_decoded": False,
            "prior_return_fields_retained": 0,
            "prior_pnl_fields_retained": 0,
            "prior_equity_cagr_mdd_computed": False,
            "timestamp_only_side_and_exit_forced_empty": True,
            "complete_comparator_identity_set_enforced": True,
            "output_columns": list(CLOCK_COLUMNS),
        },
        "clock": {
            "path": frozen_cfg.output_clock,
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": int(len(frame)),
            "columns": list(CLOCK_COLUMNS),
            "counts": counts,
            "capability_counts": capability_counts,
            "directional_comparators": sorted(EXPECTED_DIRECTIONAL_COMPARATORS),
            "timestamp_only_comparators": sorted(EXPECTED_TIMESTAMP_COMPARATORS),
        },
        "next_action": "bind only this complete sanitized bundle in CDLTR-72A preregistration",
    }
    manifest = {**manifest_core, "manifest_hash": canonical_hash(manifest_core)}
    return frame, manifest


def write_bundle(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    clock_path = _repository_path(frozen_cfg.output_clock)
    manifest_path = _repository_path(frozen_cfg.output_manifest)
    if clock_path.exists() or clock_path.is_symlink():
        raise FileExistsError("CDLTR comparator clock is immutable")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("CDLTR comparator manifest is immutable")
    frame, manifest = build_bundle(frozen_cfg)
    clock_bytes = _gzip_csv_bytes(frame)
    if hashlib.sha256(clock_bytes).hexdigest() != manifest["clock"]["sha256"]:
        raise RuntimeError("CDLTR comparator clock hash drift before write")
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write(clock_path, clock_bytes, mode=0o444)
    try:
        _atomic_write(manifest_path, manifest_bytes, mode=0o444)
    except Exception:
        clock_path.chmod(0o644)
        clock_path.unlink(missing_ok=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--output-manifest", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


def main() -> None:
    manifest = write_bundle(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "clock": manifest["clock"],
                "manifest_hash": manifest["manifest_hash"],
                "outcomes_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
