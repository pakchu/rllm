"""PSIM daily-state adapter for the frozen BCTP transition evaluator.

The BCTP reward builder consumes a wide categorical state frame, but its
economic calculation uses only ``sequence_id`` and ``entry_time``.  This module
constructs that exact identity/time schema from the sealed PSIM source rows.
The placeholder token columns are never exposed to the semantic policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from training import bctp_transition_labels as bctp


PSIM_SOURCE_SCHEMA_VERSION = "psim_d8_rllm2_source_row_v1"
UNUSED_TOKEN = "PSIM_SEMANTIC_STATE_UNUSED"
REQUIRED_SOURCE_FIELDS = (
    "row_index",
    "row_hash",
    "decision_at",
    "split",
    "split_year",
    "schedule",
    "source_payload",
    "source_payload_sha256",
)
EXPECTED_YEAR_COUNTS = {
    2020: 366,
    2021: 365,
    2022: 365,
    2023: 365,
}


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None or pd.isna(parsed):
        raise ValueError(f"PSIM semantic {name} must be timezone aware")
    return parsed.tz_convert("UTC")


def validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_year_counts: Mapping[int, int] | None = None,
    require_zero_start: bool = True,
) -> list[dict[str, Any]]:
    """Validate sealed PSIM source identity and chronology."""

    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("PSIM semantic source rows are empty")
    prior_time: pd.Timestamp | None = None
    prior_index: int | None = None
    row_hashes: set[str] = set()
    counts: dict[int, int] = {}
    for ordinal, row in enumerate(records):
        missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"PSIM semantic source row {ordinal} misses {missing}"
            )
        if row.get("schema_version") != PSIM_SOURCE_SCHEMA_VERSION:
            raise ValueError("PSIM semantic source schema changed")
        row_index = int(row["row_index"])
        if (
            (prior_index is None and require_zero_start and row_index != 0)
            or (
                prior_index is not None
                and row_index != prior_index + 1
            )
        ):
            raise ValueError("PSIM semantic source row index changed")
        prior_index = row_index
        row_hash = str(row["row_hash"])
        if len(row_hash) != 64 or row_hash in row_hashes:
            raise ValueError("PSIM semantic source row hash changed")
        row_hashes.add(row_hash)
        decision = _utc(row["decision_at"], name="decision timestamp")
        if (
            decision.hour != 12
            or decision.minute != 5
            or decision.second != 0
            or decision.microsecond != 0
        ):
            raise ValueError("PSIM semantic decision clock changed")
        if prior_time is not None and decision - prior_time != pd.Timedelta(days=1):
            raise ValueError("PSIM semantic daily chronology changed")
        prior_time = decision
        year = int(row["split_year"])
        if decision.year != year:
            raise ValueError("PSIM semantic split year changed")
        if row["schedule"] != "ARCHIVE_D90":
            raise ValueError("PSIM semantic schedule changed")
        if not isinstance(row["source_payload"], Mapping):
            raise ValueError("PSIM semantic source payload changed")
        counts[year] = counts.get(year, 0) + 1
    if expected_year_counts is not None and counts != dict(expected_year_counts):
        raise ValueError(
            f"PSIM semantic annual source counts changed: {counts}"
        )
    return records


def load_source_rows(
    path: str | Path,
    *,
    enforce_official_counts: bool = True,
) -> list[dict[str, Any]]:
    target = Path(path)
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return validate_source_rows(
        rows,
        expected_year_counts=(
            EXPECTED_YEAR_COUNTS if enforce_official_counts else None
        ),
    )


def transition_state_frame(
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build the exact BCTP identity/time frame for strict reward calculation."""

    records = validate_source_rows(rows, require_zero_start=False)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        prior_2 = records[max(0, index - 2)]
        prior_1 = records[max(0, index - 1)]
        current_hash = str(row["row_hash"])
        record: dict[str, Any] = {
            "sequence_id": current_hash,
            "entry_time": _utc(
                row["decision_at"],
                name="transition entry",
            ),
            "source_signal_id_m2": str(prior_2["row_hash"]),
            "source_signal_id_m1": str(prior_1["row_hash"]),
            "source_signal_id_s0": current_hash,
            "source_signature": str(row["source_payload_sha256"]),
        }
        record.update(
            {column: UNUSED_TOKEN for column in bctp.TOKEN_COLUMNS}
        )
        output.append(record)
    frame = pd.DataFrame(output, columns=bctp.SOURCE_COLUMNS)
    if (
        tuple(frame.columns) != bctp.SOURCE_COLUMNS
        or not frame["sequence_id"].is_unique
        or not frame["entry_time"].is_monotonic_increasing
    ):
        raise RuntimeError("PSIM semantic transition adapter changed")
    return frame


def select_year(
    rows: Sequence[Mapping[str, Any]],
    year: int,
) -> list[dict[str, Any]]:
    records = validate_source_rows(rows)
    selected = [row for row in records if int(row["split_year"]) == int(year)]
    if not selected:
        raise ValueError(f"PSIM semantic year has no source rows: {year}")
    return selected


def build_reward_tensor(
    rows: Sequence[Mapping[str, Any]],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate: float = 0.0006,
) -> dict[str, Any]:
    """Delegate strict counterfactual rewards using PSIM daily identities."""

    states = transition_state_frame(rows)
    result = bctp.build_reward_tensor(
        states,
        market,
        funding,
        start=_utc(start, name="stage start"),
        end=_utc(end, name="stage end"),
        cost_rate=float(cost_rate),
    )
    result["psim_source_row_hashes"] = [
        str(row["row_hash"]) for row in rows
    ]
    return result
