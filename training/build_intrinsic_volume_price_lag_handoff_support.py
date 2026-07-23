"""Build outcome-blind IVPLH-72 source-support clocks and diagnostics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_intrinsic_volume_latent_impact_relay_support as ivlir
from training import preregister_intrinsic_volume_price_lag_handoff as prereg


PROTOCOL_VERSION = "intrinsic_volume_price_lag_handoff_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_intrinsic_volume_price_lag_handoff_support.py")
TEST_PATH = Path("tests/test_build_intrinsic_volume_price_lag_handoff_support.py")
IMPLEMENTATION_CONTRACT = Path(
    "docs/ivplh-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "579828b3ea92430527aea82b0fdc23b19866553f7e7e8792f3389f6df4c8ec8e"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "942f519f87a86f7c4764b01aca1f2e4b524749888620a3aded51040970163551"
)
PREREGISTRATION_MANIFEST_HASH = (
    "a647a944a65b46fa52799d544acfee1a4c8c72722e05b54c3cf891c2537f0619"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/intrinsic_volume_price_lag_handoff_support_2026-07-24.json"
)

BAR = pd.Timedelta(minutes=5)
DAY = pd.Timedelta(days=1)
CALIBRATION_START = pd.Timestamp("2020-01-01T00:00:00Z")
TRAIN_START = pd.Timestamp("2021-01-01T00:00:00Z")
CALIBRATION_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = CALIBRATION_END
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")

CONTROL_ORDER = (
    "primary",
    "handoff_without_price_lag",
    "price_lag_without_handoff",
    "fixed_noon",
    "stale_24h",
    "direction_flip",
    "anchor_side_year_permutation",
    "anchor_return_year_permutation",
    "deterministic_random_side",
)
PERMUTATION_CONTROLS = (
    "anchor_side_year_permutation",
    "anchor_return_year_permutation",
)
CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "source_day",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)
FORBIDDEN_CLOCK_TOKENS = (
    "open",
    "high",
    "low",
    "close",
    "price",
    "return",
    "future",
    "label",
    "funding",
    "pnl",
    "reward",
    "cagr",
    "mdd",
)

WINDOWS = {
    "train": (TRAIN_START, CALIBRATION_END),
    "selection": (SELECTION_START, SELECTION_END),
    "2021": (TRAIN_START, pd.Timestamp("2022-01-01T00:00:00Z")),
    "2022": (pd.Timestamp("2022-01-01T00:00:00Z"), CALIBRATION_END),
    "2021_h1": (TRAIN_START, pd.Timestamp("2021-07-01T00:00:00Z")),
    "2021_h2": (
        pd.Timestamp("2021-07-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2022_h1": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2022-07-01T00:00:00Z"),
    ),
    "2022_h2": (
        pd.Timestamp("2022-07-01T00:00:00Z"),
        CALIBRATION_END,
    ),
    "2023_h1": (SELECTION_START, pd.Timestamp("2023-07-01T00:00:00Z")),
    "2023_h2": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        SELECTION_END,
    ),
}


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("IVPLH timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("IVPLH timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_day(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("IVPLH source day must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("D"):
        raise RuntimeError("IVPLH source day must be UTC midnight")
    return timestamp.strftime("%Y-%m-%d")


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (str(SCRIPT_PATH), str(TEST_PATH), str(IMPLEMENTATION_CONTRACT))
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("IVPLH source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("IVPLH source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("IVPLH preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text("utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("IVPLH preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("IVPLH preregistration manifest hash drift")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "predecessor_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"IVPLH preregistration boundary opened: {field}")
    if tuple(payload["source_only_controls"]["ordered"]) != CONTROL_ORDER:
        raise RuntimeError("IVPLH control order drift")
    return payload


def verify_pre_source_bindings(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    predecessor = payload["predecessor_lineage"]
    bindings = (
        (
            payload["frozen_documents"]["boundary"]["path"],
            payload["frozen_documents"]["boundary"]["sha256"],
            "boundary",
        ),
        (
            payload["frozen_documents"]["mechanism"]["path"],
            payload["frozen_documents"]["mechanism"]["sha256"],
            "mechanism",
        ),
        (
            payload["frozen_documents"]["common_window_policy"]["path"],
            payload["frozen_documents"]["common_window_policy"]["sha256"],
            "common_window_policy",
        ),
        (
            payload["source_contract"]["market_manifest"],
            payload["source_contract"]["market_manifest_sha256"],
            "market_manifest",
        ),
        (
            payload["source_contract"]["market"],
            payload["source_contract"]["market_sha256"],
            "market_source",
        ),
        (
            predecessor["preregistration"]["path"],
            predecessor["preregistration"]["sha256"],
            "predecessor_preregistration",
        ),
        (
            predecessor["support_report"]["path"],
            predecessor["support_report"]["sha256"],
            "predecessor_support_report",
        ),
        (
            predecessor["clock"]["path"],
            predecessor["clock"]["sha256"],
            "predecessor_clock",
        ),
        (
            IMPLEMENTATION_CONTRACT,
            IMPLEMENTATION_CONTRACT_SHA256,
            "implementation_contract",
        ),
    )
    audit: dict[str, dict[str, str]] = {}
    for path, expected, label in bindings:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"IVPLH frozen binding changed: {label}")
        audit[label] = {"path": str(path), "sha256": actual}
    return audit


def _read_exact_header(path: str | Path) -> list[str]:
    source = _path(path)
    opener: Callable[..., Any] = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        line = handle.readline()
    if not line.endswith("\n") or "\n" in line[:-1] or "\r" in line:
        raise RuntimeError(f"IVPLH clock header is not one LF line: {path}")
    return line[:-1].split(",")


def load_predecessor(
    payload: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = payload["predecessor_lineage"]
    report = json.loads(
        _path(contract["support_report"]["path"]).read_text("utf-8")
    )
    if report.get("outcomes_opened") is not False:
        raise RuntimeError("IVPLH predecessor report opened outcomes")
    if set(report.get("controls", {})) != set(contract["known_clock_names"]):
        raise RuntimeError("IVPLH predecessor report control set drift")
    selected_name = contract["selected_clock_name"]
    if (
        report["controls"][selected_name]["events"]
        != contract["disclosed_global_rows"]
    ):
        raise RuntimeError("IVPLH predecessor disclosed row count drift")

    clock_contract = contract["clock"]
    if _read_exact_header(clock_contract["path"]) != clock_contract["header"]:
        raise RuntimeError("IVPLH predecessor clock header drift")
    selected_rows: list[dict[str, Any]] = []
    physical_rows = 0
    with gzip.open(
        _path(clock_contract["path"]),
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != clock_contract["header"]:
            raise RuntimeError("IVPLH predecessor clock schema drift")
        for row in reader:
            physical_rows += 1
            name = str(row["clock_name"])
            if name not in contract["known_clock_names"]:
                raise RuntimeError(
                    f"IVPLH predecessor has unknown control: {name}"
                )
            if name != selected_name:
                continue
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows, columns=clock_contract["header"])
    for column in ("source_day", "decision_time", "entry_time", "exit_time"):
        selected[column] = pd.to_datetime(
            selected[column],
            utc=True,
            errors="raise",
        )
    if not selected["source_day"].eq(selected["source_day"].dt.floor("D")).all():
        raise RuntimeError("IVPLH predecessor source-day invariant failed")
    if not selected["side"].isin(("LONG", "SHORT")).all():
        raise RuntimeError("IVPLH predecessor side invariant failed")
    if not selected["decision_time"].eq(selected["entry_time"]).all():
        raise RuntimeError("IVPLH predecessor decision/entry invariant failed")
    if not selected["exit_time"].eq(selected["entry_time"] + 72 * BAR).all():
        raise RuntimeError("IVPLH predecessor hold invariant failed")

    selected = (
        selected.sort_values(
            ["entry_time", "source_day", "side"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if len(selected) != contract["disclosed_global_rows"]:
        raise RuntimeError("IVPLH predecessor selected rows drift")
    identity = selected.loc[:, contract["identity_key"]]
    if identity.duplicated().any():
        raise RuntimeError("IVPLH predecessor identity duplicated")
    return selected, {
        "report_rows_decoded": 1,
        "physical_clock_rows_scanned": physical_rows,
        "clock_rows_decoded": len(selected),
        "selected_rows": len(selected),
        "selected_clock_name": selected_name,
    }


def load_source(path: str | Path = prereg.MARKET_SOURCE) -> pd.DataFrame:
    if str(path) != prereg.MARKET_SOURCE:
        raise RuntimeError("IVPLH support must use the frozen source path")
    return ivlir.load_source(path)


def build_anchor_features(
    frame: pd.DataFrame,
    *,
    fixed_noon: bool = False,
) -> pd.DataFrame:
    """Build anchors from frozen state columns; high/low remain validation-only."""
    days = frame["date"].dt.floor("D")
    counts = frame.groupby(days, sort=True).size()
    if not len(counts) or not counts.eq(288).all():
        raise RuntimeError("IVPLH source lacks complete UTC days")
    daily = (
        frame.groupby(days, sort=True)["quote_asset_volume"]
        .sum()
        .astype(float)
        .rename("daily_quote_volume")
        .to_frame()
    )
    daily["expected_quote_volume"] = (
        daily["daily_quote_volume"]
        .shift(1)
        .rolling(
            prereg.Policy().utc_day_volume_lookback_days,
            min_periods=prereg.Policy().utc_day_volume_min_days,
        )
        .median()
    )
    signed_quote = 2.0 * frame["taker_buy_quote"] - frame["quote_asset_volume"]
    rows: list[dict[str, Any]] = []
    for day, daily_row in daily.iterrows():
        expected = float(daily_row["expected_quote_volume"])
        if not np.isfinite(expected) or expected <= 0:
            continue
        source_day = pd.Timestamp(day)
        start = int((source_day - CALIBRATION_START) / BAR)
        end = start + 288
        if start < 0 or end > len(frame):
            raise RuntimeError("IVPLH UTC-day slice escaped frozen source")
        cumulative_quote = np.cumsum(
            frame["quote_asset_volume"].iloc[start:end].to_numpy(float)
        )
        target = prereg.Policy().intrinsic_volume_fraction * expected
        if fixed_noon:
            local_index = prereg.Policy().fixed_noon_anchor_minute_utc // 5
            if cumulative_quote[local_index] < target:
                continue
        else:
            local_index = int(
                np.searchsorted(cumulative_quote, target, side="left")
            )
            if local_index >= len(cumulative_quote):
                continue
        anchor_index = start + local_index
        anchor_time = pd.Timestamp(frame.at[anchor_index, "date"])
        minute = anchor_time.hour * 60 + anchor_time.minute
        if minute > prereg.Policy().latest_anchor_minute_utc:
            continue
        cumulative_volume = float(cumulative_quote[local_index])
        if cumulative_volume <= 0:
            continue
        cumulative_signed = float(
            signed_quote.iloc[start : anchor_index + 1].sum()
        )
        cumulative_flow = cumulative_signed / cumulative_volume
        if not np.isfinite(cumulative_flow) or cumulative_flow == 0.0:
            continue
        day_open = float(frame.at[start, "open"])
        anchor_close = float(frame.at[anchor_index, "close"])
        anchor_return = float(np.log(anchor_close / day_open))
        if not np.isfinite(anchor_return):
            raise RuntimeError("IVPLH anchor return is non-finite")
        side_sign = 1 if cumulative_flow > 0 else -1
        rows.append(
            {
                "source_day": source_day,
                "anchor_index": anchor_index,
                "anchor_time": anchor_time,
                "side_sign": side_sign,
                "side": "LONG" if side_sign > 0 else "SHORT",
                "cumulative_flow": cumulative_flow,
                "anchor_return": anchor_return,
                "anchor_minute_utc": minute,
                "target_quote_volume": target,
                "cumulative_quote_volume": cumulative_volume,
            }
        )
    return annotate_state(pd.DataFrame(rows))


def annotate_state(
    anchors: pd.DataFrame,
    policy: prereg.Policy = prereg.Policy(),
) -> pd.DataFrame:
    """Recompute the frozen handoff state using only current/prior anchors."""
    state_columns = (
        "reference_count",
        "reference_ready",
        "calendar_consecutive",
        "prior_side",
        "handoff",
        "directional_return",
        "price_lag",
        "primary",
    )
    anchors = anchors.drop(columns=list(state_columns), errors="ignore")
    if anchors.empty:
        empty = anchors.copy()
        for column, dtype in (
            ("reference_count", int),
            ("reference_ready", bool),
            ("calendar_consecutive", bool),
            ("prior_side", object),
            ("handoff", bool),
            ("directional_return", float),
            ("price_lag", bool),
            ("primary", bool),
        ):
            empty[column] = pd.Series(dtype=dtype)
        return empty

    ordered = anchors.sort_values("source_day", kind="mergesort").reset_index(
        drop=True
    )
    if ordered["source_day"].duplicated().any():
        raise RuntimeError("IVPLH has multiple eligible anchors for one UTC day")
    if not ordered["source_day"].eq(ordered["source_day"].dt.floor("D")).all():
        raise RuntimeError("IVPLH anchor source day is not UTC midnight")
    if not ordered["side"].isin(("LONG", "SHORT")).all():
        raise RuntimeError("IVPLH anchor side is invalid")
    expected_sign = np.where(ordered["side"].eq("LONG"), 1, -1)
    if not np.array_equal(ordered["side_sign"].to_numpy(int), expected_sign):
        raise RuntimeError("IVPLH side text/sign mismatch")
    if not np.isfinite(ordered["anchor_return"].to_numpy(float)).all():
        raise RuntimeError("IVPLH anchor return is non-finite")

    records: list[dict[str, Any]] = []
    previous_day: pd.Timestamp | None = None
    previous_side = ""
    for index, row in enumerate(ordered.itertuples(index=False)):
        day = pd.Timestamp(row.source_day)
        side = str(row.side)
        reference_count = min(index, policy.event_reference_anchors)
        reference_ready = reference_count >= policy.event_reference_min_anchors
        consecutive = previous_day is not None and day == previous_day + DAY
        handoff = bool(consecutive and side != previous_side)
        directional_return = float(int(row.side_sign) * float(row.anchor_return))
        price_lag = bool(directional_return <= 0.0)
        records.append(
            {
                "reference_count": reference_count,
                "reference_ready": reference_ready,
                "calendar_consecutive": consecutive,
                "prior_side": previous_side if consecutive else "",
                "handoff": handoff,
                "directional_return": directional_return,
                "price_lag": price_lag,
                "primary": bool(reference_ready and handoff and price_lag),
            }
        )
        previous_day = day
        previous_side = side
    return pd.concat([ordered, pd.DataFrame(records)], axis=1)


def _permutation_digest(
    control: str,
    role: str,
    year: int,
    source_day: Any,
) -> str:
    text = (
        f"{prereg.Policy().policy_id}|{control}|{role}|{year}|"
        f"{_format_day(source_day)}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def permute_anchor_field(
    anchors: pd.DataFrame,
    control: str,
) -> pd.DataFrame:
    if control not in PERMUTATION_CONTROLS:
        raise RuntimeError("IVPLH unknown permutation control")
    if anchors["source_day"].duplicated().any():
        raise RuntimeError("IVPLH permutation source days duplicated")
    result = anchors.copy()
    field = "side_sign" if control == "anchor_side_year_permutation" else "anchor_return"
    for year, positions in result.groupby(
        result["source_day"].dt.year,
        sort=True,
    ).groups.items():
        indexes = list(positions)
        donor = sorted(
            indexes,
            key=lambda index: _permutation_digest(
                control,
                "donor",
                int(year),
                result.at[index, "source_day"],
            ),
        )
        destination = sorted(
            indexes,
            key=lambda index: _permutation_digest(
                control,
                "destination",
                int(year),
                result.at[index, "source_day"],
            ),
        )
        donor_hashes = {
            _permutation_digest(
                control,
                "donor",
                int(year),
                result.at[index, "source_day"],
            )
            for index in indexes
        }
        destination_hashes = {
            _permutation_digest(
                control,
                "destination",
                int(year),
                result.at[index, "source_day"],
            )
            for index in indexes
        }
        if len(donor_hashes) != len(indexes) or len(destination_hashes) != len(
            indexes
        ):
            raise RuntimeError("IVPLH permutation SHA sort-key collision")
        values = [result.at[index, field] for index in donor]
        for index, value in zip(destination, values, strict=True):
            result.at[index, field] = value
    if field == "side_sign":
        result["side_sign"] = result["side_sign"].astype(int)
        result["side"] = np.where(result["side_sign"].gt(0), "LONG", "SHORT")
    return annotate_state(result)


def signal_id(
    control: str,
    source_day: Any,
    decision_time: Any,
    side: str,
) -> str:
    if control not in CONTROL_ORDER:
        raise RuntimeError("IVPLH signal has unknown control")
    if side not in {"LONG", "SHORT"}:
        raise RuntimeError("IVPLH signal side is invalid")
    payload = {
        "control": control,
        "decision_time": _format_time(decision_time),
        "policy_id": prereg.Policy().policy_id,
        "side": side,
        "source_day": _format_day(source_day),
        "source_panel_sha256": prereg.MARKET_SOURCE_SHA256,
    }
    return canonical_hash(payload)


def _candidate_row(
    control: str,
    source_day: Any,
    decision_time: Any,
    entry_time: Any,
    exit_time: Any,
    side: str,
) -> dict[str, Any]:
    return {
        "control": control,
        "signal_id": signal_id(control, source_day, decision_time, side),
        "source_day": pd.Timestamp(source_day),
        "decision_time": pd.Timestamp(decision_time),
        "entry_time": pd.Timestamp(entry_time),
        "exit_time": pd.Timestamp(exit_time),
        "side": side,
    }


def raw_candidates(
    features: pd.DataFrame,
    control: str,
    mask: pd.Series,
) -> pd.DataFrame:
    rows = []
    for row in features.loc[mask].itertuples(index=False):
        decision = pd.Timestamp(row.anchor_time) + prereg.Policy().decision_delay_bars * BAR
        entry = pd.Timestamp(row.anchor_time) + prereg.Policy().entry_delay_bars * BAR
        rows.append(
            _candidate_row(
                control,
                row.source_day,
                decision,
                entry,
                entry + prereg.Policy().hold_bars * BAR,
                str(row.side),
            )
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stale_candidates(primary: pd.DataFrame) -> pd.DataFrame:
    shift = prereg.Policy().stale_control_delay_bars * BAR
    rows = [
        _candidate_row(
            "stale_24h",
            row.source_day,
            pd.Timestamp(row.decision_time) + shift,
            pd.Timestamp(row.entry_time) + shift,
            pd.Timestamp(row.exit_time) + shift,
            str(row.side),
        )
        for row in primary.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def direction_flip_candidates(primary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _candidate_row(
            "direction_flip",
            row.source_day,
            row.decision_time,
            row.entry_time,
            row.exit_time,
            "SHORT" if row.side == "LONG" else "LONG",
        )
        for row in primary.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def deterministic_random_side_candidates(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in primary.itertuples(index=False):
        side_free = {
            "control": "deterministic_random_side",
            "decision_time": _format_time(row.decision_time),
            "policy_id": prereg.Policy().policy_id,
            "primary_entry_time": _format_time(row.entry_time),
            "source_day": _format_day(row.source_day),
            "source_panel_sha256": prereg.MARKET_SOURCE_SHA256,
        }
        digest = hashlib.sha256(
            json.dumps(
                side_free,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        side = "LONG" if digest[0] < 128 else "SHORT"
        rows.append(
            _candidate_row(
                "deterministic_random_side",
                row.source_day,
                row.decision_time,
                row.entry_time,
                row.exit_time,
                side,
            )
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _contained(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return frame.loc[
        frame["source_day"].ge(start)
        & frame["source_day"].lt(end)
        & frame["decision_time"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["entry_time"].lt(end)
        & frame["exit_time"].le(end)
    ].copy()


def _schedule_split(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    eligible = _contained(frame, start, end).sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )
    selected: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in eligible.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.append({column: getattr(row, column) for column in CLOCK_COLUMNS})
        previous_exit = pd.Timestamp(row.exit_time)
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def schedule_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    if frame["signal_id"].duplicated().any():
        raise RuntimeError("IVPLH raw candidate signal identity duplicated")
    parts = (
        _schedule_split(frame, CALIBRATION_START, CALIBRATION_END),
        _schedule_split(frame, SELECTION_START, SELECTION_END),
    )
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(["entry_time", "signal_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_controls(
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    side_permuted = permute_anchor_field(
        features,
        "anchor_side_year_permutation",
    )
    return_permuted = permute_anchor_field(
        features,
        "anchor_return_year_permutation",
    )
    primary_raw = raw_candidates(
        features,
        "primary",
        features["primary"],
    )
    raw = {
        "primary": primary_raw,
        "handoff_without_price_lag": raw_candidates(
            features,
            "handoff_without_price_lag",
            features["reference_ready"] & features["handoff"],
        ),
        "price_lag_without_handoff": raw_candidates(
            features,
            "price_lag_without_handoff",
            features["reference_ready"] & features["price_lag"],
        ),
        "fixed_noon": raw_candidates(
            fixed_noon_features,
            "fixed_noon",
            fixed_noon_features["primary"],
        ),
        "stale_24h": stale_candidates(primary_raw),
        "direction_flip": direction_flip_candidates(primary_raw),
        "anchor_side_year_permutation": raw_candidates(
            side_permuted,
            "anchor_side_year_permutation",
            side_permuted["primary"],
        ),
        "anchor_return_year_permutation": raw_candidates(
            return_permuted,
            "anchor_return_year_permutation",
            return_permuted["primary"],
        ),
        "deterministic_random_side": deterministic_random_side_candidates(
            primary_raw
        ),
    }
    controls = {name: schedule_candidates(raw[name]) for name in CONTROL_ORDER}
    funnel = {
        "first_passage_anchors": len(features),
        "reference_ready": int(features["reference_ready"].sum()),
        "calendar_consecutive": int(features["calendar_consecutive"].sum()),
        "handoff": int(features["handoff"].sum()),
        "price_lag": int(features["price_lag"].sum()),
        "raw_primary": len(primary_raw),
        "fixed_noon_anchors": len(fixed_noon_features),
        "fixed_noon_raw_primary": int(fixed_noon_features["primary"].sum()),
        "side_permutation_raw_primary": int(side_permuted["primary"].sum()),
        "return_permutation_raw_primary": int(return_permuted["primary"].sum()),
        "raw_control_counts": {name: len(raw[name]) for name in CONTROL_ORDER},
    }
    return controls, funnel


def validate_predecessor_identity(
    primary: pd.DataFrame,
    predecessor: pd.DataFrame,
) -> dict[str, bool]:
    left = primary.sort_values(
        ["decision_time", "source_day", "side"],
        kind="mergesort",
    ).reset_index(drop=True)
    right = predecessor.sort_values(
        ["entry_time", "source_day", "side"],
        kind="mergesort",
    ).reset_index(drop=True)
    row_count = len(left) == len(right) == 66
    if not row_count:
        return {
            "predecessor_row_count_exact": False,
            "predecessor_identity_exact": False,
            "predecessor_entry_shift_exact": False,
            "predecessor_exit_shift_exact": False,
        }
    identity_exact = bool(
        left["source_day"].equals(right["source_day"])
        and left["side"].equals(right["side"])
        and left["decision_time"].equals(right["entry_time"])
    )
    return {
        "predecessor_row_count_exact": row_count,
        "predecessor_identity_exact": identity_exact,
        "predecessor_entry_shift_exact": bool(
            left["entry_time"].equals(right["entry_time"] + BAR)
        ),
        "predecessor_exit_shift_exact": bool(
            left["exit_time"].equals(right["exit_time"] + BAR)
        ),
    }


def _window(
    rows: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return _contained(rows, start, end).sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )


def _longest_run(sides: Iterable[str]) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for side in sides:
        if side == previous:
            current += 1
        else:
            current = 1
            previous = side
        best = max(best, current)
    return best


def clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    ordered = rows.sort_values(
        ["entry_time", "signal_id"],
        kind="mergesort",
    )
    total = len(ordered)
    if not total:
        return {
            "events": 0,
            "long": 0,
            "short": 0,
            "long_share": None,
            "short_share": None,
            "active_months": 0,
            "maximum_month_share": None,
            "maximum_quarter_share": None,
            "maximum_gap_days": None,
            "maximum_same_side_run": 0,
        }
    counts = ordered["side"].value_counts().to_dict()
    entry = ordered["entry_time"].dt.tz_convert(None)
    month_counts = entry.dt.to_period("M").astype(str).value_counts()
    quarter_counts = entry.dt.to_period("Q").astype(str).value_counts()
    gaps = ordered["entry_time"].diff().dropna()
    return {
        "events": total,
        "long": int(counts.get("LONG", 0)),
        "short": int(counts.get("SHORT", 0)),
        "long_share": float(counts.get("LONG", 0) / total),
        "short_share": float(counts.get("SHORT", 0) / total),
        "active_months": int(len(month_counts)),
        "maximum_month_share": float(month_counts.max() / total),
        "maximum_quarter_share": float(quarter_counts.max() / total),
        "maximum_gap_days": (
            float(gaps.max() / DAY) if not gaps.empty else None
        ),
        "maximum_same_side_run": _longest_run(ordered["side"]),
    }


def exact_entry_jaccard(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    split: str,
) -> float:
    left = set(_window(primary, split)["entry_time"])
    right = set(_window(control, split)["entry_time"])
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def same_side_reproduction(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    split: str,
) -> float:
    left = {
        (row.entry_time, row.side)
        for row in _window(primary, split).itertuples(index=False)
    }
    if not left:
        return 1.0
    right = {
        (row.entry_time, row.side)
        for row in _window(control, split).itertuples(index=False)
    }
    return float(len(left & right) / len(left))


def _timing_integrity(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return True
    return bool(
        rows["entry_time"].eq(rows["decision_time"] + BAR).all()
        and rows["exit_time"]
        .eq(rows["entry_time"] + prereg.Policy().hold_bars * BAR)
        .all()
        and rows["side"].isin(("LONG", "SHORT")).all()
        and not rows["signal_id"].duplicated().any()
        and all(
            signal_id(
                str(row.control),
                row.source_day,
                row.decision_time,
                str(row.side),
            )
            == row.signal_id
            for row in rows.itertuples(index=False)
        )
    )


def _reservation_integrity(rows: pd.DataFrame) -> bool:
    for start, end in (
        (CALIBRATION_START, CALIBRATION_END),
        (SELECTION_START, SELECTION_END),
    ):
        split = _contained(rows, start, end).sort_values(
            ["entry_time", "signal_id"],
            kind="mergesort",
        )
        if not split.empty and not split["entry_time"].iloc[1:].reset_index(
            drop=True
        ).ge(split["exit_time"].iloc[:-1].reset_index(drop=True)).all():
            return False
    return True


def support_checks(
    controls: Mapping[str, pd.DataFrame],
    identity_checks: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, bool]]:
    gate = prereg.build_manifest()["source_support_gate"]
    primary = controls["primary"]
    statistics = {name: clock_stats(_window(primary, name)) for name in WINDOWS}
    train = statistics["train"]
    selection = statistics["selection"]

    def side_ok(stats: Mapping[str, Any], minimum: int, share: float) -> bool:
        return bool(
            stats["long"] >= minimum
            and stats["short"] >= minimum
            and stats["long_share"] is not None
            and stats["short_share"] is not None
            and stats["long_share"] >= share
            and stats["short_share"] >= share
        )

    selectivity: dict[str, dict[str, dict[str, float]]] = {}
    for control in PERMUTATION_CONTROLS:
        selectivity[control] = {
            split: {
                "exact_entry_jaccard": exact_entry_jaccard(
                    primary,
                    controls[control],
                    split,
                ),
                "same_side_reproduction": same_side_reproduction(
                    primary,
                    controls[control],
                    split,
                ),
            }
            for split in ("train", "selection")
        }
    checks = {
        **dict(identity_checks),
        "train_events_min": train["events"] >= gate["train_events_min"],
        "each_train_year_events_min": all(
            statistics[year]["events"] >= gate["each_train_year_events_min"]
            for year in ("2021", "2022")
        ),
        "each_train_half_events_min": all(
            statistics[name]["events"] >= gate["each_train_half_events_min"]
            for name in ("2021_h1", "2021_h2", "2022_h1", "2022_h2")
        ),
        "train_side_support": side_ok(
            train,
            gate["train_each_side_events_min"],
            gate["train_each_side_share_min"],
        ),
        "selection_events_min": (
            selection["events"] >= gate["selection_events_min"]
        ),
        "each_selection_half_events_min": all(
            statistics[name]["events"]
            >= gate["each_selection_half_events_min"]
            for name in ("2023_h1", "2023_h2")
        ),
        "selection_side_support": side_ok(
            selection,
            gate["selection_each_side_events_min"],
            gate["selection_each_side_share_min"],
        ),
        "maximum_split_month_share": all(
            statistics[name]["maximum_month_share"] is not None
            and statistics[name]["maximum_month_share"]
            <= gate["maximum_split_month_share"]
            for name in ("train", "selection")
        ),
        "maximum_split_quarter_share": all(
            statistics[name]["maximum_quarter_share"] is not None
            and statistics[name]["maximum_quarter_share"]
            <= gate["maximum_split_quarter_share"]
            for name in ("train", "selection")
        ),
        "maximum_split_gap_days": all(
            statistics[name]["maximum_gap_days"] is not None
            and statistics[name]["maximum_gap_days"]
            <= gate["maximum_split_gap_days"]
            for name in ("train", "selection")
        ),
        "maximum_split_same_side_run": all(
            statistics[name]["maximum_same_side_run"]
            <= gate["maximum_split_same_side_run"]
            for name in ("train", "selection")
        ),
        "all_controls_exact_timing_and_identity": all(
            _timing_integrity(controls[name]) for name in CONTROL_ORDER
        ),
        "all_controls_nonoverlap": all(
            _reservation_integrity(controls[name]) for name in CONTROL_ORDER
        ),
        "clock_has_no_outcome_columns": not any(
            token in column.lower()
            for column in CLOCK_COLUMNS
            for token in FORBIDDEN_CLOCK_TOKENS
        ),
        "permutation_selectivity": all(
            metrics["exact_entry_jaccard"]
            <= gate["permutation_exact_entry_jaccard_max"]
            and metrics["same_side_reproduction"]
            <= gate["permutation_same_side_reproduction_max"]
            for control in PERMUTATION_CONTROLS
            for metrics in selectivity[control].values()
        ),
    }
    statistics["permutation_selectivity"] = selectivity
    return statistics, checks


def _combined_clock(controls: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    nonempty = [controls[name] for name in CONTROL_ORDER if not controls[name].empty]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(["entry_time", "signal_id", "control"], kind="mergesort")
        .reset_index(drop=True)
    )


def deterministic_clock_bytes(
    controls: Mapping[str, pd.DataFrame],
) -> bytes:
    combined = _combined_clock(controls)
    if list(combined.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("IVPLH clock schema drift")
    serialized = combined.copy()
    serialized["source_day"] = serialized["source_day"].map(_format_day)
    for column in ("decision_time", "entry_time", "exit_time"):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(
        index=False,
        columns=CLOCK_COLUMNS,
        lineterminator="\n",
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def _control_report(
    controls: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    primary = controls["primary"]
    return {
        control: {
            "clock_rows": len(controls[control]),
            "train": clock_stats(_window(controls[control], "train")),
            "selection": clock_stats(_window(controls[control], "selection")),
            "exact_entry_jaccard_to_primary": {
                split: exact_entry_jaccard(primary, controls[control], split)
                for split in ("train", "selection")
            },
            "same_side_reproduction_to_primary": {
                split: same_side_reproduction(
                    primary,
                    controls[control],
                    split,
                )
                for split in ("train", "selection")
            },
        }
        for control in CONTROL_ORDER
    }


def _build_core(
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
    predecessor: pd.DataFrame,
    source_audit: Mapping[str, Any],
    predecessor_audit: Mapping[str, Any],
    *,
    artifact_eligible: bool,
    clock_output: str | Path,
) -> tuple[dict[str, Any], bytes]:
    controls, funnel = build_controls(features, fixed_noon_features)
    identity = validate_predecessor_identity(controls["primary"], predecessor)
    statistics, checks = support_checks(controls, identity)
    passed = all(checks.values())
    clock_bytes = deterministic_clock_bytes(controls)
    if not passed:
        decision = "retire_IVPLH_72_unchanged_before_comparators_and_outcomes"
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_comparators_or_outcomes"
    else:
        decision = "advance_to_separately_frozen_comparator_novelty_evaluator"
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.Policy().policy_id,
        "artifact_eligible": artifact_eligible,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "predecessor_rows_decoded": True,
        "comparator_rows_decoded": False,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "tests": str(TEST_PATH),
            "tests_sha256": sha256_file(TEST_PATH),
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
        },
        "source_audit": dict(source_audit),
        "predecessor_audit": dict(predecessor_audit),
        "feature_funnel": funnel,
        "primary_statistics": statistics,
        "control_report": _control_report(controls),
        "identity_checks": identity,
        "support_checks": checks,
        "source_support_passed": passed,
        "clock": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": len(_combined_clock(controls)),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                name: len(controls[name]) for name in CONTROL_ORDER
            },
        },
        "comparator_status": "not_opened_source_support_stage",
        "decision": decision,
        "advance_to_comparator_novelty_freeze": bool(
            artifact_eligible and passed
        ),
        "outcome_boundary": {
            "source_rows_decoded": int(source_audit.get("source_rows_decoded", 0)),
            "predecessor_clock_rows_decoded": int(
                predecessor_audit.get("clock_rows_decoded", 0)
            ),
            "comparator_rows_decoded": 0,
            "post_entry_price_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "return_or_pnl_fields_decoded": 0,
            "pnl_cagr_mdd_values_decoded": 0,
            "network_calls": 0,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}, clock_bytes


def build_support_from_anchors(
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
    predecessor: pd.DataFrame,
    *,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    return _build_core(
        features,
        fixed_noon_features,
        predecessor,
        {
            "source_rows_decoded": 0,
            "synthetic_or_injected": True,
        },
        {
            "clock_rows_decoded": len(predecessor),
            "synthetic_or_injected": True,
        },
        artifact_eligible=False,
        clock_output=clock_output,
    )


def build_real_support_payload(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> tuple[dict[str, Any], bytes]:
    _assert_protocol_committed()
    preregistration = validate_preregistration()
    bindings = verify_pre_source_bindings(preregistration)
    predecessor, predecessor_audit = load_predecessor(preregistration)
    source = load_source()
    features = build_anchor_features(source)
    fixed_noon = build_anchor_features(source, fixed_noon=True)
    source_audit = {
        "path": prereg.MARKET_SOURCE,
        "sha256": prereg.MARKET_SOURCE_SHA256,
        "source_rows_decoded": len(source),
        "first_timestamp": _format_time(source["date"].iloc[0]),
        "last_timestamp": _format_time(source["date"].iloc[-1]),
        "complete_utc_days": int(source["date"].dt.floor("D").nunique()),
        "pre_source_bindings": bindings,
        "synthetic_or_injected": False,
    }
    return _build_core(
        features,
        fixed_noon,
        predecessor,
        source_audit,
        predecessor_audit,
        artifact_eligible=True,
        clock_output=clock_output,
    )


def _write_once(path: str | Path, payload: bytes) -> None:
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"IVPLH support artifact is write-once: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_support(
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    if (
        Path(clock_output) != DEFAULT_CLOCK_OUTPUT
        or Path(report_output) != DEFAULT_REPORT_OUTPUT
    ):
        raise RuntimeError("IVPLH eligible outputs must use frozen default paths")
    if _path(clock_output).exists() or _path(report_output).exists():
        raise FileExistsError("IVPLH support outputs are write-once")
    payload, clock_bytes = build_real_support_payload(clock_output)
    report_bytes = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_once(clock_output, clock_bytes)
    try:
        _write_once(report_output, report_bytes)
    except Exception:
        _path(clock_output).unlink(missing_ok=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    args = parser.parse_args()
    payload = write_support(args.clock_output, args.report_output)
    print(
        json.dumps(
            {
                "policy_id": payload["policy_id"],
                "source_support_passed": payload["source_support_passed"],
                "failed_checks": [
                    name
                    for name, passed in payload["support_checks"].items()
                    if not passed
                ],
                "primary_statistics": payload["primary_statistics"],
                "decision": payload["decision"],
                "clock": payload["clock"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
