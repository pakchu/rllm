"""Build outcome-blind RQHR-72 synthetic, support, control, and novelty clocks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd


POLICY_ID = "RQHR-72"
PROTOCOL_VERSION = "radial_quote_handoff_relay_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_radial_quote_handoff_relay_support.py")

PREREGISTRATION = Path(
    "results/radial_quote_handoff_relay_preregistration_v2_2026-07-23.json"
)
PREREGISTRATION_SHA256 = (
    "402c0b02cfea5932766f1892520860acd6beaeafb9e6948024d71e761c2d5f70"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_radial_quote_handoff_relay.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "409a8c326682ed687d584635ff0d27e588e8b74129eb7b367d04ff81c5307a7b"
)
MECHANISM_DECISION = Path(
    "docs/radial-quote-handoff-relay-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "8fb17aef3599c6ef187d561210b73984c61cc0c067b9c79d7803d350987de5d2"
)
COMMON_WINDOW_POLICY = Path(
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)

SOURCE_PANEL = Path(
    "data/binance_um_book_centroid_btcusdt_2023/"
    "BTCUSDT_um_book_centroid_skew_5m_2023.csv.gz"
)
SOURCE_PANEL_SHA256 = (
    "c4053ce27d28bebda4137349192b1a940360231469f63edc32bacabb2ce54131"
)
SOURCE_MANIFEST = Path(
    "results/binance_um_book_centroid_btcusdt_2023_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "d8237c4562d33c12eff162776f723cc5fc94649b69d26a6230e16fc38c52bba1"
)
SOURCE_BUILDER = Path("training/build_binance_um_book_centroid_2023.py")
SOURCE_BUILDER_SHA256 = (
    "6021a1ee140500350e8b6bc0e8dae5ca32a84db39039c21d809ca798909a5c24"
)
RNCM_PREREGISTRATION = Path(
    "training/preregister_residual_notional_centroid_migration.py"
)
RNCM_PREREGISTRATION_SHA256 = (
    "733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f"
)

GRID_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
GRID_END = datetime(2024, 1, 1, tzinfo=timezone.utc)
BAR = timedelta(minutes=5)
HOLD = timedelta(hours=6)
GRID_ROWS = 105_120
PRIOR_WINDOW = 8_640
PRIOR_MINIMUM = 4_032
NEAR_QUANTILE = (39, 40)
FAR_QUANTILE = (9, 10)
ALGEBRA_TOLERANCE = Decimal("5e-12")
NEAR_ARM_EFFICIENCY = Decimal(3) / Decimal(5)
CONFIRM_EFFICIENCY = Decimal(1) / Decimal(2)
CONTROL_NAMES = (
    "primary",
    "simultaneous_near_far",
    "far_to_near_reverse_relay",
    "no_efficiency_relay",
    "near_only",
    "far_only",
    "one_bar_stale",
    "five_bar_stale",
    "quarter_far_triple_permutation",
    "deterministic_random_side",
    "exact_direction_flip",
    "constant_long",
    "constant_short",
)

RQHR_COLUMNS = (
    "date",
    "skew_2_net",
    "skew_2_path",
    "skew_2_efficiency",
    "skew_3_net",
    "skew_3_path",
    "skew_3_efficiency",
    "skew_4_net",
    "skew_4_path",
    "skew_4_efficiency",
    "skew_5_net",
    "skew_5_path",
    "skew_5_efficiency",
    "source_complete",
    "source_available_at",
)
NUMERIC_COLUMNS = tuple(
    column
    for column in RQHR_COLUMNS
    if column not in {"date", "source_complete", "source_available_at"}
)
NULL_SCENARIOS = (
    "smooth_symmetric",
    "tick_rounded_anchor",
    "stepped_asymmetric",
    "missing_rows",
    "discrete_asymmetric_ladder",
)
RACE_AUDIT_FIELDS = (
    "arms",
    "confirmations",
    "cancellations",
    "ambiguities",
    "incomplete_cancellations",
    "timeouts",
    "terminal_consumed_rows",
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "group": "ccbvfr:primary",
        "path": Path(
            "results/cross_collateral_book_validated_flow_rejection_"
            "event_clock_2026-07-18.json"
        ),
        "sha256": (
            "79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f"
        ),
        "protocol": "CBFR-72 canonical outcome-blind event-clock freeze",
        "closed_flags": {
            "post_entry_outcomes_opened": False,
            "entry_or_later_ohlc_loaded": False,
        },
        "expected_rows": 144,
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "canonical_fields": None,
        "clock_hash": (
            "d2cdcad8f57867722c220e32029d0ccbf1f1aa511e5ae590cf43411a588af4bd"
        ),
        "kind": "embedded_full_dict",
        "producer": Path(
            "training/preregister_cross_collateral_book_validated_flow_rejection.py"
        ),
        "producer_sha256": (
            "004fa71b1951eff58eca592863cf7ad09e0e36e4749a3e611ce299e1ac3d601f"
        ),
    },
    {
        "group": "pdf10:primary",
        "path": Path(
            "results/cross_collateral_liquidity_credibility_fracture_"
            "event_clock_2026-07-14.json"
        ),
        "sha256": (
            "ab8209308619b97880277b95fcc1a2f825b050a603e24b3e2125ddd5bfb226f8"
        ),
        "protocol": "PDF-10 canonical event-clock freeze",
        "closed_flags": {
            "outcomes_opened_for_pdf10": False,
            "price_or_return_loaded": False,
        },
        "expected_rows": 591,
        "selection_end_exclusive": None,
        "canonical_fields": [
            "signal_position",
            "entry_position",
            "exit_position",
            "side",
            "branch",
            "hold_bars",
        ],
        "serialization": (
            "JSON list; each object keys sorted; separators comma/colon; UTF-8"
        ),
        "quarter_boundary_policy": (
            "four quarter-contained non-overlap schedules concatenated in "
            "chronological order"
        ),
        "clock_hash": (
            "ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2"
        ),
        "kind": "replay_pdf10",
        "producer": Path(
            "training/preregister_radial_liquidity_wavefront_cascade.py"
        ),
        "producer_sha256": (
            "9f94706ef05750bc08ce7ef56672512ff7d245a31f830ae1064d1d1c2b02a7a9"
        ),
        "replay_source": Path(
            "training/preregister_cross_collateral_liquidity_credibility_fracture.py"
        ),
        "replay_source_sha256": (
            "8947050c990b5638f6d8b2e952f252289ddef6c92f85fb13f75001fe721e6e28"
        ),
        "dependency": Path(
            "results/cross_collateral_liquidity_credibility_fracture_"
            "support_2026-07-14.json"
        ),
        "dependency_sha256": (
            "9a3001db640ec8041d885645d33f11dd6075276685eb22f8ae3c618363d3099a"
        ),
    },
    {
        "group": "crrc:primary",
        "path": Path(
            "results/cross_venue_radial_refill_compression_"
            "event_clock_2026-07-17.json"
        ),
        "sha256": (
            "09d2ca954c5c4d06b981575c6b0f0e4dc6b49d8a693da418f3f26e5cc454c835"
        ),
        "protocol": "CRRC-72 canonical outcome-blind event-clock freeze",
        "closed_flags": {
            "outcomes_opened": False,
            "price_funding_return_or_equity_loaded": False,
        },
        "expected_rows": 156,
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "canonical_fields": [
            "signal_position",
            "entry_position",
            "exit_position",
            "side",
            "hold_bars",
        ],
        "clock_hash": (
            "81e09e3d1d5592f12ce1994077efa279ebf1de4c29a6f5a144060d16ee6b2e9f"
        ),
        "kind": "embedded_crrc_projection",
        "producer": Path(
            "training/qualify_cross_venue_radial_refill_compression.py"
        ),
        "producer_sha256": (
            "96372733a597ca486b52292480ceacde631056054b2d914aa9180024218fa0e7"
        ),
    },
)

CLOCK_COLUMNS = (
    "control",
    "signal_position",
    "signal_time",
    "entry_time",
    "exit_time",
    "quarter",
    "side",
    "confirmation_age",
)

Number: TypeAlias = Decimal | float


@dataclass(frozen=True)
class Config:
    output: str = "results/radial_quote_handoff_relay_support_2026-07-23.json"
    clock_output: str = (
        "results/radial_quote_handoff_relay_clocks_2026-07-23.csv.gz"
    )
    synthetic_output: str = (
        "results/radial_quote_handoff_relay_synthetic_null_2026-07-23.json"
    )


@dataclass(frozen=True)
class SourceRow:
    position: int
    date: datetime
    available_at: datetime
    complete: bool
    net: tuple[Number, Number, Number, Number]
    path: tuple[Number, Number, Number, Number]
    efficiency: tuple[Number, Number, Number, Number]


@dataclass(frozen=True)
class SignalRow:
    source: SourceRow
    near_sign: int
    far_sign: int
    near_intensity: Number | None
    far_intensity: Number | None
    near_efficiency: Number | None
    far_efficiency: Number | None
    near_threshold: Number | None
    far_threshold: Number | None


@dataclass(frozen=True)
class Candidate:
    control: str
    signal_position: int
    signal_time: datetime
    side: int
    confirmation_age: int


@dataclass(frozen=True)
class Scheduled:
    control: str
    signal_position: int
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    quarter: str
    side: int
    confirmation_age: int


@dataclass(frozen=True)
class ComparatorEvent:
    entry_time: datetime
    exit_time: datetime
    side: int


class ComparatorValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        rows_read: int,
        window_counts: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.rows_read = rows_read
        self.window_counts = dict(window_counts or {})


class Fenwick:
    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        index += 1
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index

    def kth(self, rank: int) -> int:
        total = self.prefix(len(self.tree) - 2)
        if rank < 1 or rank > total:
            raise ValueError("Fenwick rank outside active population")
        index = 0
        bit = 1 << ((len(self.tree) - 1).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and self.tree[candidate] < rank:
                index = candidate
                rank -= self.tree[candidate]
            bit >>= 1
        return index

    def prefix(self, index: int) -> int:
        total = 0
        index += 1
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("RQHR support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RQHR support path escaped repository") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"RQHR {label} hash mismatch: {path}")
    return observed


def _validate_config(cfg: Config) -> None:
    if cfg.synthetic_output != Config.synthetic_output:
        raise RuntimeError("RQHR synthetic artifact path is frozen")
    for field in (cfg.output, cfg.clock_output, cfg.synthetic_output):
        _repository_path(field)


def load_registration() -> dict[str, Any]:
    _verify_hash(PREREGISTRATION, PREREGISTRATION_SHA256, "preregistration")
    payload = json.loads(
        _repository_path(PREREGISTRATION).read_text(encoding="utf-8")
    )
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("RQHR preregistration candidate mismatch")
    if payload.get("protocol_version") != (
        "radial_quote_handoff_relay_preregistration_v2"
    ):
        raise RuntimeError("RQHR preregistration is not authoritative v2")
    if payload.get("artifact_eligible") is not True:
        raise RuntimeError("RQHR preregistration is not artifact eligible")
    if payload.get("mechanism_decision") != {
        "path": str(MECHANISM_DECISION),
        "sha256": MECHANISM_DECISION_SHA256,
    }:
        raise RuntimeError("RQHR preregistration mechanism binding mismatch")
    supersedes = payload.get("supersedes", {})
    if supersedes.get("before_rqhr_incidence") is not True:
        raise RuntimeError("RQHR v1 supersession boundary missing")
    pdf = next(
        (
            row
            for row in payload.get("comparator_bindings", [])
            if row.get("group") == "pdf10:primary"
        ),
        None,
    )
    if not isinstance(pdf, Mapping) or pdf.get("canonical_projection") != (
        "ordered signal_position, entry_position, exit_position, numeric "
        "side, branch, and hold_bars"
    ):
        raise RuntimeError("RQHR PDF comparator projection is not six-field v2")
    if payload.get("manifest_hash") != canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    ):
        raise RuntimeError("RQHR preregistration canonical hash mismatch")
    for field in (
        "rqhr_net_path_efficiency_values_opened",
        "rqhr_features_arms_confirmations_or_events_opened",
        "synthetic_nulls_run",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"RQHR preregistration boundary opened: {field}")
    return payload


def _frozen_dependency_specs() -> dict[str, tuple[Path, str]]:
    return {
        "preregistration": (
            PREREGISTRATION,
            PREREGISTRATION_SHA256,
        ),
        "preregistration_source": (
            PREREGISTRATION_SOURCE,
            PREREGISTRATION_SOURCE_SHA256,
        ),
        "mechanism_decision": (
            MECHANISM_DECISION,
            MECHANISM_DECISION_SHA256,
        ),
        "common_window_policy": (
            COMMON_WINDOW_POLICY,
            COMMON_WINDOW_POLICY_SHA256,
        ),
        "source_panel": (SOURCE_PANEL, SOURCE_PANEL_SHA256),
        "source_manifest": (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        "source_builder": (SOURCE_BUILDER, SOURCE_BUILDER_SHA256),
        "rncm_synthetic_source": (
            RNCM_PREREGISTRATION,
            RNCM_PREREGISTRATION_SHA256,
        ),
    }


def _frozen_dependency_payload() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path),
            "sha256": sha256,
            "read_mode": "raw bytes for SHA-256",
        }
        for name, (path, sha256) in _frozen_dependency_specs().items()
    }


def validate_frozen_dependencies() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (path, expected) in _frozen_dependency_specs().items():
        result[name] = {
            "path": str(path),
            "sha256": _verify_hash(path, expected, name),
            "read_mode": "raw bytes for SHA-256",
        }
    return result


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("empty RQHR timestamp")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid RQHR boolean: {value!r}")


def _decimal(value: str, *, field: str) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid RQHR decimal in {field}") from exc
    if not parsed.is_finite():
        raise ValueError(f"non-finite RQHR decimal in {field}")
    return parsed


def validate_algebra(
    net: Decimal,
    path: Decimal,
    efficiency: Decimal,
    *,
    tolerance: Decimal = ALGEBRA_TOLERANCE,
) -> None:
    if path < 0:
        raise ValueError("RQHR path is negative")
    if path == 0:
        if net != 0 or efficiency != 0:
            raise ValueError("RQHR zero path algebra mismatch")
    if efficiency < 0 or efficiency > 1:
        raise ValueError("RQHR efficiency outside [0,1]")
    if path + tolerance < abs(net):
        raise ValueError("RQHR path is below absolute net")
    if path > 0 and abs(efficiency - abs(net) / path) > tolerance:
        raise ValueError("RQHR efficiency algebra mismatch")


def load_real_source() -> tuple[list[SourceRow], dict[str, Any]]:
    _verify_hash(SOURCE_PANEL, SOURCE_PANEL_SHA256, "source panel")
    frame = pd.read_csv(
        _repository_path(SOURCE_PANEL),
        compression="gzip",
        usecols=list(RQHR_COLUMNS),
        dtype=str,
        keep_default_na=False,
    )
    if set(frame.columns) != set(RQHR_COLUMNS):
        raise RuntimeError("RQHR source projection changed")
    frame = frame.loc[:, list(RQHR_COLUMNS)]
    if len(frame) != GRID_ROWS:
        raise RuntimeError("RQHR source grid row count changed")

    rows: list[SourceRow] = []
    complete_count = 0
    for position, raw in enumerate(frame.itertuples(index=False, name=None)):
        values = dict(zip(RQHR_COLUMNS, raw, strict=True))
        expected_date = GRID_START + position * BAR
        date = _parse_utc(values["date"])
        if date != expected_date:
            raise RuntimeError("RQHR source grid timestamp mismatch")
        available_at = _parse_utc(values["source_available_at"])
        if available_at != date + BAR:
            raise RuntimeError("RQHR source availability mismatch")
        complete = _parse_bool(values["source_complete"])
        if not complete:
            leaked = [
                field
                for field in NUMERIC_COLUMNS
                if values[field].strip().lower() not in {"", "nan", "na", "null"}
            ]
            if leaked:
                raise RuntimeError(
                    f"RQHR incomplete row contains usable values: {leaked}"
                )
            rows.append(
                SourceRow(
                    position=position,
                    date=date,
                    available_at=available_at,
                    complete=False,
                    net=(Decimal(0),) * 4,
                    path=(Decimal(0),) * 4,
                    efficiency=(Decimal(0),) * 4,
                )
            )
            continue
        nets: list[Decimal] = []
        paths: list[Decimal] = []
        efficiencies: list[Decimal] = []
        for radius in range(2, 6):
            net = _decimal(values[f"skew_{radius}_net"], field=f"skew_{radius}_net")
            path = _decimal(
                values[f"skew_{radius}_path"], field=f"skew_{radius}_path"
            )
            efficiency = _decimal(
                values[f"skew_{radius}_efficiency"],
                field=f"skew_{radius}_efficiency",
            )
            validate_algebra(net, path, efficiency)
            nets.append(net)
            paths.append(path)
            efficiencies.append(efficiency)
        complete_count += 1
        rows.append(
            SourceRow(
                position=position,
                date=date,
                available_at=available_at,
                complete=True,
                net=tuple(nets),  # type: ignore[arg-type]
                path=tuple(paths),  # type: ignore[arg-type]
                efficiency=tuple(efficiencies),  # type: ignore[arg-type]
            )
        )
    if rows[-1].date + BAR != GRID_END:
        raise RuntimeError("RQHR source contains a post-2023 or missing terminal row")
    return rows, {
        "rows": len(rows),
        "complete_rows": complete_count,
        "incomplete_rows": len(rows) - complete_count,
        "rqhr_columns_read": list(RQHR_COLUMNS),
        "forbidden_columns_read": [],
        "post_2023_rows_read": 0,
        "market_or_outcome_rows_read": 0,
    }


def _common_sign(first: Number, second: Number) -> int:
    if first > 0 and second > 0:
        return 1
    if first < 0 and second < 0:
        return -1
    return 0


def _intensities(
    rows: Sequence[SourceRow],
) -> tuple[list[Number | None], list[Number | None]]:
    near: list[Number | None] = []
    far: list[Number | None] = []
    for row in rows:
        if not row.complete:
            near.append(None)
            far.append(None)
            continue
        near.append((abs(row.net[0]) + abs(row.net[1])) / 2)
        far.append((abs(row.net[2]) + abs(row.net[3])) / 2)
    return near, far


def strict_prior_thresholds(
    values: Sequence[Number | None],
    *,
    window: int,
    minimum: int,
    quantile: tuple[int, int],
) -> list[Number | None]:
    if not 1 <= minimum <= window:
        raise ValueError("invalid strict-prior window/minimum")
    numerator, denominator = quantile
    if not 0 < numerator <= denominator:
        raise ValueError("invalid nearest-rank quantile")
    unique = sorted({value for value in values if value is not None})
    if not unique:
        return [None] * len(values)
    index = {value: position for position, value in enumerate(unique)}
    tree = Fenwick(len(unique))
    active = 0
    output: list[Number | None] = [None] * len(values)
    for position, value in enumerate(values):
        expired = position - window - 1
        if expired >= 0 and values[expired] is not None:
            tree.add(index[values[expired]], -1)  # type: ignore[index]
            active -= 1
        if active >= minimum:
            rank = (numerator * active + denominator - 1) // denominator
            output[position] = unique[tree.kth(rank)]
        if value is not None:
            tree.add(index[value], 1)
            active += 1
    return output


def derive_signal_rows(
    rows: Sequence[SourceRow],
    *,
    window: int = PRIOR_WINDOW,
    minimum: int = PRIOR_MINIMUM,
) -> list[SignalRow]:
    near_values, far_values = _intensities(rows)
    near_thresholds = strict_prior_thresholds(
        near_values,
        window=window,
        minimum=minimum,
        quantile=NEAR_QUANTILE,
    )
    far_thresholds = strict_prior_thresholds(
        far_values,
        window=window,
        minimum=minimum,
        quantile=FAR_QUANTILE,
    )
    output: list[SignalRow] = []
    for row, near, far, near_threshold, far_threshold in zip(
        rows,
        near_values,
        far_values,
        near_thresholds,
        far_thresholds,
        strict=True,
    ):
        if not row.complete:
            output.append(
                SignalRow(
                    source=row,
                    near_sign=0,
                    far_sign=0,
                    near_intensity=None,
                    far_intensity=None,
                    near_efficiency=None,
                    far_efficiency=None,
                    near_threshold=near_threshold,
                    far_threshold=far_threshold,
                )
            )
            continue
        output.append(
            SignalRow(
                source=row,
                near_sign=_common_sign(row.net[0], row.net[1]),
                far_sign=_common_sign(row.net[2], row.net[3]),
                near_intensity=near,
                far_intensity=far,
                near_efficiency=min(row.efficiency[0], row.efficiency[1]),
                far_efficiency=min(row.efficiency[2], row.efficiency[3]),
                near_threshold=near_threshold,
                far_threshold=far_threshold,
            )
        )
    return output


def _leg_sign(row: SignalRow, leg: str) -> int:
    return row.near_sign if leg == "near" else row.far_sign


def _leg_intensity(row: SignalRow, leg: str) -> Number | None:
    return row.near_intensity if leg == "near" else row.far_intensity


def _leg_efficiency(row: SignalRow, leg: str) -> Number | None:
    return row.near_efficiency if leg == "near" else row.far_efficiency


def _leg_threshold(row: SignalRow, leg: str) -> Number | None:
    return row.near_threshold if leg == "near" else row.far_threshold


def _leg_net_sum(row: SignalRow, leg: str) -> Number:
    if leg == "near":
        return row.source.net[0] + row.source.net[1]
    return row.source.net[2] + row.source.net[3]


def _qualified(
    row: SignalRow,
    *,
    leg: str,
    sign: int | None,
    efficiency_minimum: Number | None,
) -> bool:
    if not row.source.complete:
        return False
    observed_sign = _leg_sign(row, leg)
    if observed_sign == 0 or (sign is not None and observed_sign != sign):
        return False
    intensity = _leg_intensity(row, leg)
    threshold = _leg_threshold(row, leg)
    efficiency = _leg_efficiency(row, leg)
    if intensity is None or threshold is None or intensity < threshold:
        return False
    if (
        efficiency_minimum is not None
        and (efficiency is None or efficiency < efficiency_minimum)
    ):
        return False
    return True


def _crossing(
    rows: Sequence[SignalRow],
    position: int,
    *,
    leg: str,
    efficiency_minimum: Number | None,
) -> bool:
    if position <= 0:
        return False
    current = rows[position]
    previous = rows[position - 1]
    if not _qualified(
        current,
        leg=leg,
        sign=None,
        efficiency_minimum=efficiency_minimum,
    ):
        return False
    previous_intensity = _leg_intensity(previous, leg)
    previous_threshold = _leg_threshold(previous, leg)
    return bool(
        previous.source.complete
        and previous_intensity is not None
        and previous_threshold is not None
        and previous_intensity < previous_threshold
    )


def relay_candidates(
    rows: Sequence[SignalRow],
    *,
    control: str,
    arm_leg: str = "near",
    confirmation_leg: str = "far",
    use_efficiency: bool = True,
) -> tuple[list[Candidate], dict[str, int]]:
    if arm_leg == confirmation_leg or {arm_leg, confirmation_leg} != {
        "near",
        "far",
    }:
        raise ValueError("RQHR relay legs must be near/far opposites")
    arm_efficiency: Number | None = (
        NEAR_ARM_EFFICIENCY if use_efficiency else None
    )
    confirmation_efficiency: Number | None = (
        CONFIRM_EFFICIENCY if use_efficiency else None
    )
    active_position: int | None = None
    active_sign = 0
    cumulative: Number | None = None
    candidates: list[Candidate] = []
    audit = {
        "arms": 0,
        "confirmations": 0,
        "cancellations": 0,
        "ambiguities": 0,
        "incomplete_cancellations": 0,
        "timeouts": 0,
        "terminal_consumed_rows": 0,
    }

    for position, row in enumerate(rows):
        terminal_consumed = False
        if active_position is not None:
            age = position - active_position
            if not row.source.complete:
                audit["incomplete_cancellations"] += 1
                terminal_consumed = True
            else:
                if cumulative is None:
                    raise RuntimeError("RQHR active race lost cumulative state")
                cumulative += _leg_net_sum(row, arm_leg)
                confirmation = _qualified(
                    row,
                    leg=confirmation_leg,
                    sign=active_sign,
                    efficiency_minimum=confirmation_efficiency,
                ) and (
                    (active_sign > 0 and cumulative > 0)
                    or (active_sign < 0 and cumulative < 0)
                )
                cancellation_efficiency = _leg_efficiency(row, arm_leg)
                cancellation = (
                    _leg_sign(row, arm_leg) == -active_sign
                    and (
                        not use_efficiency
                        or (
                            cancellation_efficiency is not None
                            and cancellation_efficiency >= CONFIRM_EFFICIENCY
                        )
                    )
                )
                if confirmation and cancellation:
                    audit["ambiguities"] += 1
                    terminal_consumed = True
                elif confirmation:
                    candidates.append(
                        Candidate(
                            control=control,
                            signal_position=position,
                            signal_time=row.source.available_at,
                            side=active_sign,
                            confirmation_age=age,
                        )
                    )
                    audit["confirmations"] += 1
                    terminal_consumed = True
                elif cancellation:
                    audit["cancellations"] += 1
                    terminal_consumed = True
                elif age >= 6:
                    audit["timeouts"] += 1
                    terminal_consumed = True
            if terminal_consumed:
                active_position = None
                active_sign = 0
                cumulative = None
                audit["terminal_consumed_rows"] += 1
                continue

        if active_position is not None or not row.source.complete:
            continue
        if not _crossing(
            rows,
            position,
            leg=arm_leg,
            efficiency_minimum=arm_efficiency,
        ):
            continue
        sign = _leg_sign(row, arm_leg)
        if _qualified(
            row,
            leg=confirmation_leg,
            sign=sign,
            efficiency_minimum=confirmation_efficiency,
        ):
            continue
        active_position = position
        active_sign = sign
        cumulative = _leg_net_sum(row, arm_leg)
        audit["arms"] += 1
    return candidates, audit


def immediate_candidates(
    rows: Sequence[SignalRow],
    *,
    control: str,
    mode: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for position, row in enumerate(rows):
        if mode in {"near_only", "simultaneous_near_far"}:
            if not _crossing(
                rows,
                position,
                leg="near",
                efficiency_minimum=NEAR_ARM_EFFICIENCY,
            ):
                continue
            side = row.near_sign
            if mode == "simultaneous_near_far" and not _qualified(
                row,
                leg="far",
                sign=side,
                efficiency_minimum=CONFIRM_EFFICIENCY,
            ):
                continue
        elif mode == "far_only":
            if not _crossing(
                rows,
                position,
                leg="far",
                efficiency_minimum=CONFIRM_EFFICIENCY,
            ):
                continue
            side = row.far_sign
        else:
            raise ValueError(f"unknown RQHR immediate control: {mode}")
        candidates.append(
            Candidate(
                control=control,
                signal_position=position,
                signal_time=row.source.available_at,
                side=side,
                confirmation_age=0,
            )
        )
    return candidates


def stale_candidates(
    primary: Sequence[Candidate],
    rows: Sequence[SignalRow],
    *,
    lag: int,
    control: str,
) -> list[Candidate]:
    output: list[Candidate] = []
    for candidate in primary:
        destination = candidate.signal_position + lag
        if destination >= len(rows) or not rows[destination].source.complete:
            continue
        output.append(
            Candidate(
                control=control,
                signal_position=destination,
                signal_time=rows[destination].source.available_at,
                side=candidate.side,
                confirmation_age=candidate.confirmation_age,
            )
        )
    return output


def _quarter(value: datetime) -> str:
    return f"{value.year}-Q{((value.month - 1) // 3) + 1}"


def _same_quarter(*values: datetime) -> bool:
    return len({_quarter(value) for value in values}) == 1


def schedule(control: str, candidates: Sequence[Candidate]) -> list[Scheduled]:
    accepted: list[Scheduled] = []
    reserved_until: dict[str, datetime] = {}
    for candidate in sorted(
        candidates,
        key=lambda row: (row.signal_time, row.signal_position, row.side),
    ):
        entry = candidate.signal_time + BAR
        exit_time = entry + HOLD
        if not (
            GRID_START <= candidate.signal_time < GRID_END
            and GRID_START <= entry < GRID_END
            and exit_time <= GRID_END
        ):
            continue
        if not _same_quarter(candidate.signal_time, entry, exit_time):
            continue
        quarter = _quarter(entry)
        if entry < reserved_until.get(quarter, entry):
            continue
        accepted.append(
            Scheduled(
                control=control,
                signal_position=candidate.signal_position,
                signal_time=candidate.signal_time,
                entry_time=entry,
                exit_time=exit_time,
                quarter=quarter,
                side=candidate.side,
                confirmation_age=candidate.confirmation_age,
            )
        )
        reserved_until[quarter] = exit_time
    return accepted


def _random_side(entry_time: datetime) -> int:
    digest = hashlib.sha256(
        f"RQHR-72|deterministic_random_side|{entry_time.isoformat()}".encode()
    ).digest()
    return 1 if digest[0] % 2 == 0 else -1


def side_control(
    primary: Sequence[Scheduled],
    *,
    control: str,
) -> list[Scheduled]:
    output: list[Scheduled] = []
    for row in primary:
        if control == "deterministic_random_side":
            side = _random_side(row.entry_time)
        elif control == "exact_direction_flip":
            side = -row.side
        elif control == "constant_long":
            side = 1
        elif control == "constant_short":
            side = -1
        else:
            raise ValueError(f"unknown RQHR side control: {control}")
        output.append(
            Scheduled(
                control=control,
                signal_position=row.signal_position,
                signal_time=row.signal_time,
                entry_time=row.entry_time,
                exit_time=row.exit_time,
                quarter=row.quarter,
                side=side,
                confirmation_age=row.confirmation_age,
            )
        )
    return output


def permute_far_tuples(rows: Sequence[SourceRow]) -> list[SourceRow]:
    output = list(rows)
    by_quarter: dict[str, list[int]] = {}
    for row in rows:
        if row.complete:
            by_quarter.setdefault(_quarter(row.date), []).append(row.position)
    for quarter, recipients in sorted(by_quarter.items()):
        donors = sorted(
            recipients,
            key=lambda position: hashlib.sha256(
                (
                    "RQHR-72|quarter_far_triple_permutation|"
                    f"{quarter}|{rows[position].date.isoformat()}"
                ).encode()
            ).digest(),
        )
        for recipient, donor in zip(recipients, donors, strict=True):
            target = rows[recipient]
            source = rows[donor]
            output[recipient] = SourceRow(
                position=target.position,
                date=target.date,
                available_at=target.available_at,
                complete=True,
                net=(
                    target.net[0],
                    target.net[1],
                    source.net[2],
                    source.net[3],
                ),
                path=(
                    target.path[0],
                    target.path[1],
                    source.path[2],
                    source.path[3],
                ),
                efficiency=(
                    target.efficiency[0],
                    target.efficiency[1],
                    source.efficiency[2],
                    source.efficiency[3],
                ),
            )
    return output


def build_clocks(
    source_rows: Sequence[SourceRow],
    *,
    window: int = PRIOR_WINDOW,
    minimum: int = PRIOR_MINIMUM,
) -> tuple[
    dict[str, list[Scheduled]],
    dict[str, int],
    dict[str, dict[str, int]],
]:
    signals = derive_signal_rows(source_rows, window=window, minimum=minimum)
    primary_candidates, primary_audit = relay_candidates(
        signals,
        control="primary",
    )
    clocks: dict[str, list[Scheduled]] = {
        "primary": schedule("primary", primary_candidates)
    }
    raw_counts = {"primary": len(primary_candidates)}
    race_audits = {"primary": primary_audit}

    simultaneous = immediate_candidates(
        signals,
        control="simultaneous_near_far",
        mode="simultaneous_near_far",
    )
    near_only = immediate_candidates(
        signals,
        control="near_only",
        mode="near_only",
    )
    far_only = immediate_candidates(
        signals,
        control="far_only",
        mode="far_only",
    )
    reverse, reverse_audit = relay_candidates(
        signals,
        control="far_to_near_reverse_relay",
        arm_leg="far",
        confirmation_leg="near",
    )
    no_efficiency, no_efficiency_audit = relay_candidates(
        signals,
        control="no_efficiency_relay",
        use_efficiency=False,
    )
    candidates_by_control = {
        "simultaneous_near_far": simultaneous,
        "far_to_near_reverse_relay": reverse,
        "no_efficiency_relay": no_efficiency,
        "near_only": near_only,
        "far_only": far_only,
        "one_bar_stale": stale_candidates(
            primary_candidates,
            signals,
            lag=1,
            control="one_bar_stale",
        ),
        "five_bar_stale": stale_candidates(
            primary_candidates,
            signals,
            lag=5,
            control="five_bar_stale",
        ),
    }
    race_audits["far_to_near_reverse_relay"] = reverse_audit
    race_audits["no_efficiency_relay"] = no_efficiency_audit
    for control, candidates in candidates_by_control.items():
        raw_counts[control] = len(candidates)
        clocks[control] = schedule(control, candidates)

    permuted_signals = derive_signal_rows(
        permute_far_tuples(source_rows),
        window=window,
        minimum=minimum,
    )
    permutation, permutation_audit = relay_candidates(
        permuted_signals,
        control="quarter_far_triple_permutation",
    )
    raw_counts["quarter_far_triple_permutation"] = len(permutation)
    race_audits["quarter_far_triple_permutation"] = permutation_audit
    clocks["quarter_far_triple_permutation"] = schedule(
        "quarter_far_triple_permutation",
        permutation,
    )

    for control in (
        "deterministic_random_side",
        "exact_direction_flip",
        "constant_long",
        "constant_short",
    ):
        clocks[control] = side_control(clocks["primary"], control=control)
        raw_counts[control] = len(clocks[control])
    if tuple(clocks) != CONTROL_NAMES:
        raise RuntimeError("RQHR control cohort drift")
    return clocks, raw_counts, race_audits


def _synthetic_average_quotes(
    scenario: str,
    positions: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    smooth_anchor = 100.0 * (
        1.0
        + 0.0012 * np.sin(2.0 * np.pi * positions / 288.0)
        + 0.0005 * np.sin(2.0 * np.pi * positions / (288.0 * 7.0))
        + 0.0002 * np.sin(2.0 * np.pi * positions / (288.0 * 29.0))
    )
    if scenario in {"smooth_symmetric", "missing_rows"}:
        anchor = smooth_anchor
        bid_best, ask_best = 99.8, 100.2
    elif scenario == "tick_rounded_anchor":
        anchor = np.round(smooth_anchor, 2)
        bid_best, ask_best = 99.8, 100.2
    elif scenario == "stepped_asymmetric":
        phase = (positions % (288.0 * 5.0)) / (288.0 * 5.0)
        anchor = np.round(100.0 + 0.15 * (2.0 * phase - 1.0), 2)
        bid_best, ask_best = 99.75, 100.25
    elif scenario == "discrete_asymmetric_ladder":
        anchor = 100.0 * (
            1.0
            + 0.0011 * np.sin(2.0 * np.pi * positions / 288.0)
            + 0.0004 * np.sin(
                2.0 * np.pi * positions / (288.0 * 11.0)
            )
        )
        anchor = np.round(anchor, 2)
        bid_price = np.arange(94.0, 99.8001, 0.002)
        ask_price = np.arange(100.2, 106.0001, 0.002)
        bid_quantity = (
            1.2
            + 0.20 * np.sin(7.0 * bid_price)
            + 0.08 * np.cos(19.0 * bid_price)
        )
        ask_quantity = (
            1.1
            + 0.18 * np.cos(5.0 * ask_price)
            + 0.07 * np.sin(23.0 * ask_price)
        )
        bid_depth_suffix = np.cumsum(bid_quantity[::-1])[::-1]
        bid_notional_suffix = np.cumsum(
            (bid_quantity * bid_price)[::-1]
        )[::-1]
        ask_depth_prefix = np.cumsum(ask_quantity)
        ask_notional_prefix = np.cumsum(ask_quantity * ask_price)
        average: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for distance in range(1, 6):
            bid_index = np.searchsorted(
                bid_price,
                anchor * (1.0 - distance / 100.0),
                side="left",
            )
            ask_index = (
                np.searchsorted(
                    ask_price,
                    anchor * (1.0 + distance / 100.0),
                    side="right",
                )
                - 1
            )
            average[distance] = (
                bid_notional_suffix[bid_index] / bid_depth_suffix[bid_index],
                ask_notional_prefix[ask_index] / ask_depth_prefix[ask_index],
            )
        return average
    else:
        raise ValueError(f"unknown RQHR synthetic scenario: {scenario}")

    average = {}
    for distance in range(1, 6):
        bid_lower = anchor * (1.0 - distance / 100.0)
        ask_upper = anchor * (1.0 + distance / 100.0)
        average[distance] = (
            0.5 * (bid_lower + bid_best),
            0.5 * (ask_best + ask_upper),
        )
    return average


def aggregate_snapshot_skews(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("RQHR synthetic bars require exactly ten snapshots")
    net = values[:, -1] - values[:, 0]
    path = np.abs(np.diff(values, axis=1)).sum(axis=1)
    efficiency = np.divide(
        np.abs(net),
        path,
        out=np.zeros_like(path),
        where=path != 0.0,
    )
    return net, path, efficiency


def synthetic_source_rows(
    scenario: str,
    *,
    rows: int = GRID_ROWS,
) -> list[SourceRow]:
    if scenario not in NULL_SCENARIOS:
        raise ValueError(f"unknown RQHR synthetic scenario: {scenario}")
    if rows <= 0:
        raise ValueError("RQHR synthetic row count must be positive")
    positions = np.arange(rows * 10, dtype=np.float64) / 10.0
    average = _synthetic_average_quotes(scenario, positions)
    inner_bid, inner_ask = average[1]
    aggregates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for distance in range(2, 6):
        bid, ask = average[distance]
        skew = np.log(ask / inner_ask) - np.log(inner_bid / bid)
        aggregates.append(aggregate_snapshot_skews(skew.reshape(rows, 10)))
    complete = np.ones(rows, dtype=bool)
    if scenario == "missing_rows":
        complete[(np.arange(rows, dtype=np.int64) % 1_009) < 3] = False

    output: list[SourceRow] = []
    zero = (0.0, 0.0, 0.0, 0.0)
    for position in range(rows):
        date = GRID_START + position * BAR
        if not complete[position]:
            output.append(
                SourceRow(
                    position=position,
                    date=date,
                    available_at=date + BAR,
                    complete=False,
                    net=zero,
                    path=zero,
                    efficiency=zero,
                )
            )
            continue
        output.append(
            SourceRow(
                position=position,
                date=date,
                available_at=date + BAR,
                complete=True,
                net=tuple(
                    float(aggregate[0][position]) for aggregate in aggregates
                ),  # type: ignore[arg-type]
                path=tuple(
                    float(aggregate[1][position]) for aggregate in aggregates
                ),  # type: ignore[arg-type]
                efficiency=tuple(
                    float(aggregate[2][position]) for aggregate in aggregates
                ),  # type: ignore[arg-type]
            )
        )
    return output


def synthetic_null_report(
    *,
    rows: int = GRID_ROWS,
    window: int = PRIOR_WINDOW,
    minimum: int = PRIOR_MINIMUM,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    passed = True
    for scenario in NULL_SCENARIOS:
        source = synthetic_source_rows(scenario, rows=rows)
        signals = derive_signal_rows(source, window=window, minimum=minimum)
        candidates, audit = relay_candidates(signals, control="primary")
        accepted = schedule("primary", candidates)
        scenario_passed = len(candidates) == 0 and len(accepted) == 0
        scenarios[scenario] = {
            "grid_rows": rows,
            "complete_rows": sum(row.complete for row in source),
            "scheduled_snapshot_slots": rows * 10,
            "raw_confirmations": len(candidates),
            "accepted_events": len(accepted),
            "race_audit": audit,
            "passed": scenario_passed,
        }
        passed &= scenario_passed
    return {
        "protocol": "RQHR-72 hash-bound synthetic moving-band null",
        "candidate": POLICY_ID,
        "source_rows_read": 0,
        "real_rqhr_columns_read": 0,
        "comparator_rows_read": 0,
        "market_or_outcome_rows_read": 0,
        "parameters": {
            "grid_rows": rows,
            "snapshots_per_non_suppressed_bar": 10,
            "strict_prior_window_rows": window,
            "strict_prior_minimum_values": minimum,
            "near_quantile": "39/40",
            "far_quantile": "9/10",
        },
        "scenarios": scenarios,
        "passed": passed,
        "failure_action": (
            "reject before real source"
            if not passed
            else "real source may be opened only by frozen support builder"
        ),
    }


def _clock_valid(control: str, rows: Sequence[Scheduled]) -> bool:
    ordered = sorted(rows, key=lambda row: row.entry_time)
    if list(rows) != ordered:
        return False
    if len({row.entry_time for row in rows}) != len(rows):
        return False
    prior_exit_by_quarter: dict[str, datetime] = {}
    for row in rows:
        if row.control != control or row.side not in {-1, 1}:
            return False
        if control in {
            "simultaneous_near_far",
            "near_only",
            "far_only",
        }:
            if row.confirmation_age != 0:
                return False
        elif not 1 <= row.confirmation_age <= 6:
            return False
        if not 0 <= row.signal_position < GRID_ROWS:
            return False
        if row.signal_time != GRID_START + row.signal_position * BAR + BAR:
            return False
        if not (
            GRID_START <= row.signal_time < GRID_END
            and GRID_START <= row.entry_time < GRID_END
            and row.exit_time <= GRID_END
        ):
            return False
        if row.entry_time != row.signal_time + BAR:
            return False
        if row.exit_time != row.entry_time + HOLD:
            return False
        if row.quarter != _quarter(row.entry_time):
            return False
        if not _same_quarter(row.signal_time, row.entry_time, row.exit_time):
            return False
        if row.entry_time < prior_exit_by_quarter.get(row.quarter, row.entry_time):
            return False
        prior_exit_by_quarter[row.quarter] = row.exit_time
    return True


def _counts_by(
    rows: Sequence[Scheduled],
    key: Any,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for row in rows:
        label = str(key(row))
        output[label] = output.get(label, 0) + 1
    return dict(sorted(output.items()))


def source_support_summary(
    primary: Sequence[Scheduled],
    *,
    clocks: Mapping[str, Sequence[Scheduled]],
) -> dict[str, Any]:
    total = len(primary)
    h1_end = datetime(2023, 7, 1, tzinfo=timezone.utc)
    halves = {
        "h1": sum(row.entry_time < h1_end for row in primary),
        "h2": sum(row.entry_time >= h1_end for row in primary),
    }
    quarters = _counts_by(primary, lambda row: row.quarter)
    for quarter in ("2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"):
        quarters.setdefault(quarter, 0)
    quarters = dict(sorted(quarters.items()))
    months = _counts_by(
        primary,
        lambda row: f"{row.entry_time.year:04d}-{row.entry_time.month:02d}",
    )
    longs = sum(row.side > 0 for row in primary)
    shorts = total - longs
    ordered_entries = sorted(row.entry_time for row in primary)
    gaps = [
        (current - previous).total_seconds() / 86_400.0
        for previous, current in zip(ordered_entries, ordered_entries[1:])
    ]
    max_gap = max(gaps, default=None)
    age_ge_2 = sum(row.confirmation_age >= 2 for row in primary)
    age_by_half = {
        "h1": sum(
            row.entry_time < h1_end and row.confirmation_age >= 2
            for row in primary
        ),
        "h2": sum(
            row.entry_time >= h1_end and row.confirmation_age >= 2
            for row in primary
        ),
    }
    expected_quarters = {"2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"}
    checks = {
        "total": total >= 120,
        "each_half": all(value >= 45 for value in halves.values()),
        "each_quarter": all(value >= 20 for value in quarters.values()),
        "each_side_share": (
            total > 0
            and longs * 20 >= total * 7
            and shorts * 20 >= total * 7
        ),
        "quarter_concentration": (
            total > 0
            and max(quarters.values(), default=0) * 5 <= total * 2
        ),
        "month_concentration": (
            total > 0
            and max(months.values(), default=0) * 20 <= total * 3
        ),
        "maximum_gap": max_gap is not None and max_gap <= 21.0,
        "age_ge_2_total": total > 0 and age_ge_2 * 5 >= total,
        "age_ge_2_each_half": all(
            halves[name] > 0 and age_by_half[name] * 10 >= halves[name]
            for name in halves
        ),
        "exact_2023_window": (
            set(quarters) == expected_quarters
            and all(
                GRID_START <= row.signal_time < GRID_END
                and GRID_START <= row.entry_time < GRID_END
                and row.exit_time <= GRID_END
                for row in primary
            )
        ),
        "all_clocks_valid": (
            tuple(clocks) == CONTROL_NAMES
            and all(_clock_valid(name, clocks[name]) for name in CONTROL_NAMES)
        ),
    }
    return {
        "accepted_events": total,
        "long_events": longs,
        "short_events": shorts,
        "long_share": longs / total if total else 0.0,
        "short_share": shorts / total if total else 0.0,
        "half_counts": halves,
        "quarter_counts": quarters,
        "month_counts": months,
        "maximum_entry_gap_elapsed_days": max_gap,
        "age_ge_2_total": age_ge_2,
        "age_ge_2_by_half": age_by_half,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _canonical_clock_rows(
    clocks: Mapping[str, Sequence[Scheduled]],
) -> list[dict[str, Any]]:
    return [
        {
            "control": row.control,
            "signal_position": row.signal_position,
            "signal_time": row.signal_time.isoformat(),
            "entry_time": row.entry_time.isoformat(),
            "exit_time": row.exit_time.isoformat(),
            "quarter": row.quarter,
            "side": row.side,
            "confirmation_age": row.confirmation_age,
        }
        for control in CONTROL_NAMES
        for row in clocks[control]
    ]


def clock_hash(clocks: Mapping[str, Sequence[Scheduled]]) -> str:
    return canonical_hash(_canonical_clock_rows(clocks))


def clock_payload(clocks: Mapping[str, Sequence[Scheduled]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CLOCK_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(_canonical_clock_rows(clocks))
    return gzip.compress(buffer.getvalue().encode(), compresslevel=9, mtime=0)


def _required_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid comparator integer: {field}")
    return value


def _display_timestamp(value: Any, expected: datetime, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"invalid comparator display timestamp: {field}")
    rendered = expected.astimezone(timezone.utc).replace(tzinfo=None).isoformat(
        sep=" "
    )
    if value != rendered:
        raise ValueError(f"comparator position/date mismatch: {field}")


def _short_quarter(value: datetime) -> str:
    return f"q{((value.month - 1) // 3) + 1}"


def _comparator_event(raw: Mapping[str, Any]) -> ComparatorEvent:
    signal_position = _required_int(raw, "signal_position")
    entry_position = _required_int(raw, "entry_position")
    exit_position = _required_int(raw, "exit_position")
    side = _required_int(raw, "side")
    hold_bars = _required_int(raw, "hold_bars")
    if min(signal_position, entry_position, exit_position) < 0:
        raise ValueError("negative comparator position")
    if side not in {-1, 1}:
        raise ValueError("invalid comparator side")
    if hold_bars <= 0 or exit_position - entry_position != hold_bars:
        raise ValueError("comparator hold interval mismatch")
    if entry_position < signal_position or exit_position <= entry_position:
        raise ValueError("invalid comparator interval ordering")
    signal = GRID_START + signal_position * BAR
    entry = GRID_START + entry_position * BAR
    exit_time = GRID_START + exit_position * BAR
    _display_timestamp(raw.get("signal_date"), signal, "signal_date")
    _display_timestamp(raw.get("entry_date"), entry, "entry_date")
    _display_timestamp(raw.get("exit_date"), exit_time, "exit_date")
    if raw.get("quarter") != _short_quarter(entry):
        raise ValueError("comparator quarter label mismatch")
    if not _same_quarter(signal, entry, exit_time):
        raise ValueError("comparator interval is not quarter contained")
    return ComparatorEvent(entry_time=entry, exit_time=exit_time, side=side)


def _validate_comparator_group(
    group: str,
    events: Sequence[ComparatorEvent],
) -> None:
    if not events:
        raise ValueError(f"empty comparator group: {group}")
    if list(events) != sorted(events, key=lambda row: row.entry_time):
        raise ValueError(f"unordered comparator group: {group}")
    if len({row.entry_time for row in events}) != len(events):
        raise ValueError(f"duplicate comparator entry: {group}")
    if any(
        current.entry_time < previous.exit_time
        for previous, current in zip(events, events[1:])
    ):
        raise ValueError(f"overlapping comparator group: {group}")


def _validate_comparator_header(
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    if payload.get("protocol") != spec["protocol"]:
        raise ValueError(f"comparator protocol mismatch: {spec['group']}")
    for field, expected in spec["closed_flags"].items():
        if payload.get(field) is not expected:
            raise ValueError(
                f"comparator outcome boundary opened: {spec['group']}:{field}"
            )
    if payload.get("event_count") != spec["expected_rows"]:
        raise ValueError(f"comparator event count mismatch: {spec['group']}")
    if payload.get("event_clock_sha256") != spec["clock_hash"]:
        raise ValueError(f"comparator declared clock hash mismatch: {spec['group']}")
    if spec.get("selection_end_exclusive") is not None and payload.get(
        "selection_end_exclusive"
    ) != spec["selection_end_exclusive"]:
        raise ValueError(f"comparator selection boundary mismatch: {spec['group']}")
    if spec.get("canonical_fields") is not None and payload.get(
        "canonical_fields"
    ) != spec["canonical_fields"]:
        raise ValueError(
            f"comparator canonical field declaration mismatch: {spec['group']}"
        )
    if spec.get("serialization") is not None and payload.get(
        "serialization"
    ) != spec["serialization"]:
        raise ValueError(
            f"comparator serialization declaration mismatch: {spec['group']}"
        )
    if spec.get("quarter_boundary_policy") is not None and payload.get(
        "quarter_boundary_policy"
    ) != spec["quarter_boundary_policy"]:
        raise ValueError(
            f"comparator quarter policy declaration mismatch: {spec['group']}"
        )


def _embedded_comparator(
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[ComparatorEvent]:
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != spec["expected_rows"]:
        raise ValueError(f"comparator embedded rows missing: {spec['group']}")
    if spec["kind"] == "embedded_full_dict":
        observed_hash = canonical_hash(events)
    elif spec["kind"] == "embedded_crrc_projection":
        projection = [
            {
                "signal_position": _required_int(raw, "signal_position"),
                "entry_position": _required_int(raw, "entry_position"),
                "exit_position": _required_int(raw, "exit_position"),
                "side": _required_int(raw, "side"),
                "hold_bars": _required_int(raw, "hold_bars"),
            }
            for raw in events
        ]
        observed_hash = canonical_hash(projection)
    else:
        raise ValueError(f"invalid embedded comparator kind: {spec['kind']}")
    if observed_hash != spec["clock_hash"]:
        raise ValueError(f"comparator canonical hash mismatch: {spec['group']}")
    return [_comparator_event(raw) for raw in events]


def _pdf_projection(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "signal_position": _required_int(raw, "signal_position"),
            "entry_position": _required_int(raw, "entry_position"),
            "exit_position": _required_int(raw, "exit_position"),
            "side": _required_int(raw, "side"),
            "branch": str(raw["branch"]),
            "hold_bars": _required_int(raw, "hold_bars"),
        }
        for raw in records
    ]


def _replay_pdf_comparator(
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[ComparatorEvent]:
    from training import (
        preregister_cross_collateral_liquidity_credibility_fracture as pdf,
    )

    dependency = json.loads(
        _repository_path(spec["dependency"]).read_text(encoding="utf-8")
    )
    protocol = dependency.get("protocol", {})
    if protocol.get("outcomes_opened_for_pdf10") is not False:
        raise ValueError("PDF comparator support opened outcomes")
    if protocol.get("selection_end_exclusive") != "2024-01-01 00:00:00":
        raise ValueError("PDF comparator support is not sealed to 2023")
    if payload.get("preregistration_result_sha256") != spec["dependency_sha256"]:
        raise ValueError("PDF comparator does not bind support artifact")

    cfg = pdf.Config()
    pdf._validate_frozen_config(cfg)
    frame, _ = pdf.load_credibility(cfg)
    signal = pdf.build_signal(frame, cfg)
    schedule = pdf._quarterly_schedule(signal, frame)
    records: list[dict[str, Any]] = []
    for row in schedule.itertuples(index=False):
        records.append(
            {
                "quarter": str(row.quarter),
                "signal_position": int(row.signal_position),
                "entry_position": int(row.entry_position),
                "exit_position": int(row.exit_position),
                "signal_date": str(row.signal_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "branch": str(row.branch),
                "hold_bars": int(row.hold_bars),
            }
        )
    if len(records) != spec["expected_rows"]:
        raise ValueError("PDF comparator replay count mismatch")
    if canonical_hash(_pdf_projection(records)) != spec["clock_hash"]:
        raise ValueError("PDF comparator replay canonical hash mismatch")
    return [_comparator_event(raw) for raw in records]


def _window_bucket(event: ComparatorEvent) -> str:
    if event.entry_time >= GRID_START and event.exit_time <= GRID_END:
        return "fully_contained_rows_used"
    if event.exit_time <= GRID_START:
        return "rows_before_window"
    if event.entry_time >= GRID_END:
        return "rows_after_window"
    return "rows_crossing_boundary"


def _window_filter(
    group: str,
    events: Sequence[ComparatorEvent],
) -> tuple[list[ComparatorEvent], dict[str, int]]:
    counts = {
        "total_raw_rows_parsed": len(events),
        "fully_contained_rows_used": 0,
        "rows_before_window": 0,
        "rows_after_window": 0,
        "rows_crossing_boundary": 0,
    }
    contained: list[ComparatorEvent] = []
    for event in events:
        bucket = _window_bucket(event)
        counts[bucket] += 1
        if bucket == "fully_contained_rows_used":
            contained.append(event)
    if len(contained) < 10:
        raise ValueError(f"comparator group has fewer than ten contained rows: {group}")
    return contained, counts


def load_comparator_groups() -> tuple[
    dict[str, list[ComparatorEvent]],
    int,
    dict[str, dict[str, int]],
]:
    groups: dict[str, list[ComparatorEvent]] = {}
    window_counts: dict[str, dict[str, int]] = {}
    rows_read = 0
    try:
        for spec in COMPARATOR_SPECS:
            _verify_hash(spec["path"], spec["sha256"], f"{spec['group']} artifact")
            _verify_hash(
                spec["producer"],
                spec["producer_sha256"],
                f"{spec['group']} producer",
            )
            if "replay_source" in spec:
                _verify_hash(
                    spec["replay_source"],
                    spec["replay_source_sha256"],
                    f"{spec['group']} replay source",
                )
            if "dependency" in spec:
                _verify_hash(
                    spec["dependency"],
                    spec["dependency_sha256"],
                    f"{spec['group']} dependency",
                )
            payload = json.loads(
                _repository_path(spec["path"]).read_text(encoding="utf-8")
            )
            embedded = payload.get("events")
            if spec["kind"] != "replay_pdf10" and isinstance(embedded, list):
                rows_read += len(embedded)
            _validate_comparator_header(payload, spec)
            if spec["kind"] == "replay_pdf10":
                events = _replay_pdf_comparator(payload, spec)
                rows_read += len(events)
            else:
                events = _embedded_comparator(payload, spec)
            _validate_comparator_group(spec["group"], events)
            contained, counts = _window_filter(spec["group"], events)
            groups[spec["group"]] = contained
            window_counts[spec["group"]] = counts
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ComparatorValidationError(
            str(exc),
            rows_read=rows_read,
            window_counts=window_counts,
        ) from exc
    return groups, rows_read, window_counts


def one_to_one_matches(
    left: Sequence[datetime],
    right: Sequence[datetime],
    tolerance: timedelta,
) -> int:
    left = sorted(left)
    right = sorted(right)
    i = j = matches = 0
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < timedelta(0):
            i += 1
        else:
            j += 1
    return matches


def _exposure(events: Sequence[ComparatorEvent]) -> np.ndarray:
    values = np.zeros(GRID_ROWS, dtype=np.float64)
    for event in events:
        if event.entry_time < GRID_START or event.exit_time > GRID_END:
            raise ValueError("novelty received a non-contained interval")
        begin = int((event.entry_time - GRID_START) / BAR)
        finish = int((event.exit_time - GRID_START) / BAR)
        values[begin:finish] = event.side
    return values


def novelty_metrics(
    primary: Sequence[Scheduled],
    comparator: Sequence[ComparatorEvent],
) -> dict[str, Any]:
    candidate = [
        ComparatorEvent(row.entry_time, row.exit_time, row.side)
        for row in primary
        if row.entry_time >= GRID_START and row.exit_time <= GRID_END
    ]
    left = [row.entry_time for row in candidate]
    right = [row.entry_time for row in comparator]
    exact_left, exact_right = set(left), set(right)
    union = exact_left | exact_right
    matches = one_to_one_matches(left, right, 12 * BAR)
    left_exposure = _exposure(candidate)
    right_exposure = _exposure(comparator)
    correlation: float | None
    if np.std(left_exposure) == 0 or np.std(right_exposure) == 0:
        correlation = None
    else:
        value = float(np.corrcoef(left_exposure, right_exposure)[0, 1])
        correlation = value if math.isfinite(value) else None
    return {
        "candidate_entries": len(left),
        "comparator_entries": len(right),
        "exact_entry_intersection": len(exact_left & exact_right),
        "exact_entry_jaccard": (
            len(exact_left & exact_right) / len(union) if union else 0.0
        ),
        "one_to_one_matches_within_12_bars": matches,
        "candidate_containment": matches / len(left) if left else 0.0,
        "comparator_containment": matches / len(right) if right else 0.0,
        "signed_5m_occupied_exposure_correlation": correlation,
    }


def candidate_window_counts(primary: Sequence[Scheduled]) -> dict[str, int]:
    events = [
        ComparatorEvent(row.entry_time, row.exit_time, row.side)
        for row in primary
    ]
    _, counts = _window_filter("rqhr:primary", events)
    return counts


def _json_payload(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RQHR support output escaped repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_or_verify(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing RQHR artifact differs: {path.name}")
        return "verified_existing"
    try:
        _atomic_write(path, payload)
        return "created"
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"concurrent RQHR artifact differs: {path.name}")
        return "verified_existing"


def build_synthetic_artifact() -> dict[str, Any]:
    registration = load_registration()
    dependencies = validate_frozen_dependencies()
    null = synthetic_null_report()
    payload: dict[str, Any] = {
        "protocol_version": f"{PROTOCOL_VERSION}_synthetic_v1",
        "candidate": POLICY_ID,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": registration["manifest_hash"],
            "policy_hash": registration["policy_hash"],
        },
        "support_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "dependencies": dependencies,
        "synthetic_null": null,
        "real_source_rows_read": 0,
        "real_rqhr_columns_read": 0,
        "comparator_rows_read": 0,
        "market_or_outcome_rows_read": 0,
        "passed": null["passed"],
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_synthetic_artifact_payload(
    payload: Mapping[str, Any],
) -> None:
    expected_top_keys = {
        "protocol_version",
        "candidate",
        "preregistration",
        "support_source",
        "dependencies",
        "synthetic_null",
        "real_source_rows_read",
        "real_rqhr_columns_read",
        "comparator_rows_read",
        "market_or_outcome_rows_read",
        "passed",
        "manifest_hash",
    }
    if set(payload) != expected_top_keys:
        raise RuntimeError("RQHR synthetic artifact schema drift")
    if payload.get("protocol_version") != f"{PROTOCOL_VERSION}_synthetic_v1":
        raise RuntimeError("RQHR synthetic artifact protocol drift")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("RQHR synthetic artifact candidate drift")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("RQHR synthetic artifact canonical hash mismatch")
    preregistration = payload.get("preregistration")
    if (
        not isinstance(preregistration, Mapping)
        or set(preregistration)
        != {"path", "sha256", "manifest_hash", "policy_hash"}
        or preregistration.get("path") != str(PREREGISTRATION)
        or preregistration.get("sha256") != PREREGISTRATION_SHA256
        or not all(
            isinstance(preregistration.get(field), str)
            and len(preregistration[field]) == 64
            for field in ("manifest_hash", "policy_hash")
        )
    ):
        raise RuntimeError("RQHR synthetic artifact preregistration drift")
    if payload.get("support_source") != {
        "path": str(SCRIPT_PATH),
        "sha256": sha256_file(SCRIPT_PATH),
    }:
        raise RuntimeError("RQHR synthetic artifact support source drift")
    if payload.get("dependencies") != _frozen_dependency_payload():
        raise RuntimeError("RQHR synthetic artifact dependency drift")
    for field in (
        "real_source_rows_read",
        "real_rqhr_columns_read",
        "comparator_rows_read",
        "market_or_outcome_rows_read",
    ):
        if payload.get(field) != 0:
            raise RuntimeError(f"RQHR synthetic artifact boundary opened: {field}")
    null = payload.get("synthetic_null")
    if not isinstance(null, Mapping):
        raise RuntimeError("RQHR synthetic null payload missing")
    expected_null_keys = {
        "protocol",
        "candidate",
        "source_rows_read",
        "real_rqhr_columns_read",
        "comparator_rows_read",
        "market_or_outcome_rows_read",
        "parameters",
        "scenarios",
        "passed",
        "failure_action",
    }
    if set(null) != expected_null_keys:
        raise RuntimeError("RQHR synthetic null schema drift")
    if null.get("protocol") != "RQHR-72 hash-bound synthetic moving-band null":
        raise RuntimeError("RQHR synthetic null protocol drift")
    if null.get("candidate") != POLICY_ID:
        raise RuntimeError("RQHR synthetic null candidate drift")
    for field in (
        "source_rows_read",
        "real_rqhr_columns_read",
        "comparator_rows_read",
        "market_or_outcome_rows_read",
    ):
        if null.get(field) != 0:
            raise RuntimeError(f"RQHR synthetic null boundary opened: {field}")
    if null.get("passed") is not True or payload.get("passed") is not True:
        raise RuntimeError("RQHR synthetic null did not pass")
    if null.get("failure_action") != (
        "real source may be opened only by frozen support builder"
    ):
        raise RuntimeError("RQHR synthetic null failure action drift")
    parameters = null.get("parameters")
    if parameters != {
        "grid_rows": GRID_ROWS,
        "snapshots_per_non_suppressed_bar": 10,
        "strict_prior_window_rows": PRIOR_WINDOW,
        "strict_prior_minimum_values": PRIOR_MINIMUM,
        "near_quantile": "39/40",
        "far_quantile": "9/10",
    }:
        raise RuntimeError("RQHR synthetic null parameters drift")
    scenarios = null.get("scenarios")
    if (
        not isinstance(scenarios, Mapping)
        or set(scenarios) != set(NULL_SCENARIOS)
        or len(scenarios) != len(NULL_SCENARIOS)
    ):
        raise RuntimeError("RQHR synthetic scenario cohort drift")
    for name in NULL_SCENARIOS:
        result = scenarios[name]
        missing_rows = (
            ((GRID_ROWS - 1) // 1_009) * 3
            + min(((GRID_ROWS - 1) % 1_009) + 1, 3)
        )
        expected_complete = (
            GRID_ROWS - missing_rows if name == "missing_rows" else GRID_ROWS
        )
        expected_scenario_keys = {
            "grid_rows",
            "complete_rows",
            "scheduled_snapshot_slots",
            "raw_confirmations",
            "accepted_events",
            "race_audit",
            "passed",
        }
        if (
            not isinstance(result, Mapping)
            or set(result) != expected_scenario_keys
            or result.get("grid_rows") != GRID_ROWS
            or result.get("complete_rows") != expected_complete
            or result.get("scheduled_snapshot_slots") != GRID_ROWS * 10
            or result.get("raw_confirmations") != 0
            or result.get("accepted_events") != 0
            or result.get("passed") is not True
        ):
            raise RuntimeError(f"RQHR synthetic scenario failed: {name}")
        audit = result.get("race_audit")
        if (
            not isinstance(audit, Mapping)
            or set(audit) != set(RACE_AUDIT_FIELDS)
            or any(
                isinstance(audit[field], bool)
                or not isinstance(audit[field], int)
                or audit[field] < 0
                for field in RACE_AUDIT_FIELDS
            )
        ):
            raise RuntimeError(f"RQHR synthetic race audit drift: {name}")


def load_sealed_synthetic_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = _repository_path(path)
    if not artifact_path.exists():
        raise RuntimeError(
            "RQHR synthetic artifact must be created and committed before real source"
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_synthetic_artifact_payload(payload)
    return payload


def _novelty_result(
    primary: Sequence[Scheduled],
    groups: Mapping[str, Sequence[ComparatorEvent]],
    *,
    rows_read: int,
    window_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    metrics = {
        name: novelty_metrics(primary, events)
        for name, events in sorted(groups.items())
    }
    checks: dict[str, dict[str, bool]] = {}
    for name, row in metrics.items():
        correlation = row["signed_5m_occupied_exposure_correlation"]
        checks[name] = {
            "minimum_rows": row["comparator_entries"] >= 10,
            "exact_entry_jaccard": row["exact_entry_jaccard"] <= 0.10,
            "candidate_containment": row["candidate_containment"] <= 0.35,
            "signed_exposure_correlation": (
                correlation is not None and abs(correlation) <= 0.35
            ),
        }
    return {
        "evaluated": True,
        "passed": bool(checks) and all(
            all(group_checks.values()) for group_checks in checks.values()
        ),
        "comparator_rows_read": rows_read,
        "candidate_window_counts": candidate_window_counts(primary),
        "comparator_window_counts": {
            name: dict(counts) for name, counts in sorted(window_counts.items())
        },
        "metrics": metrics,
        "checks": checks,
    }


def build_support_report(
    cfg: Config = Config(),
    *,
    source_loader: Any = load_real_source,
    comparator_loader: Any = load_comparator_groups,
) -> tuple[dict[str, Any], dict[str, list[Scheduled]]]:
    _validate_config(cfg)
    registration = load_registration()
    dependencies = validate_frozen_dependencies()
    synthetic = load_sealed_synthetic_artifact(cfg.synthetic_output)
    replayed_synthetic = build_synthetic_artifact()
    if synthetic != replayed_synthetic:
        raise RuntimeError(
            "RQHR sealed synthetic artifact does not match fresh frozen replay"
        )

    source_rows, source_audit = source_loader()
    clocks, raw_counts, race_audits = build_clocks(source_rows)
    support = source_support_summary(clocks["primary"], clocks=clocks)
    novelty: dict[str, Any] = {
        "evaluated": False,
        "passed": False,
        "reason": "source support failed before comparator access",
        "comparator_rows_read": 0,
    }
    if support["passed"]:
        try:
            groups, rows_read, window_counts = comparator_loader()
            novelty = _novelty_result(
                clocks["primary"],
                groups,
                rows_read=rows_read,
                window_counts=window_counts,
            )
        except ComparatorValidationError as exc:
            novelty = {
                "evaluated": True,
                "passed": False,
                "reason": "comparator validation failed closed",
                "error": str(exc),
                "comparator_rows_read": exc.rows_read,
                "comparator_window_counts": exc.window_counts,
            }

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": registration["manifest_hash"],
            "policy_hash": registration["policy_hash"],
        },
        "support_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "dependencies": dependencies,
        "synthetic_artifact": {
            "path": cfg.synthetic_output,
            "manifest_hash": synthetic.get("manifest_hash"),
            "passed": True,
        },
        "source_audit": source_audit,
        "source_support": support,
        "raw_candidate_counts": raw_counts,
        "accepted_clock_counts": {
            name: len(clocks[name]) for name in CONTROL_NAMES
        },
        "race_audits": race_audits,
        "clock_sha256": clock_hash(clocks),
        "novelty": novelty,
        "real_source_rows_read": source_audit["rows"],
        "real_rqhr_columns_read": len(RQHR_COLUMNS),
        "comparator_rows_read": novelty.get("comparator_rows_read", 0),
        "market_or_outcome_rows_read": 0,
        "price_funding_return_pnl_cagr_mdd_opened": False,
        "advance_to_evaluator_freeze": bool(
            support["passed"] and novelty["passed"]
        ),
        "failure_action": (
            "freeze strict evaluator"
            if support["passed"] and novelty["passed"]
            else "retire RQHR-72 unchanged before market outcomes"
        ),
    }
    report["manifest_hash"] = canonical_hash(report)
    return report, clocks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--clock-output", default=Config.clock_output)
    parser.add_argument("--synthetic-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(
        output=args.output,
        clock_output=args.clock_output,
        synthetic_output=Config.synthetic_output,
    )
    _validate_config(cfg)
    if args.synthetic_only:
        payload = build_synthetic_artifact()
        status = _write_or_verify(
            _repository_path(cfg.synthetic_output),
            _json_payload(payload),
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "candidate": POLICY_ID,
                    "output": cfg.synthetic_output,
                    "manifest_hash": payload["manifest_hash"],
                    "passed": payload["passed"],
                    "real_rqhr_columns_read": payload[
                        "real_rqhr_columns_read"
                    ],
                    "comparator_rows_read": payload["comparator_rows_read"],
                    "market_or_outcome_rows_read": payload[
                        "market_or_outcome_rows_read"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["passed"] else 2

    report, clocks = build_support_report(cfg)
    report_status = _write_or_verify(
        _repository_path(cfg.output),
        _json_payload(report),
    )
    clock_status = "not_written"
    if clocks:
        clock_status = _write_or_verify(
            _repository_path(cfg.clock_output),
            clock_payload(clocks),
        )
    print(
        json.dumps(
            {
                "candidate": POLICY_ID,
                "report_status": report_status,
                "clock_status": clock_status,
                "source_support_passed": report["source_support"]["passed"],
                "novelty_evaluated": report["novelty"]["evaluated"],
                "novelty_passed": report["novelty"]["passed"],
                "advance_to_evaluator_freeze": report[
                    "advance_to_evaluator_freeze"
                ],
                "market_or_outcome_rows_read": report[
                    "market_or_outcome_rows_read"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["advance_to_evaluator_freeze"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
