"""Build outcome-blind RCRE-72 source-support, control, and novelty clocks."""

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
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from training import build_ofr_repo_mix_shock_resolution_race_support as rmsr_support
from training import build_ofr_repo_venue_fragmentation_consensus_support as rvfc_support
from training import preregister_ofr_repo_collateral_routing_efficiency as prereg


PROTOCOL_VERSION = "ofr_repo_collateral_routing_efficiency_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_ofr_repo_collateral_routing_efficiency_support.py"
)
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "1cd5c773e22101ae19d8d0753ca53e3248b10de3ecc947fc2e4e353961ee69e2"
)
PREREGISTRATION_MANIFEST_HASH = (
    "a8a7831c773666b42b98d673165b3e7844111c5d5ed280468307306c5843bf4b"
)
PARSER_DEPENDENCIES: tuple[tuple[Path, str], ...] = (
    (
        Path("training/build_ofr_repo_venue_fragmentation_consensus_support.py"),
        "3d254eafc99c127cb16d398aecd5444cb952ee1fc451d455a42f35e0c3824c89",
    ),
    (
        Path("training/build_ofr_repo_mix_shock_resolution_race_support.py"),
        "d00fa29f04c5eb09ffbc7787ccdf959643d579614f3eca8caa52d3ce8c18100d",
    ),
)
DEFAULT_CLOCK = Path(
    "results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/ofr_repo_collateral_routing_efficiency_support_2026-07-23.json"
)

UTC = timezone.utc
BAR = timedelta(minutes=5)
HOLD = timedelta(hours=72)
LOOKBACK = 252
END_DATE = date(2023, 12, 31)
TRAIN_START = datetime(2021, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
SELECTION_START = TRAIN_END
SELECTION_END = datetime(2024, 1, 1, tzinfo=UTC)

SOURCE_CONTROLS = (
    "quantity_gap_original",
    "quantity_gap_swapped",
    "rate_gap_original",
    "rate_gap_swapped",
    "absolute_pressure",
    "both_legs_extreme",
    "absolute_rank_additive",
    "sign_without_magnitude",
    "one_complete_date_stale",
    "five_complete_date_stale",
    "year_rate_gap_permutation",
    "year_product_permutation",
)
ECONOMIC_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
)
CONTROL_NAMES = ("primary", *SOURCE_CONTROLS, *ECONOMIC_CONTROLS)

CLOCK_COLUMNS = (
    "control",
    "observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "split",
    "side",
    "state",
    "quantity_sign",
    "rate_sign",
    "quadrant",
    "quantity_gap",
    "rate_gap",
    "routing_pressure",
    "score",
    "u_quantity_gap",
    "u_rate_gap",
    "u_routing_pressure",
    "u_absolute_pressure",
    "u_absolute_quantity_gap",
    "u_absolute_rate_gap",
)


@dataclass(frozen=True)
class FeatureRow:
    observation_date: date
    available_at: datetime
    epoch: int
    decision_allowed: bool
    components: Mapping[str, Fraction]


@dataclass(frozen=True)
class RankRow:
    feature: FeatureRow
    units: Mapping[str, Fraction]


@dataclass(frozen=True)
class StatePoint:
    control: str
    rank: RankRow
    epoch: int
    state: int
    score: Fraction


@dataclass(frozen=True)
class Candidate:
    control: str
    rank: RankRow
    state: int
    side: int
    score: Fraction


@dataclass(frozen=True)
class Scheduled:
    control: str
    observation_date: date
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    split: str
    side: int
    state: int
    score: Fraction
    components: Mapping[str, Fraction]
    units: Mapping[str, Fraction]


@dataclass(frozen=True)
class ComparatorEvent:
    entry_time: datetime
    exit_time: datetime
    side: int


@dataclass(frozen=True)
class SourceAudit:
    normalized_rows_read: int
    required_rows_read: int
    source_dates_seen: int
    valid_feature_dates: int
    invalid_missing_or_null_dates: int
    invalid_materiality_dates: int
    equal_availability_rows_suppressed: int
    venue_swap_dates_checked: int
    venue_swap_identity_failures: int


class ComparatorValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        rows_read: int,
        window_counts: Mapping[str, Mapping[str, int]] | None = None,
        raw_group_counts: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.rows_read = rows_read
        self.window_counts = dict(window_counts or {})
        self.raw_group_counts = dict(raw_group_counts or {})


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("RCRE support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RCRE support path must remain repository-relative") from exc
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


def _load_registration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("RCRE preregistration file hash mismatch")
    for path, expected_hash in PARSER_DEPENDENCIES:
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"RCRE parser dependency hash mismatch: {path}")
    payload = json.loads(_repository_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload, verify_sources=True)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("RCRE preregistration manifest hash mismatch")
    return payload


def _required_value(
    rows: Mapping[str, rmsr_support.SourceRow], mnemonic: str
) -> Fraction:
    row = rows[mnemonic]
    if row.value is None:
        raise RuntimeError("required RCRE value is null")
    return row.value


def _components(quantity_gap: Fraction, rate_gap: Fraction) -> dict[str, Fraction]:
    routing_pressure = quantity_gap * rate_gap
    return {
        "quantity_gap": quantity_gap,
        "rate_gap": rate_gap,
        "routing_pressure": routing_pressure,
        "absolute_pressure": abs(routing_pressure),
        "absolute_quantity_gap": abs(quantity_gap),
        "absolute_rate_gap": abs(rate_gap),
    }


def build_features(
    by_date: Mapping[date, Mapping[str, rmsr_support.SourceRow]],
    *,
    normalized_rows_read: int = 0,
    required_rows_read: int = 0,
) -> tuple[list[FeatureRow], SourceAudit]:
    required = set(prereg.REQUIRED_SERIES)
    features: list[FeatureRow] = []
    epoch = 0
    invalid_missing_or_null = 0
    invalid_materiality = 0
    swap_failures = 0
    for observation_day in sorted(by_date):
        rows = by_date[observation_day]
        if set(rows) != required or any(
            row.value is None or row.disclosure_edit for row in rows.values()
        ):
            epoch += 1
            invalid_missing_or_null += 1
            continue
        value = lambda mnemonic: _required_value(rows, mnemonic)
        gcf_ag = value("REPO-GCF_TV_AG-P")
        gcf_t = value("REPO-GCF_TV_T-P")
        tri_ag = value("REPO-TRIV1_TV_AG-P")
        tri_t = value("REPO-TRIV1_TV_T-P")
        gcf_total = gcf_ag + gcf_t
        tri_total = tri_ag + tri_t
        material = (
            gcf_total > 0
            and tri_total > 0
            and gcf_ag * 20 >= gcf_total
            and gcf_t * 20 >= gcf_total
            and tri_ag * 20 >= tri_total
            and tri_t * 20 >= tri_total
        )
        if not material:
            epoch += 1
            invalid_materiality += 1
            continue
        gcf_share = gcf_ag / gcf_total
        tri_share = tri_ag / tri_total
        gcf_relative_rate = value("REPO-GCF_AR_AG-P") - value(
            "REPO-GCF_AR_T-P"
        )
        tri_relative_rate = value("REPO-TRIV1_AR_AG-P") - value(
            "REPO-TRIV1_AR_T-P"
        )
        quantity_gap = gcf_share - tri_share
        rate_gap = gcf_relative_rate - tri_relative_rate
        components = _components(quantity_gap, rate_gap)
        swapped = _components(
            tri_share - gcf_share,
            tri_relative_rate - gcf_relative_rate,
        )
        identity_valid = (
            swapped["quantity_gap"] == -components["quantity_gap"]
            and swapped["rate_gap"] == -components["rate_gap"]
            and swapped["routing_pressure"] == components["routing_pressure"]
        )
        if not identity_valid:
            swap_failures += 1
            raise RuntimeError("RCRE venue-swap identity failed")
        features.append(
            FeatureRow(
                observation_date=observation_day,
                available_at=max(row.available_at for row in rows.values()),
                epoch=epoch,
                decision_allowed=False,
                components=components,
            )
        )
    latest_by_availability: dict[datetime, date] = {}
    for row in features:
        latest_by_availability[row.available_at] = max(
            row.observation_date,
            latest_by_availability.get(row.available_at, row.observation_date),
        )
    features = [
        replace(
            row,
            decision_allowed=(
                latest_by_availability[row.available_at] == row.observation_date
            ),
        )
        for row in features
    ]
    audit = SourceAudit(
        normalized_rows_read=normalized_rows_read,
        required_rows_read=required_rows_read,
        source_dates_seen=len(by_date),
        valid_feature_dates=len(features),
        invalid_missing_or_null_dates=invalid_missing_or_null,
        invalid_materiality_dates=invalid_materiality,
        equal_availability_rows_suppressed=sum(
            not row.decision_allowed for row in features
        ),
        venue_swap_dates_checked=len(features),
        venue_swap_identity_failures=swap_failures,
    )
    return features, audit


def midrank_unit(current: Fraction, prior: Sequence[Fraction]) -> Fraction:
    if len(prior) != LOOKBACK:
        raise RuntimeError("RCRE midrank history length changed")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return Fraction(2 * less + equal - LOOKBACK, LOOKBACK)


def build_rank_rows(features: Sequence[FeatureRow]) -> list[RankRow]:
    histories = {name: deque(maxlen=LOOKBACK) for name in prereg.COMPONENTS}
    output: list[RankRow] = []
    batches: dict[datetime, list[FeatureRow]] = defaultdict(list)
    for row in features:
        batches[row.available_at].append(row)
    for available_at in sorted(batches):
        batch = sorted(batches[available_at], key=lambda row: row.observation_date)
        if all(len(history) == LOOKBACK for history in histories.values()):
            prior = {name: tuple(history) for name, history in histories.items()}
            output.extend(
                RankRow(
                    feature=row,
                    units={
                        name: midrank_unit(row.components[name], prior[name])
                        for name in prereg.COMPONENTS
                    },
                )
                for row in batch
            )
        for row in batch:
            for name in prereg.COMPONENTS:
                histories[name].append(row.components[name])
    return output


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def component_state(unit: Fraction) -> int:
    if unit >= Fraction(1, 2):
        return 1
    if unit <= Fraction(-1, 2):
        return -1
    return 0


def primary_state(row: RankRow) -> int:
    product = row.feature.components["routing_pressure"]
    unit = row.units["routing_pressure"]
    if product > 0 and unit >= Fraction(1, 2):
        return 1
    if product < 0 and unit <= Fraction(-1, 2):
        return -1
    return 0


def venue_swapped_rank(row: RankRow) -> RankRow:
    components = dict(row.feature.components)
    components["quantity_gap"] = -components["quantity_gap"]
    components["rate_gap"] = -components["rate_gap"]
    units = dict(row.units)
    units["quantity_gap"] = -units["quantity_gap"]
    units["rate_gap"] = -units["rate_gap"]
    return RankRow(
        feature=replace(row.feature, components=components),
        units=units,
    )


def _state_points(rank_rows: Sequence[RankRow], control: str) -> list[StatePoint]:
    output: list[StatePoint] = []
    for row in rank_rows:
        if not row.feature.decision_allowed:
            continue
        effective = (
            venue_swapped_rank(row)
            if control in {"quantity_gap_swapped", "rate_gap_swapped"}
            else row
        )
        product = effective.feature.components["routing_pressure"]
        if control == "primary":
            state = primary_state(effective)
            score = effective.units["routing_pressure"]
        elif control == "quantity_gap_original":
            score = effective.units["quantity_gap"]
            state = component_state(score)
        elif control == "quantity_gap_swapped":
            score = effective.units["quantity_gap"]
            state = component_state(score)
        elif control == "rate_gap_original":
            score = effective.units["rate_gap"]
            state = component_state(score)
        elif control == "rate_gap_swapped":
            score = effective.units["rate_gap"]
            state = component_state(score)
        elif control == "absolute_pressure":
            score = effective.units["absolute_pressure"]
            state = component_state(score)
        elif control == "both_legs_extreme":
            score = min(
                effective.units["absolute_quantity_gap"],
                effective.units["absolute_rate_gap"],
            )
            state = _sign(product) if score >= Fraction(1, 2) else 0
        elif control == "absolute_rank_additive":
            score = (
                effective.units["absolute_quantity_gap"]
                + effective.units["absolute_rate_gap"]
            ) / 2
            state = _sign(product) if score >= Fraction(1, 2) else 0
        elif control == "sign_without_magnitude":
            state = _sign(product)
            score = Fraction(state)
        else:
            raise RuntimeError(f"unknown RCRE state control: {control}")
        output.append(
            StatePoint(
                control=control,
                rank=effective,
                epoch=effective.feature.epoch,
                state=state,
                score=score,
            )
        )
    return output


def stale_state_points(rows: Sequence[StatePoint], lag: int, control: str) -> list[StatePoint]:
    if lag <= 0:
        raise RuntimeError("RCRE stale lag must be positive")
    output: list[StatePoint] = []
    segment = 0
    previous_destination: int | None = None
    previous_epoch: int | None = None
    for index, current in enumerate(rows):
        source_index = index - lag
        if source_index < 0 or rows[source_index].epoch != current.epoch:
            previous_destination = None
            previous_epoch = None
            continue
        source = rows[source_index]
        contiguous = (
            previous_destination is not None
            and previous_destination == index - 1
            and previous_epoch == current.epoch
        )
        if not contiguous:
            segment += 1
        output.append(
            StatePoint(
                control=control,
                rank=current.rank,
                epoch=segment,
                state=source.state,
                score=source.score,
            )
        )
        previous_destination = index
        previous_epoch = current.epoch
    return output


def _permutation_key(mode: str, year: int, row: FeatureRow) -> bytes:
    return hashlib.sha256(
        f"RCRE-72|{mode}|{year}|{row.observation_date.isoformat()}".encode()
    ).digest()


def permute_features(features: Sequence[FeatureRow], mode: str) -> list[FeatureRow]:
    if mode not in {"year_rate_gap_permutation", "year_product_permutation"}:
        raise RuntimeError("unknown RCRE permutation mode")
    by_year: dict[int, list[FeatureRow]] = defaultdict(list)
    for row in features:
        by_year[row.observation_date.year].append(row)
    output: list[FeatureRow] = []
    for year in sorted(by_year):
        destinations = sorted(by_year[year], key=lambda row: row.observation_date)
        sources = sorted(
            destinations,
            key=lambda row: _permutation_key(mode, year, row),
        )
        for destination, source in zip(destinations, sources):
            components = dict(destination.components)
            if mode == "year_rate_gap_permutation":
                components = _components(
                    destination.components["quantity_gap"],
                    source.components["rate_gap"],
                )
            else:
                product = source.components["routing_pressure"]
                components["routing_pressure"] = product
                components["absolute_pressure"] = abs(product)
            output.append(replace(destination, components=components))
    return sorted(output, key=lambda row: row.observation_date)


def transition_candidates(points: Sequence[StatePoint]) -> list[Candidate]:
    output: list[Candidate] = []
    previous: StatePoint | None = None
    for current in points:
        if previous is None or current.epoch != previous.epoch:
            previous = current
            continue
        if current.state in {-1, 1} and current.state != previous.state:
            output.append(
                Candidate(
                    control=current.control,
                    rank=current.rank,
                    state=current.state,
                    side=-current.state,
                    score=current.score,
                )
            )
        previous = current
    return output


def _ceil_5m(value: datetime) -> datetime:
    seconds = int(value.timestamp())
    interval = int(BAR.total_seconds())
    return datetime.fromtimestamp(((seconds + interval - 1) // interval) * interval, UTC)


def _split(entry: datetime, exit_time: datetime) -> str | None:
    if TRAIN_START <= entry and exit_time <= TRAIN_END:
        return "train"
    if SELECTION_START <= entry and exit_time <= SELECTION_END:
        return "selection"
    return None


def schedule(control: str, candidates: Sequence[Candidate]) -> list[Scheduled]:
    output: list[Scheduled] = []
    reserved_until: datetime | None = None
    for candidate in sorted(
        candidates,
        key=lambda row: (row.rank.feature.available_at, row.rank.feature.observation_date),
    ):
        entry = _ceil_5m(candidate.rank.feature.available_at) + BAR
        exit_time = entry + HOLD
        split = _split(entry, exit_time)
        if split is None:
            continue
        if reserved_until is not None and entry < reserved_until:
            continue
        output.append(
            Scheduled(
                control=control,
                observation_date=candidate.rank.feature.observation_date,
                signal_time=candidate.rank.feature.available_at,
                entry_time=entry,
                exit_time=exit_time,
                split=split,
                side=candidate.side,
                state=candidate.state,
                score=candidate.score,
                components=candidate.rank.feature.components,
                units=candidate.rank.units,
            )
        )
        reserved_until = exit_time
    return output


def _random_side(entry: datetime) -> int:
    key = f"RCRE-72|deterministic_random_side|{entry.isoformat()}".encode()
    return 1 if hashlib.sha256(key).digest()[0] < 128 else -1


def _clone_clock(
    rows: Sequence[Scheduled], control: str, side: Callable[[Scheduled], int]
) -> list[Scheduled]:
    return [replace(row, control=control, side=side(row)) for row in rows]


def build_clocks(
    features: Sequence[FeatureRow], rank_rows: Sequence[RankRow]
) -> dict[str, list[Scheduled]]:
    primary_points = _state_points(rank_rows, "primary")
    primary = schedule("primary", transition_candidates(primary_points))
    clocks: dict[str, list[Scheduled]] = {
        "primary": primary,
        "quantity_gap_original": schedule(
            "quantity_gap_original",
            transition_candidates(_state_points(rank_rows, "quantity_gap_original")),
        ),
        "quantity_gap_swapped": schedule(
            "quantity_gap_swapped",
            transition_candidates(_state_points(rank_rows, "quantity_gap_swapped")),
        ),
        "rate_gap_original": schedule(
            "rate_gap_original",
            transition_candidates(_state_points(rank_rows, "rate_gap_original")),
        ),
        "rate_gap_swapped": schedule(
            "rate_gap_swapped",
            transition_candidates(_state_points(rank_rows, "rate_gap_swapped")),
        ),
        "absolute_pressure": schedule(
            "absolute_pressure",
            transition_candidates(_state_points(rank_rows, "absolute_pressure")),
        ),
        "both_legs_extreme": schedule(
            "both_legs_extreme",
            transition_candidates(_state_points(rank_rows, "both_legs_extreme")),
        ),
        "absolute_rank_additive": schedule(
            "absolute_rank_additive",
            transition_candidates(_state_points(rank_rows, "absolute_rank_additive")),
        ),
        "sign_without_magnitude": schedule(
            "sign_without_magnitude",
            transition_candidates(_state_points(rank_rows, "sign_without_magnitude")),
        ),
        "one_complete_date_stale": schedule(
            "one_complete_date_stale",
            transition_candidates(
                stale_state_points(primary_points, 1, "one_complete_date_stale")
            ),
        ),
        "five_complete_date_stale": schedule(
            "five_complete_date_stale",
            transition_candidates(
                stale_state_points(primary_points, 5, "five_complete_date_stale")
            ),
        ),
    }
    for control in ("year_rate_gap_permutation", "year_product_permutation"):
        permuted = permute_features(features, control)
        permuted_ranks = build_rank_rows(permuted)
        points = [
            replace(point, control=control)
            for point in _state_points(permuted_ranks, "primary")
        ]
        clocks[control] = schedule(control, transition_candidates(points))
    clocks.update(
        {
            "exact_direction_flip": _clone_clock(
                primary, "exact_direction_flip", lambda row: -row.side
            ),
            "deterministic_random_side": _clone_clock(
                primary,
                "deterministic_random_side",
                lambda row: _random_side(row.entry_time),
            ),
            "constant_long": _clone_clock(primary, "constant_long", lambda _row: 1),
            "constant_short": _clone_clock(primary, "constant_short", lambda _row: -1),
        }
    )
    if set(clocks) != set(CONTROL_NAMES):
        raise RuntimeError("RCRE control clock set changed")
    return clocks


def _contained(
    rows: Sequence[Scheduled], start: datetime, end: datetime
) -> list[Scheduled]:
    return [row for row in rows if row.entry_time >= start and row.exit_time <= end]


def _quadrant(components: Mapping[str, Fraction]) -> str:
    quantity_sign = _sign(components["quantity_gap"])
    rate_sign = _sign(components["rate_gap"])
    if quantity_sign == 0 or rate_sign == 0:
        return "ZERO"
    return f"q{'+' if quantity_sign > 0 else '-'}r{'+' if rate_sign > 0 else '-'}"


def _summary(
    rows: Sequence[Scheduled], start: datetime, end: datetime
) -> dict[str, Any]:
    selected = _contained(rows, start, end)
    month_counts = Counter(row.entry_time.strftime("%Y-%m") for row in selected)
    quarter_counts = Counter(
        f"{row.entry_time.year}-Q{(row.entry_time.month - 1) // 3 + 1}"
        for row in selected
    )
    product_counts = Counter(
        "positive" if row.components["routing_pressure"] > 0 else "negative"
        for row in selected
        if row.components["routing_pressure"] != 0
    )
    quadrant_counts = Counter(_quadrant(row.components) for row in selected)
    gaps = [
        (current.entry_time - previous.entry_time).total_seconds() / 86_400
        for previous, current in zip(selected, selected[1:])
    ]
    return {
        "events": len(selected),
        "longs": sum(row.side == 1 for row in selected),
        "shorts": sum(row.side == -1 for row in selected),
        "active_months": len(month_counts),
        "active_quarters": len(quarter_counts),
        "max_single_month_share": (
            max(month_counts.values()) / len(selected) if selected else 0.0
        ),
        "maximum_entry_gap_elapsed_days": max(gaps, default=0.0),
        "product_sign_counts": dict(sorted(product_counts.items())),
        "product_sign_shares": (
            {
                name: count / len(selected)
                for name, count in sorted(product_counts.items())
            }
            if selected
            else {}
        ),
        "quadrant_counts": dict(sorted(quadrant_counts.items())),
        "quadrant_shares": (
            {
                name: count / len(selected)
                for name, count in sorted(quadrant_counts.items())
            }
            if selected
            else {}
        ),
    }


def _clock_valid(control: str, rows: Sequence[Scheduled]) -> bool:
    return (
        all(row.control == control for row in rows)
        and all(row.side in {-1, 1} for row in rows)
        and all(row.signal_time < row.entry_time < row.exit_time for row in rows)
        and all(row.entry_time == _ceil_5m(row.signal_time) + BAR for row in rows)
        and all(row.exit_time - row.entry_time == HOLD for row in rows)
        and all(_split(row.entry_time, row.exit_time) == row.split for row in rows)
        and len({row.entry_time for row in rows}) == len(rows)
        and all(
            current.entry_time >= previous.exit_time
            for previous, current in zip(rows, rows[1:])
        )
    )


def _primary_valid(rows: Sequence[Scheduled]) -> bool:
    for row in rows:
        components = row.components
        units = row.units
        product = components["routing_pressure"]
        valid_state = (
            row.state == 1
            and product > 0
            and units["routing_pressure"] >= Fraction(1, 2)
            and row.side == -1
        ) or (
            row.state == -1
            and product < 0
            and units["routing_pressure"] <= Fraction(-1, 2)
            and row.side == 1
        )
        if not (
            valid_state
            and set(components) == set(prereg.COMPONENTS)
            and set(units) == set(prereg.COMPONENTS)
            and all(isinstance(value, Fraction) for value in components.values())
            and all(isinstance(value, Fraction) for value in units.values())
            and components["quantity_gap"] != 0
            and components["rate_gap"] != 0
            and product == components["quantity_gap"] * components["rate_gap"]
            and _quadrant(components) != "ZERO"
        ):
            return False
    return True


def _label_pair_valid(
    original: Sequence[Scheduled], swapped: Sequence[Scheduled]
) -> bool:
    if len(original) != len(swapped):
        return False
    for left, right in zip(original, swapped):
        if not (
            left.observation_date == right.observation_date
            and left.signal_time == right.signal_time
            and left.entry_time == right.entry_time
            and left.exit_time == right.exit_time
            and left.side == -right.side
            and left.state == -right.state
            and left.score == -right.score
            and left.components["quantity_gap"]
            == -right.components["quantity_gap"]
            and left.components["rate_gap"] == -right.components["rate_gap"]
            and left.components["routing_pressure"]
            == right.components["routing_pressure"]
            and left.units["quantity_gap"] == -right.units["quantity_gap"]
            and left.units["rate_gap"] == -right.units["rate_gap"]
            and left.units["routing_pressure"]
            == right.units["routing_pressure"]
        ):
            return False
    return True


def _side_control_valid(
    primary: Sequence[Scheduled],
    control: Sequence[Scheduled],
    expected_side: Callable[[Scheduled], int],
) -> bool:
    if len(primary) != len(control):
        return False
    return all(
        right.observation_date == left.observation_date
        and right.signal_time == left.signal_time
        and right.entry_time == left.entry_time
        and right.exit_time == left.exit_time
        and right.split == left.split
        and right.state == left.state
        and right.score == left.score
        and right.components == left.components
        and right.units == left.units
        and right.side == expected_side(left)
        for left, right in zip(primary, control)
    )


def source_support(
    clocks: Mapping[str, Sequence[Scheduled]],
    policy: Mapping[str, Any],
    source_audit: SourceAudit,
    *,
    post_2023_source_rows: int = 0,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    primary = clocks["primary"]
    summaries = {
        "train_selection": _summary(primary, TRAIN_START, SELECTION_END),
        "train": _summary(primary, TRAIN_START, TRAIN_END),
        "selection": _summary(primary, SELECTION_START, SELECTION_END),
        "2021": _summary(primary, TRAIN_START, datetime(2022, 1, 1, tzinfo=UTC)),
        "2022": _summary(primary, datetime(2022, 1, 1, tzinfo=UTC), TRAIN_END),
        "train_h1_2021": _summary(
            primary, TRAIN_START, datetime(2021, 7, 1, tzinfo=UTC)
        ),
        "train_h2_2021": _summary(
            primary,
            datetime(2021, 7, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
        "train_h1_2022": _summary(
            primary,
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2022, 7, 1, tzinfo=UTC),
        ),
        "train_h2_2022": _summary(
            primary, datetime(2022, 7, 1, tzinfo=UTC), TRAIN_END
        ),
        "selection_h1": _summary(
            primary, SELECTION_START, datetime(2023, 7, 1, tzinfo=UTC)
        ),
        "selection_h2": _summary(
            primary, datetime(2023, 7, 1, tzinfo=UTC), SELECTION_END
        ),
    }
    gates = policy["source_support_gates"]
    quadrants = ("q+r+", "q-r-", "q+r-", "q-r+")

    def minimum_share(split: str, names: Sequence[str], field: str) -> float:
        return min(summaries[split][field].get(name, 0.0) for name in names)

    def maximum_share(split: str, names: Sequence[str], field: str) -> float:
        return max(summaries[split][field].get(name, 0.0) for name in names)

    checks = {
        "all_control_clocks_valid": all(
            _clock_valid(name, clocks[name]) for name in CONTROL_NAMES
        ),
        "primary_exact_interaction_valid": _primary_valid(primary),
        "quantity_gap_label_pair_identity": _label_pair_valid(
            clocks["quantity_gap_original"], clocks["quantity_gap_swapped"]
        ),
        "rate_gap_label_pair_identity": _label_pair_valid(
            clocks["rate_gap_original"], clocks["rate_gap_swapped"]
        ),
        "economic_side_controls_exact": all(
            (
                _side_control_valid(
                    primary,
                    clocks["exact_direction_flip"],
                    lambda row: -row.side,
                ),
                _side_control_valid(
                    primary,
                    clocks["deterministic_random_side"],
                    lambda row: _random_side(row.entry_time),
                ),
                _side_control_valid(
                    primary, clocks["constant_long"], lambda _row: 1
                ),
                _side_control_valid(
                    primary, clocks["constant_short"], lambda _row: -1
                ),
            )
        ),
        "train_total": summaries["train"]["events"]
        >= gates["train_total_minimum"],
        "each_train_year": all(
            summaries[year]["events"] >= gates["each_train_year_minimum"]
            for year in ("2021", "2022")
        ),
        "each_train_half": all(
            summaries[name]["events"] >= gates["each_train_half_minimum"]
            for name in (
                "train_h1_2021",
                "train_h2_2021",
                "train_h1_2022",
                "train_h2_2022",
            )
        ),
        "train_each_side": min(
            summaries["train"]["longs"], summaries["train"]["shorts"]
        )
        >= gates["train_each_side_minimum"],
        "selection_total": summaries["selection"]["events"]
        >= gates["selection_total_minimum"],
        "each_selection_half": all(
            summaries[name]["events"] >= gates["each_selection_half_minimum"]
            for name in ("selection_h1", "selection_h2")
        ),
        "selection_each_side": min(
            summaries["selection"]["longs"], summaries["selection"]["shorts"]
        )
        >= gates["selection_each_side_minimum"],
        "every_quarter_active": (
            summaries["train"]["active_quarters"] == 8
            and summaries["selection"]["active_quarters"] == 4
        ),
        "train_month_concentration": summaries["train"]["max_single_month_share"]
        <= gates["train_maximum_month_share"],
        "selection_month_concentration": summaries["selection"][
            "max_single_month_share"
        ]
        <= gates["selection_maximum_month_share"],
        "maximum_entry_gap": summaries["train_selection"][
            "maximum_entry_gap_elapsed_days"
        ]
        <= gates["maximum_accepted_entry_gap_elapsed_days"],
        "train_each_product_sign": minimum_share(
            "train", ("positive", "negative"), "product_sign_shares"
        )
        >= gates["train_each_product_sign_minimum_share"],
        "selection_each_product_sign": minimum_share(
            "selection", ("positive", "negative"), "product_sign_shares"
        )
        >= gates["selection_each_product_sign_minimum_share"],
        "train_each_quadrant": minimum_share(
            "train", quadrants, "quadrant_shares"
        )
        >= gates["train_each_quadrant_minimum_share"],
        "selection_each_quadrant": minimum_share(
            "selection", quadrants, "quadrant_shares"
        )
        >= gates["selection_each_quadrant_minimum_share"],
        "train_quadrant_concentration": maximum_share(
            "train", quadrants, "quadrant_shares"
        )
        <= gates["train_maximum_quadrant_share"],
        "selection_quadrant_concentration": maximum_share(
            "selection", quadrants, "quadrant_shares"
        )
        <= gates["selection_maximum_quadrant_share"],
        "venue_swap_identity": (
            source_audit.venue_swap_identity_failures == 0
            and source_audit.venue_swap_dates_checked
            == source_audit.valid_feature_dates
        ),
        "post_2023_source_rows": post_2023_source_rows
        == gates["post_2023_source_rows_read_required"],
    }
    return checks, summaries


def _special_comparator_identity(
    name: str, raw: Mapping[str, str]
) -> tuple[str, str, str, bool]:
    if name == "ofr_repo_venue_fragmentation_consensus_primary":
        control = raw["control"]
        return f"rvfc:{control}", "entry_time", "exit_time", control == "primary"
    if name == "ofr_repo_mix_shock_resolution_race_primary":
        control = raw["control"]
        return f"rmsr:{control}", "entry_time", "exit_time", control == "primary"
    group, entry_field, exit_field, include = rvfc_support._comparator_identity(
        name, raw
    )
    if name == "fed_h8_deposit_migration_primary":
        group = f"h8dm:{raw['clock_mode']}"
    elif name == "soma_lending_collateral_scarcity_primary":
        group = f"slcs:{raw['control']}"
    return group, entry_field, exit_field, include


def _comparator_header(name: str) -> tuple[str, ...]:
    if name == "ofr_repo_venue_fragmentation_consensus_primary":
        return rvfc_support.CLOCK_COLUMNS
    if name == "ofr_repo_mix_shock_resolution_race_primary":
        return rmsr_support.CLOCK_COLUMNS
    return rvfc_support.COMPARATOR_HEADERS[name]


def _window_bucket(event: ComparatorEvent) -> str:
    if event.entry_time >= TRAIN_START and event.exit_time <= SELECTION_END:
        return "fully_contained_rows_used"
    if event.exit_time <= TRAIN_START:
        return "rows_before_window"
    if event.entry_time >= SELECTION_END:
        return "rows_after_window"
    return "rows_crossing_boundary"


def _validate_raw_groups(
    raw_groups: Mapping[str, Sequence[ComparatorEvent]], *, rows_read: int
) -> None:
    for name, events in sorted(raw_groups.items()):
        rows = sorted(events, key=lambda row: row.entry_time)
        if len({row.entry_time for row in rows}) != len(rows):
            raise ComparatorValidationError(
                f"duplicate comparator entry: {name}", rows_read=rows_read
            )
        if any(
            current.entry_time < previous.exit_time
            for previous, current in zip(rows, rows[1:])
        ):
            raise ComparatorValidationError(
                f"overlapping comparator: {name}", rows_read=rows_read
            )


def _filter_required_groups(
    groups: Mapping[str, Sequence[ComparatorEvent]],
    *,
    rows_read: int,
    raw_group_counts: Mapping[str, int],
) -> tuple[dict[str, list[ComparatorEvent]], dict[str, dict[str, int]]]:
    contained: dict[str, list[ComparatorEvent]] = {}
    counts: dict[str, dict[str, int]] = {}
    for name, events in sorted(groups.items()):
        group_counts = {
            "total_raw_rows_parsed": len(events),
            "fully_contained_rows_used": 0,
            "rows_before_window": 0,
            "rows_after_window": 0,
            "rows_crossing_boundary": 0,
        }
        selected: list[ComparatorEvent] = []
        for event in events:
            bucket = _window_bucket(event)
            group_counts[bucket] += 1
            if bucket == "fully_contained_rows_used":
                selected.append(event)
        counts[name] = group_counts
        if not selected:
            raise ComparatorValidationError(
                f"required comparator has zero contained rows: {name}",
                rows_read=rows_read,
                window_counts=counts,
                raw_group_counts=raw_group_counts,
            )
        contained[name] = selected
    return contained, counts


def load_comparator_groups() -> tuple[
    dict[str, list[ComparatorEvent]],
    int,
    dict[str, dict[str, int]],
    dict[str, int],
]:
    raw_groups: dict[str, list[ComparatorEvent]] = defaultdict(list)
    required_groups: dict[str, list[ComparatorEvent]] = defaultdict(list)
    rows_read = 0
    for spec in prereg.COMPARATOR_SPECS:
        name = str(spec["name"])
        try:
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"comparator hash mismatch: {name}")
            artifact_rows = 0
            included_for_spec = 0
            with gzip.open(_repository_path(spec["path"]), "rt", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != _comparator_header(name):
                    raise RuntimeError(f"comparator header changed: {name}")
                for raw in reader:
                    rows_read += 1
                    artifact_rows += 1
                    group, entry_field, exit_field, include = (
                        _special_comparator_identity(name, raw)
                    )
                    event = ComparatorEvent(
                        entry_time=rmsr_support._timestamp(raw[entry_field]),
                        exit_time=rmsr_support._timestamp(raw[exit_field]),
                        side=rmsr_support._side(raw["side"]),
                    )
                    if event.exit_time <= event.entry_time:
                        raise RuntimeError(f"invalid comparator interval: {name}")
                    raw_groups[group].append(event)
                    if include:
                        included_for_spec += 1
                        required_groups[group].append(event)
            if artifact_rows == 0:
                raise RuntimeError(f"comparator artifact is empty: {name}")
            if included_for_spec == 0:
                raise RuntimeError(f"required comparator group is empty: {name}")
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            raise ComparatorValidationError(str(exc), rows_read=rows_read) from exc
    if not required_groups:
        raise ComparatorValidationError(
            "RCRE comparator cohort is empty", rows_read=rows_read
        )
    raw_counts = {name: len(rows) for name, rows in sorted(raw_groups.items())}
    _validate_raw_groups(raw_groups, rows_read=rows_read)
    contained, window_counts = _filter_required_groups(
        required_groups,
        rows_read=rows_read,
        raw_group_counts=raw_counts,
    )
    return contained, rows_read, window_counts, raw_counts


def one_to_one_matches(
    left: Sequence[datetime], right: Sequence[datetime], tolerance: timedelta
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
    size = int((SELECTION_END - TRAIN_START).total_seconds() // BAR.total_seconds())
    values = np.zeros(size, dtype=np.float64)
    for event in events:
        if not (
            event.entry_time >= TRAIN_START and event.exit_time <= SELECTION_END
        ):
            raise RuntimeError("RCRE novelty received a non-contained interval")
        begin = int((event.entry_time - TRAIN_START).total_seconds() // 300)
        finish = int((event.exit_time - TRAIN_START).total_seconds() // 300)
        values[begin:finish] += event.side
    return values


def novelty_metrics(
    primary: Sequence[Scheduled], comparator: Sequence[ComparatorEvent]
) -> dict[str, Any]:
    primary_events = [
        ComparatorEvent(row.entry_time, row.exit_time, row.side)
        for row in primary
        if row.entry_time >= TRAIN_START and row.exit_time <= SELECTION_END
    ]
    comparator_events = list(comparator)
    a = [row.entry_time for row in primary_events]
    b = [row.entry_time for row in comparator_events]
    exact_a, exact_b = set(a), set(b)
    union = exact_a | exact_b
    matches = one_to_one_matches(a, b, timedelta(hours=24))
    left_exposure = _exposure(primary_events)
    right_exposure = _exposure(comparator_events)
    correlation: float | None
    if np.std(left_exposure) == 0 or np.std(right_exposure) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(left_exposure, right_exposure)[0, 1])
        if not math.isfinite(correlation):
            correlation = None
    return {
        "primary_entries": len(a),
        "comparator_entries": len(b),
        "exact_entry_intersection": len(exact_a & exact_b),
        "exact_entry_jaccard": len(exact_a & exact_b) / len(union) if union else 0.0,
        "one_day_one_to_one_matches": matches,
        "rcre_one_day_containment": matches / len(a) if a else 0.0,
        "comparator_one_day_containment": matches / len(b) if b else 0.0,
        "signed_5m_occupied_exposure_correlation": correlation,
    }


def _clock_row(row: Scheduled) -> dict[str, str]:
    components = row.components
    units = row.units
    return {
        "control": row.control,
        "observation_date": row.observation_date.isoformat(),
        "signal_time": row.signal_time.isoformat(),
        "entry_time": row.entry_time.isoformat(),
        "exit_time": row.exit_time.isoformat(),
        "split": row.split,
        "side": str(row.side),
        "state": str(row.state),
        "quantity_sign": str(_sign(components["quantity_gap"])),
        "rate_sign": str(_sign(components["rate_gap"])),
        "quadrant": _quadrant(components),
        "quantity_gap": str(components["quantity_gap"]),
        "rate_gap": str(components["rate_gap"]),
        "routing_pressure": str(components["routing_pressure"]),
        "score": str(row.score),
        "u_quantity_gap": str(units["quantity_gap"]),
        "u_rate_gap": str(units["rate_gap"]),
        "u_routing_pressure": str(units["routing_pressure"]),
        "u_absolute_pressure": str(units["absolute_pressure"]),
        "u_absolute_quantity_gap": str(units["absolute_quantity_gap"]),
        "u_absolute_rate_gap": str(units["absolute_rate_gap"]),
    }


def _gzip_csv(rows: Iterable[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CLOCK_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode(), compresslevel=9, mtime=0)


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RCRE output escaped repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_or_verify(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing RCRE artifact differs: {path.name}")
        return "verified_existing"
    try:
        _atomic_write(path, payload)
        return "created"
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"concurrent RCRE artifact differs: {path.name}")
        return "verified_existing"


def _candidate_window_counts(primary: Sequence[Scheduled]) -> dict[str, int]:
    counts = {
        "total_raw_rows_parsed": len(primary),
        "fully_contained_rows_used": 0,
        "rows_before_window": 0,
        "rows_after_window": 0,
        "rows_crossing_boundary": 0,
    }
    for row in primary:
        event = ComparatorEvent(row.entry_time, row.exit_time, row.side)
        counts[_window_bucket(event)] += 1
    return counts


def build_report(
    *,
    clock_output: str | Path = DEFAULT_CLOCK,
    load_comparators: Callable[
        [],
        tuple[
            dict[str, list[ComparatorEvent]],
            int,
            dict[str, dict[str, int]],
            dict[str, int],
        ],
    ] = load_comparator_groups,
    write_clock: bool = True,
) -> dict[str, Any]:
    registration = _load_registration()
    by_date, normalized_rows, required_rows = rmsr_support.load_source()
    features, source_audit = build_features(
        by_date,
        normalized_rows_read=normalized_rows,
        required_rows_read=required_rows,
    )
    rank_rows = build_rank_rows(features)
    clocks = build_clocks(features, rank_rows)
    post_2023_source_rows = sum(day > END_DATE for day in by_date)
    source_checks, summaries = source_support(
        clocks,
        registration["policy"],
        source_audit,
        post_2023_source_rows=post_2023_source_rows,
    )
    source_passed = all(source_checks.values())
    comparator_rows_read = 0
    comparator_window_counts: dict[str, dict[str, int]] = {}
    raw_comparator_group_counts: dict[str, int] = {}
    novelty: dict[str, Any] = {
        "evaluated": False,
        "reason": "source support failed before comparator access",
        "metrics": {},
        "checks": {},
        "qualifying_groups": [],
        "passed": False,
    }
    if source_passed:
        try:
            (
                groups,
                comparator_rows_read,
                comparator_window_counts,
                raw_comparator_group_counts,
            ) = load_comparators()
        except ComparatorValidationError as exc:
            comparator_rows_read = exc.rows_read
            comparator_window_counts = dict(exc.window_counts)
            raw_comparator_group_counts = dict(exc.raw_group_counts)
            novelty = {
                "evaluated": False,
                "reason": "comparator validation failed closed",
                "validation_error": str(exc),
                "metrics": {},
                "checks": {},
                "qualifying_groups": [],
                "passed": False,
            }
        else:
            metrics = {
                name: novelty_metrics(clocks["primary"], rows)
                for name, rows in groups.items()
            }
            limits = registration["policy"]["novelty"]
            minimum = limits["minimum_comparator_entries"]
            qualifying = sorted(
                name
                for name, row in metrics.items()
                if row["comparator_entries"] >= minimum
            )
            if not qualifying:
                novelty = {
                    "evaluated": False,
                    "reason": "no qualifying comparator groups",
                    "metrics": metrics,
                    "checks": {},
                    "qualifying_groups": [],
                    "passed": False,
                }
            else:
                checks: dict[str, dict[str, bool]] = {}
                for name in qualifying:
                    row = metrics[name]
                    correlation = row["signed_5m_occupied_exposure_correlation"]
                    checks[name] = {
                        "exact_entry_jaccard": row["exact_entry_jaccard"]
                        <= limits["maximum_exact_entry_jaccard"],
                        "one_day_containment": row["rcre_one_day_containment"]
                        <= limits["maximum_rcre_one_day_containment"],
                        "signed_exposure_correlation": correlation is not None
                        and abs(correlation)
                        <= limits["maximum_absolute_signed_exposure_correlation"],
                    }
                novelty = {
                    "evaluated": True,
                    "reason": "source support passed",
                    "metrics": metrics,
                    "checks": checks,
                    "qualifying_groups": qualifying,
                    "passed": all(all(row.values()) for row in checks.values()),
                }
    clock_payload = _gzip_csv(
        _clock_row(row) for control in CONTROL_NAMES for row in clocks[control]
    )
    if write_clock:
        _write_or_verify(_repository_path(clock_output), clock_payload)
    control_summaries = {
        name: {
            "train": _summary(rows, TRAIN_START, TRAIN_END),
            "selection": _summary(rows, SELECTION_START, SELECTION_END),
        }
        for name, rows in clocks.items()
    }
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "support_builder": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "parser_dependencies": [
            {"path": str(path), "sha256": digest}
            for path, digest in PARSER_DEPENDENCIES
        ],
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "source_audit": asdict(source_audit),
        "rank_ready_rows": len(rank_rows),
        "decision_state_rows": sum(
            row.feature.decision_allowed for row in rank_rows
        ),
        "clock_summaries": control_summaries,
        "primary_support_summaries": summaries,
        "source_checks": source_checks,
        "source_support_passed": source_passed,
        "control_diagnostics": {
            "quantity_gap_label_pair_exact": _label_pair_valid(
                clocks["quantity_gap_original"], clocks["quantity_gap_swapped"]
            ),
            "rate_gap_label_pair_exact": _label_pair_valid(
                clocks["rate_gap_original"], clocks["rate_gap_swapped"]
            ),
            "label_pair_controls_economically_falsifying": False,
        },
        "novelty": novelty,
        "common_window_audit": {
            "policy": {
                "path": str(prereg.COMMON_WINDOW_POLICY),
                "sha256": prereg.COMMON_WINDOW_POLICY_SHA256,
            },
            "window": [TRAIN_START.isoformat(), SELECTION_END.isoformat()],
            "candidate": _candidate_window_counts(clocks["primary"]),
            "comparators": comparator_window_counts,
            "raw_comparator_group_counts": raw_comparator_group_counts,
            "intervals_clipped": 0,
        },
        "clock_artifact": {
            "path": str(clock_output),
            "sha256": hashlib.sha256(clock_payload).hexdigest(),
            "rows": sum(len(rows) for rows in clocks.values()),
            "primary_rows": len(clocks["primary"]),
            "columns": list(CLOCK_COLUMNS),
        },
        "outcome_boundary": {
            "source_observation_rows_read": normalized_rows,
            "signed_features_and_rcre_incidence_opened": True,
            "comparator_rows_read": comparator_rows_read,
            "comparator_access_short_circuited_on_source_failure": not source_passed,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "post_2023_source_rows_read": post_2023_source_rows,
        },
        "advance_to_evaluator_freeze": bool(source_passed and novelty["passed"]),
        "disposition": (
            "advance RCRE-72 unchanged to evaluator freeze"
            if source_passed and novelty["passed"]
            else "reject RCRE-72 unchanged before outcomes"
        ),
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)
    report = build_report(clock_output=args.clock_output)
    _write_or_verify(
        _repository_path(args.report_output),
        (
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode(),
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "source_support_passed": report["source_support_passed"],
                "novelty_evaluated": report["novelty"]["evaluated"],
                "novelty_passed": report["novelty"]["passed"],
                "advance_to_evaluator_freeze": report[
                    "advance_to_evaluator_freeze"
                ],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
