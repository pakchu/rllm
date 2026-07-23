"""Build outcome-blind DMSH-168 source-support and causal control clocks."""

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
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_ofr_dvp_maturity_stock_flow_handoff as prereg


PROTOCOL_VERSION = "ofr_dvp_maturity_stock_flow_handoff_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_ofr_dvp_maturity_stock_flow_handoff_support.py")
TEST_PATH = Path("tests/test_build_ofr_dvp_maturity_stock_flow_handoff_support.py")
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "3351845c3a18e6e9af30cdc58b7ce6f71dc9d55674b3fa6e330e521ee1e4c475"
)
PREREGISTRATION_MANIFEST_HASH = (
    "8c958e00649db244aff147c82c9ed1b9631ca4d015bdf8ce8085383e0619c678"
)
DEFAULT_CLOCK = Path(
    "results/ofr_dvp_maturity_stock_flow_handoff_clocks_2026-07-23.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/ofr_dvp_maturity_stock_flow_handoff_support_2026-07-23.json"
)

UTC = timezone.utc
BAR = timedelta(minutes=5)
HOLD = timedelta(hours=168)
LOOKBACK = 252
MAX_CONFIRMATION_AGE = 10
START_DATE = date(2019, 1, 1)
END_DATE = date(2023, 12, 31)
FEED_FLOOR = datetime(2020, 9, 10, tzinfo=UTC)
TRAIN_START = datetime(2021, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
SELECTION_START = TRAIN_END
SELECTION_END = datetime(2024, 1, 1, tzinfo=UTC)

SOURCE_COLUMNS = (
    "mnemonic",
    "observation_date",
    "available_at_utc",
    "value",
    "disclosure_edit",
    "segment",
    "measure",
    "subset",
    "series_name",
)
SCHEDULED_SOURCE_CONTROLS = (
    "flow_transition_only",
    "curve_transition_only",
    "same_date_conjunction",
    "reverse_order_handoff",
    "five_date_window",
    "twenty_date_window",
    "one_complete_date_stale",
    "five_complete_date_stale",
)
ECONOMIC_SIDE_CONTROLS = (
    "exact_direction_flip",
    "deterministic_random_side",
    "constant_long",
    "constant_short",
)
CLOCK_NAMES = ("primary", *SCHEDULED_SOURCE_CONTROLS, *ECONOMIC_SIDE_CONTROLS)
PLACEBO_NAMES = (
    "year_curve_permutation_placebo",
    "year_flow_permutation_placebo",
)
CLOCK_COLUMNS = (
    "control",
    "precursor_observation_date",
    "confirmation_observation_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "split",
    "side",
    "polarity",
    "confirmation_age_rows",
    "dominant_rate_bucket",
    "maturity_flow_gap",
    "curve_gap",
    "u_maturity_flow_gap",
    "u_curve_gap",
)


@dataclass(frozen=True)
class SourceRow:
    mnemonic: str
    observation_date: date
    available_at: datetime
    value: Fraction | None
    disclosure_edit: bool


@dataclass(frozen=True)
class FeatureRow:
    observation_date: date
    available_at: datetime
    epoch: int
    decision_allowed: bool
    components: Mapping[str, Fraction]
    dominant_rate_bucket: str


@dataclass(frozen=True)
class RankRow:
    feature: FeatureRow
    units: Mapping[str, Fraction]


@dataclass(frozen=True)
class StateRow:
    rank: RankRow
    epoch: int
    flow_state: int
    curve_state: int


@dataclass(frozen=True)
class Pending:
    precursor: StateRow
    polarity: int
    age: int = 0


@dataclass(frozen=True)
class Candidate:
    precursor: StateRow
    confirmation: StateRow
    polarity: int
    age: int


@dataclass(frozen=True)
class Scheduled:
    control: str
    precursor_observation_date: date
    confirmation_observation_date: date
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    split: str
    side: int
    polarity: int
    confirmation_age_rows: int
    dominant_rate_bucket: str
    components: Mapping[str, Fraction]
    units: Mapping[str, Fraction]


@dataclass(frozen=True)
class SourceAudit:
    normalized_rows_read: int
    required_rows_read: int
    source_dates_seen: int
    valid_feature_dates: int
    invalid_missing_null_or_edit_dates: int
    invalid_negative_volume_dates: int
    invalid_denominator_dates: int
    equal_availability_rows_suppressed: int


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("DMSH support path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("DMSH support path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("DMSH timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _fraction(value: str, label: str, *, optional: bool = False) -> Fraction | None:
    text = value.strip()
    if optional and not text:
        return None
    if not text or text.lower() in {"nan", "inf", "+inf", "-inf"}:
        raise RuntimeError(f"invalid DMSH decimal: {label}")
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"invalid DMSH decimal: {label}") from exc


def _expected_availability(day: date) -> datetime:
    delayed = datetime.combine(day + timedelta(days=8), time(), UTC)
    return max(delayed, FEED_FLOOR)


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_protocol_committed() -> None:
    paths = (SCRIPT_PATH, TEST_PATH)
    if not all(_repository_path(path).is_file() for path in paths):
        raise RuntimeError("DMSH source-support protocol file is missing")
    labels = tuple(str(path) for path in paths)
    tracked = _git_check("ls-files", "--error-unmatch", "--", *labels)
    if tracked.returncode != 0:
        raise RuntimeError("DMSH source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *labels)
    if clean.returncode != 0:
        raise RuntimeError("DMSH source-support protocol differs from HEAD")


def _load_registration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("DMSH preregistration file hash mismatch")
    payload = json.loads(_repository_path(PREREGISTRATION).read_text(encoding="utf-8"))
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("DMSH preregistration manifest hash mismatch")
    if payload.get("candidate") != prereg.POLICY_ID:
        raise RuntimeError("DMSH preregistration candidate mismatch")
    if payload.get("policy_hash") != prereg.canonical_hash(payload.get("policy")):
        raise RuntimeError("DMSH preregistration policy hash mismatch")
    for field in (
        "candidate_features_or_incidence_opened",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"DMSH preregistration boundary opened: {field}")
    return payload


def load_source() -> tuple[dict[date, dict[str, SourceRow]], int, int]:
    if sha256_file(prereg.OBSERVATIONS) != prereg.OBSERVATIONS_SHA256:
        raise RuntimeError("DMSH source observation hash mismatch")
    required = set(prereg.REQUIRED_SERIES)
    by_date: dict[date, dict[str, SourceRow]] = defaultdict(dict)
    normalized_rows = 0
    required_rows = 0
    with gzip.open(_repository_path(prereg.OBSERVATIONS), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise RuntimeError("DMSH source columns changed")
        for raw in reader:
            normalized_rows += 1
            try:
                observation_day = date.fromisoformat(raw["observation_date"])
            except ValueError as exc:
                raise RuntimeError("DMSH observation date changed") from exc
            if not START_DATE <= observation_day <= END_DATE:
                raise RuntimeError("DMSH source date escaped frozen window")
            mnemonic = raw["mnemonic"]
            if mnemonic not in required:
                continue
            required_rows += 1
            available_at = _timestamp(raw["available_at_utc"])
            if available_at != _expected_availability(observation_day):
                raise RuntimeError("DMSH source availability changed")
            if mnemonic in by_date[observation_day]:
                raise RuntimeError("duplicate DMSH required source row")
            disclosure = raw["disclosure_edit"].strip()
            if disclosure not in {"0", "1"}:
                raise RuntimeError("DMSH disclosure-edit flag changed")
            by_date[observation_day][mnemonic] = SourceRow(
                mnemonic=mnemonic,
                observation_date=observation_day,
                available_at=available_at,
                value=_fraction(raw["value"], f"{mnemonic} value", optional=True),
                disclosure_edit=disclosure == "1",
            )
    if normalized_rows != 77_369:
        raise RuntimeError("DMSH normalized source row count changed")
    if not by_date:
        raise RuntimeError("DMSH required source is empty")
    return dict(by_date), normalized_rows, required_rows


def _required_value(rows: Mapping[str, SourceRow], mnemonic: str) -> Fraction:
    value = rows[mnemonic].value
    if value is None:
        raise RuntimeError("required DMSH value is null")
    return value


def _dominant_rate_bucket(le30: Fraction, g30: Fraction) -> str:
    left = abs(le30)
    right = abs(g30)
    if left > right:
        return "LE30"
    if right > left:
        return "G30"
    return "TIE"


def build_features(
    by_date: Mapping[date, Mapping[str, SourceRow]],
    *,
    normalized_rows_read: int = 0,
    required_rows_read: int = 0,
) -> tuple[list[FeatureRow], SourceAudit]:
    required = set(prereg.REQUIRED_SERIES)
    features: list[FeatureRow] = []
    epoch = 0
    invalid_missing = 0
    invalid_negative = 0
    invalid_denominator = 0
    for observation_day in sorted(by_date):
        rows = by_date[observation_day]
        if set(rows) != required or any(
            row.value is None or row.disclosure_edit for row in rows.values()
        ):
            epoch += 1
            invalid_missing += 1
            continue
        value = lambda mnemonic: _required_value(rows, mnemonic)
        volumes = {
            mnemonic: value(mnemonic)
            for mnemonic in prereg.REQUIRED_SERIES
            if "_OV_" in mnemonic or "_TV_" in mnemonic
        }
        if any(item < 0 for item in volumes.values()):
            epoch += 1
            invalid_negative += 1
            continue
        ov_total = (
            value("REPO-DVP_OV_OO-P")
            + value("REPO-DVP_OV_LE30-P")
            + value("REPO-DVP_OV_G30-P")
        )
        tv_total = (
            value("REPO-DVP_TV_OO-P")
            + value("REPO-DVP_TV_LE30-P")
            + value("REPO-DVP_TV_G30-P")
        )
        term_tv = value("REPO-DVP_TV_LE30-P") + value("REPO-DVP_TV_G30-P")
        if ov_total <= 0 or tv_total <= 0 or term_tv <= 0:
            epoch += 1
            invalid_denominator += 1
            continue
        stock_overnight_share = value("REPO-DVP_OV_OO-P") / ov_total
        flow_overnight_share = value("REPO-DVP_TV_OO-P") / tv_total
        maturity_flow_gap = flow_overnight_share - stock_overnight_share
        le30_contribution = value("REPO-DVP_AR_LE30-P") * value(
            "REPO-DVP_TV_LE30-P"
        )
        g30_contribution = value("REPO-DVP_AR_G30-P") * value(
            "REPO-DVP_TV_G30-P"
        )
        term_rate = (le30_contribution + g30_contribution) / term_tv
        curve_gap = term_rate - value("REPO-DVP_AR_OO-P")
        features.append(
            FeatureRow(
                observation_date=observation_day,
                available_at=max(row.available_at for row in rows.values()),
                epoch=epoch,
                decision_allowed=False,
                components={
                    "maturity_flow_gap": maturity_flow_gap,
                    "curve_gap": curve_gap,
                },
                dominant_rate_bucket=_dominant_rate_bucket(
                    le30_contribution, g30_contribution
                ),
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
        invalid_missing_null_or_edit_dates=invalid_missing,
        invalid_negative_volume_dates=invalid_negative,
        invalid_denominator_dates=invalid_denominator,
        equal_availability_rows_suppressed=sum(
            not row.decision_allowed for row in features
        ),
    )
    return features, audit


def midrank_unit(current: Fraction, prior: Sequence[Fraction]) -> Fraction:
    if len(prior) != LOOKBACK:
        raise RuntimeError("DMSH midrank history length changed")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return Fraction(2 * less + equal - LOOKBACK, LOOKBACK)


def build_rank_rows(features: Sequence[FeatureRow]) -> list[RankRow]:
    histories = {name: deque(maxlen=LOOKBACK) for name in prereg.COMPONENTS}
    batches: dict[datetime, list[FeatureRow]] = defaultdict(list)
    for row in features:
        batches[row.available_at].append(row)
    output: list[RankRow] = []
    for available_at in sorted(batches):
        batch = sorted(batches[available_at], key=lambda row: row.observation_date)
        if all(len(history) == LOOKBACK for history in histories.values()):
            prior = {name: tuple(history) for name, history in histories.items()}
            for row in batch:
                output.append(
                    RankRow(
                        feature=row,
                        units={
                            name: midrank_unit(row.components[name], prior[name])
                            for name in prereg.COMPONENTS
                        },
                    )
                )
        for row in batch:
            for name in prereg.COMPONENTS:
                histories[name].append(row.components[name])
    return output


def component_state(unit: Fraction) -> int:
    if unit >= Fraction(1, 2):
        return 1
    if unit <= Fraction(-1, 2):
        return -1
    return 0


def build_state_rows(rank_rows: Sequence[RankRow]) -> list[StateRow]:
    return [
        StateRow(
            rank=row,
            epoch=row.feature.epoch,
            flow_state=component_state(row.units["maturity_flow_gap"]),
            curve_state=component_state(row.units["curve_gap"]),
        )
        for row in rank_rows
        if row.feature.decision_allowed
    ]


def _transition(current: int, previous: int) -> int:
    return current if current != 0 and current != previous else 0


def derive_handoff_candidates(
    rows: Sequence[StateRow],
    *,
    precursor_field: str = "flow_state",
    confirmation_field: str = "curve_state",
    window: int = MAX_CONFIRMATION_AGE,
) -> list[Candidate]:
    if window <= 0:
        raise RuntimeError("DMSH confirmation window must be positive")
    output: list[Candidate] = []
    previous: StateRow | None = None
    pending: Pending | None = None
    for current in rows:
        if previous is None or previous.epoch != current.epoch:
            pending = None
            previous = current
            continue
        precursor_transition = _transition(
            getattr(current, precursor_field), getattr(previous, precursor_field)
        )
        confirmation_transition = _transition(
            getattr(current, confirmation_field),
            getattr(previous, confirmation_field),
        )
        had_pending = pending is not None
        if pending is not None:
            polarity = pending.polarity
            age = pending.age + 1
            contradiction = (
                precursor_transition == -polarity
                or confirmation_transition == -polarity
            )
            if contradiction:
                pending = None
            elif confirmation_transition == polarity:
                output.append(
                    Candidate(
                        precursor=pending.precursor,
                        confirmation=current,
                        polarity=polarity,
                        age=age,
                    )
                )
                pending = None
            elif age >= window:
                pending = None
            else:
                pending = replace(pending, age=age)
        if not had_pending and precursor_transition in {-1, 1}:
            polarity = precursor_transition
            if getattr(current, confirmation_field) != polarity:
                pending = Pending(precursor=current, polarity=polarity)
        previous = current
    return output


def transition_candidates(rows: Sequence[StateRow], field: str) -> list[Candidate]:
    output: list[Candidate] = []
    previous: StateRow | None = None
    for current in rows:
        if previous is None or previous.epoch != current.epoch:
            previous = current
            continue
        polarity = _transition(getattr(current, field), getattr(previous, field))
        if polarity:
            output.append(
                Candidate(
                    precursor=current,
                    confirmation=current,
                    polarity=polarity,
                    age=0,
                )
            )
        previous = current
    return output


def same_date_conjunction_candidates(rows: Sequence[StateRow]) -> list[Candidate]:
    output: list[Candidate] = []
    previous: StateRow | None = None
    for current in rows:
        if previous is None or previous.epoch != current.epoch:
            previous = current
            continue
        flow = _transition(current.flow_state, previous.flow_state)
        curve = _transition(current.curve_state, previous.curve_state)
        if flow in {-1, 1} and flow == curve:
            output.append(
                Candidate(
                    precursor=current,
                    confirmation=current,
                    polarity=flow,
                    age=0,
                )
            )
        previous = current
    return output


def stale_state_rows(
    rows: Sequence[StateRow], lag: int
) -> list[StateRow]:
    if lag <= 0:
        raise RuntimeError("DMSH stale lag must be positive")
    output: list[StateRow] = []
    synthetic_epoch = 0
    previous_destination: int | None = None
    for index, destination in enumerate(rows):
        source_index = index - lag
        if source_index < 0 or rows[source_index].epoch != destination.epoch:
            previous_destination = None
            continue
        if previous_destination != index - 1:
            synthetic_epoch += 1
        source = rows[source_index]
        output.append(
            StateRow(
                rank=destination.rank,
                epoch=synthetic_epoch,
                flow_state=source.flow_state,
                curve_state=source.curve_state,
            )
        )
        previous_destination = index
    return output


def _ceil_5m(value: datetime) -> datetime:
    epoch = int(value.timestamp())
    rounded = ((epoch + 299) // 300) * 300
    return datetime.fromtimestamp(rounded, tz=UTC)


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
        key=lambda row: (
            row.confirmation.rank.feature.available_at,
            row.confirmation.rank.feature.observation_date,
        ),
    ):
        signal = candidate.confirmation.rank.feature.available_at
        entry = _ceil_5m(signal) + BAR
        exit_time = entry + HOLD
        split = _split(entry, exit_time)
        if split is None or (reserved_until is not None and entry < reserved_until):
            continue
        confirmation = candidate.confirmation.rank
        output.append(
            Scheduled(
                control=control,
                precursor_observation_date=(
                    candidate.precursor.rank.feature.observation_date
                ),
                confirmation_observation_date=(
                    confirmation.feature.observation_date
                ),
                signal_time=signal,
                entry_time=entry,
                exit_time=exit_time,
                split=split,
                side=-candidate.polarity,
                polarity=candidate.polarity,
                confirmation_age_rows=candidate.age,
                dominant_rate_bucket=confirmation.feature.dominant_rate_bucket,
                components=confirmation.feature.components,
                units=confirmation.units,
            )
        )
        reserved_until = exit_time
    return output


def _canonical_utc_second(value: datetime) -> str:
    normalized = value.astimezone(UTC)
    if normalized.microsecond:
        raise RuntimeError("DMSH entry time must be whole-second")
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _random_side(entry: datetime) -> int:
    token = (
        "DMSH-168|deterministic_random_side|" + _canonical_utc_second(entry)
    ).encode("utf-8")
    return 1 if hashlib.sha256(token).digest()[0] < 128 else -1


def _clone_clock(
    rows: Sequence[Scheduled], control: str, side_fn
) -> list[Scheduled]:
    return [replace(row, control=control, side=side_fn(row)) for row in rows]


def build_clocks(rows: Sequence[StateRow]) -> dict[str, list[Scheduled]]:
    candidates = {
        "primary": derive_handoff_candidates(rows),
        "flow_transition_only": transition_candidates(rows, "flow_state"),
        "curve_transition_only": transition_candidates(rows, "curve_state"),
        "same_date_conjunction": same_date_conjunction_candidates(rows),
        "reverse_order_handoff": derive_handoff_candidates(
            rows,
            precursor_field="curve_state",
            confirmation_field="flow_state",
        ),
        "five_date_window": derive_handoff_candidates(rows, window=5),
        "twenty_date_window": derive_handoff_candidates(rows, window=20),
        "one_complete_date_stale": derive_handoff_candidates(
            stale_state_rows(rows, 1)
        ),
        "five_complete_date_stale": derive_handoff_candidates(
            stale_state_rows(rows, 5)
        ),
    }
    clocks = {name: schedule(name, values) for name, values in candidates.items()}
    primary = clocks["primary"]
    clocks["exact_direction_flip"] = _clone_clock(
        primary, "exact_direction_flip", lambda row: -row.side
    )
    clocks["deterministic_random_side"] = _clone_clock(
        primary, "deterministic_random_side", lambda row: _random_side(row.entry_time)
    )
    clocks["constant_long"] = _clone_clock(
        primary, "constant_long", lambda _row: 1
    )
    clocks["constant_short"] = _clone_clock(
        primary, "constant_short", lambda _row: -1
    )
    if set(clocks) != set(CLOCK_NAMES):
        raise RuntimeError("DMSH control clock set changed")
    return clocks


def _permutation_key(mode: str, year: int, row: FeatureRow) -> bytes:
    return hashlib.sha256(
        f"DMSH-168|{mode}|{year}|{row.observation_date.isoformat()}".encode(
            "utf-8"
        )
    ).digest()


def permute_features(
    features: Sequence[FeatureRow], mode: str
) -> list[FeatureRow]:
    if mode not in PLACEBO_NAMES:
        raise RuntimeError("unknown DMSH placebo mode")
    seed_mode = mode.removesuffix("_placebo")
    component = "curve_gap" if "curve" in mode else "maturity_flow_gap"
    by_year: dict[int, list[FeatureRow]] = defaultdict(list)
    for row in features:
        by_year[row.observation_date.year].append(row)
    output: list[FeatureRow] = []
    for year in sorted(by_year):
        destinations = sorted(by_year[year], key=lambda row: row.observation_date)
        sources = sorted(
            destinations,
            key=lambda row: (
                _permutation_key(seed_mode, year, row),
                row.observation_date,
            ),
        )
        for destination, source in zip(destinations, sources):
            components = dict(destination.components)
            components[component] = source.components[component]
            output.append(replace(destination, components=components))
    return sorted(output, key=lambda row: row.observation_date)


def placebo_incidence(features: Sequence[FeatureRow]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in PLACEBO_NAMES:
        ranks = build_rank_rows(permute_features(features, name))
        candidates = derive_handoff_candidates(build_state_rows(ranks))
        counts = Counter(
            "train"
            if candidate.confirmation.rank.feature.observation_date.year in {2021, 2022}
            else "selection"
            if candidate.confirmation.rank.feature.observation_date.year == 2023
            else "outside"
            for candidate in candidates
        )
        output[name] = {
            "causal": False,
            "economic_evaluation_forbidden": True,
            "execution_clock_emitted": False,
            "raw_candidate_incidence": dict(sorted(counts.items())),
        }
    return output


def _month_share(rows: Sequence[Scheduled]) -> Fraction:
    if not rows:
        return Fraction(1)
    counts = Counter(row.entry_time.strftime("%Y-%m") for row in rows)
    return Fraction(max(counts.values()), len(rows))


def _elapsed_days(left: datetime, right: datetime) -> Fraction:
    elapsed = right - left
    microseconds = (
        (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000
        + elapsed.microseconds
    )
    return Fraction(microseconds, 86_400 * 1_000_000)


def _max_gap_days(rows: Sequence[Scheduled]) -> Fraction | None:
    ordered = sorted(row.entry_time for row in rows)
    if len(ordered) < 2:
        return None
    return max(
        _elapsed_days(left, right)
        for left, right in zip(ordered, ordered[1:])
    )


def _summary(rows: Sequence[Scheduled], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row.split == split]
    years = Counter(row.entry_time.year for row in selected)
    halves = Counter(
        f"{row.entry_time.year}-H{1 if row.entry_time.month <= 6 else 2}"
        for row in selected
    )
    quarters = Counter(
        f"{row.entry_time.year}-Q{(row.entry_time.month - 1) // 3 + 1}"
        for row in selected
    )
    sides = Counter("LONG" if row.side == 1 else "SHORT" for row in selected)
    polarities = Counter(str(row.polarity) for row in selected)
    ages = Counter(
        "1-3"
        if 1 <= row.confirmation_age_rows <= 3
        else "4-6"
        if 4 <= row.confirmation_age_rows <= 6
        else "7-10"
        if 7 <= row.confirmation_age_rows <= 10
        else "other"
        for row in selected
    )
    buckets = Counter(row.dominant_rate_bucket for row in selected)
    dominant_bucket_share = (
        Fraction(
            max(buckets.get("LE30", 0), buckets.get("G30", 0)),
            len(selected),
        )
        if selected
        else Fraction(1)
    )
    return {
        "events": len(selected),
        "years": dict(sorted(years.items())),
        "halves": dict(sorted(halves.items())),
        "quarters": dict(sorted(quarters.items())),
        "sides": dict(sorted(sides.items())),
        "precursor_polarities": dict(sorted(polarities.items())),
        "confirmation_age_bins": dict(sorted(ages.items())),
        "dominant_rate_buckets": dict(sorted(buckets.items())),
        "maximum_single_rate_bucket_share": dominant_bucket_share,
        "maximum_month_share": _month_share(selected),
        "maximum_entry_gap_elapsed_days": _max_gap_days(selected),
    }


def _serialize_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _fraction_text(value) if isinstance(value, Fraction) else value
        for key, value in summary.items()
    }


def _clock_valid(rows: Sequence[Scheduled]) -> bool:
    for left, right in zip(rows, rows[1:]):
        if left.entry_time >= right.entry_time or left.exit_time > right.entry_time:
            return False
    return all(
        row.side in {-1, 1}
        and row.polarity in {-1, 1}
        and row.entry_time == _ceil_5m(row.signal_time) + BAR
        and row.exit_time == row.entry_time + HOLD
        and _split(row.entry_time, row.exit_time) == row.split
        for row in rows
    )


def _primary_chronology_valid(rows: Sequence[Scheduled]) -> bool:
    return all(
        row.confirmation_observation_date > row.precursor_observation_date
        and 1 <= row.confirmation_age_rows <= MAX_CONFIRMATION_AGE
        for row in rows
    )


def source_support(clocks: Mapping[str, Sequence[Scheduled]]) -> dict[str, Any]:
    if set(clocks) != set(CLOCK_NAMES):
        raise RuntimeError("DMSH support clock set changed")
    clock_integrity = all(_clock_valid(clocks[name]) for name in CLOCK_NAMES)
    primary_chronology = _primary_chronology_valid(clocks["primary"])
    primary = clocks["primary"]
    train = _summary(primary, "train")
    selection = _summary(primary, "selection")
    expected_train_halves = {f"{year}-H{half}" for year in (2021, 2022) for half in (1, 2)}
    expected_selection_halves = {"2023-H1", "2023-H2"}
    expected_quarters = {
        f"{year}-Q{quarter}"
        for year in (2021, 2022, 2023)
        for quarter in range(1, 5)
    }
    train_gap = train["maximum_entry_gap_elapsed_days"]
    selection_gap = selection["maximum_entry_gap_elapsed_days"]
    gates = {
        "train_total": train["events"] >= 40,
        "each_train_year": all(
            train["years"].get(year, 0) >= 16 for year in (2021, 2022)
        ),
        "each_train_half": all(
            train["halves"].get(key, 0) >= 7 for key in expected_train_halves
        ),
        "train_each_side": all(
            train["sides"].get(side, 0) >= 10 for side in ("LONG", "SHORT")
        ),
        "selection_total": selection["events"] >= 18,
        "each_selection_half": all(
            selection["halves"].get(key, 0) >= 6
            for key in expected_selection_halves
        ),
        "selection_each_side": all(
            selection["sides"].get(side, 0) >= 4
            for side in ("LONG", "SHORT")
        ),
        "every_quarter_active": all(
            (
                train["quarters"]
                if key[:4] in {"2021", "2022"}
                else selection["quarters"]
            ).get(key, 0)
            > 0
            for key in expected_quarters
        ),
        "train_month_concentration": (
            train["maximum_month_share"] <= Fraction(1, 5)
        ),
        "selection_month_concentration": (
            selection["maximum_month_share"] <= Fraction(1, 4)
        ),
        "train_maximum_gap": train_gap is not None and train_gap <= Fraction(120),
        "selection_maximum_gap": (
            selection_gap is not None and selection_gap <= Fraction(120)
        ),
        "train_each_precursor_polarity": all(
            Fraction(
                train["precursor_polarities"].get(str(polarity), 0),
                max(train["events"], 1),
            )
            >= Fraction(1, 5)
            for polarity in (-1, 1)
        ),
        "selection_each_precursor_polarity": all(
            Fraction(
                selection["precursor_polarities"].get(str(polarity), 0),
                max(selection["events"], 1),
            )
            >= Fraction(3, 20)
            for polarity in (-1, 1)
        ),
        "train_all_confirmation_age_bins": all(
            train["confirmation_age_bins"].get(key, 0) > 0
            for key in ("1-3", "4-6", "7-10")
        ),
        "train_rate_bucket_dominance": (
            train["maximum_single_rate_bucket_share"] <= Fraction(17, 20)
        ),
        "selection_rate_bucket_dominance": (
            selection["maximum_single_rate_bucket_share"] <= Fraction(17, 20)
        ),
        "exact_clock_integrity": clock_integrity and primary_chronology,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "train": _serialize_summary(train),
        "selection": _serialize_summary(selection),
        "control_event_counts": {
            name: len(clocks[name]) for name in CLOCK_NAMES
        },
    }


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _clock_row(row: Scheduled) -> dict[str, str]:
    return {
        "control": row.control,
        "precursor_observation_date": row.precursor_observation_date.isoformat(),
        "confirmation_observation_date": row.confirmation_observation_date.isoformat(),
        "signal_time": row.signal_time.isoformat(),
        "entry_time": row.entry_time.isoformat(),
        "exit_time": row.exit_time.isoformat(),
        "split": row.split,
        "side": str(row.side),
        "polarity": str(row.polarity),
        "confirmation_age_rows": str(row.confirmation_age_rows),
        "dominant_rate_bucket": row.dominant_rate_bucket,
        "maturity_flow_gap": _fraction_text(row.components["maturity_flow_gap"]),
        "curve_gap": _fraction_text(row.components["curve_gap"]),
        "u_maturity_flow_gap": _fraction_text(row.units["maturity_flow_gap"]),
        "u_curve_gap": _fraction_text(row.units["curve_gap"]),
    }


def _gzip_csv(rows: Iterable[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CLOCK_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        handle.write(stream.getvalue().encode("utf-8"))
    return output.getvalue()


def _serialized_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("DMSH support output must remain inside repository") from exc
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
    resolved = _repository_path(path)
    if resolved.exists():
        if resolved.read_bytes() != payload:
            raise RuntimeError(f"existing DMSH support artifact differs: {path}")
        return "verified_existing"
    try:
        _atomic_write(resolved, payload)
        return "created"
    except FileExistsError:
        if resolved.read_bytes() != payload:
            raise RuntimeError(f"concurrent DMSH support artifact differs: {path}")
        return "verified_existing"


def build_report(
    *,
    audit: SourceAudit,
    features: Sequence[FeatureRow],
    ranks: Sequence[RankRow],
    clocks: Mapping[str, Sequence[Scheduled]],
    support: Mapping[str, Any],
    placebos: Mapping[str, Any],
    clock_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "support_builder": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_audit": asdict(audit),
        "rank_ready_rows": len(ranks),
        "decision_rows": sum(row.feature.decision_allowed for row in ranks),
        "source_support": dict(support),
        "noncausal_placebos": dict(placebos),
        "clock_artifact": {
            "path": str(DEFAULT_CLOCK),
            "sha256": clock_sha256,
            "rows": sum(len(rows) for rows in clocks.values()),
            "causal_controls_only": True,
            "placebo_rows": 0,
        },
        "decision": (
            "PASS_SOURCE_SUPPORT_READY_FOR_NOVELTY"
            if support["passed"]
            else "REJECT_NO_REPAIR"
        ),
        "research_boundary": {
            "normalized_source_rows_read": audit.normalized_rows_read,
            "required_source_rows_read": audit.required_rows_read,
            "candidate_features_computed": len(features),
            "candidate_incidence_derived": True,
            "comparator_files_opened": 0,
            "comparator_rows_read": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "network_calls": 0,
            "subprocess_calls": 2,
            "subprocess_scope": "fixed git protocol-commit checks only",
        },
        "outcomes_opened": False,
        "next_action": (
            "freeze and run comparator novelty evaluator"
            if support["passed"]
            else "retire DMSH-168 unchanged; select a new mechanism"
        ),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def run(
    *,
    clock_path: Path = DEFAULT_CLOCK,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[dict[str, Any], Mapping[str, str]]:
    _assert_protocol_committed()
    _load_registration()
    source, normalized_rows, required_rows = load_source()
    features, audit = build_features(
        source,
        normalized_rows_read=normalized_rows,
        required_rows_read=required_rows,
    )
    ranks = build_rank_rows(features)
    states = build_state_rows(ranks)
    clocks = build_clocks(states)
    support = source_support(clocks)
    placebos = placebo_incidence(features)
    clock_bytes = _gzip_csv(
        _clock_row(row) for name in CLOCK_NAMES for row in clocks[name]
    )
    clock_sha256 = hashlib.sha256(clock_bytes).hexdigest()
    report = build_report(
        audit=audit,
        features=features,
        ranks=ranks,
        clocks=clocks,
        support=support,
        placebos=placebos,
        clock_sha256=clock_sha256,
    )
    report["clock_artifact"]["path"] = str(clock_path)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    report["manifest_hash"] = canonical_hash(core)
    statuses = {
        "clock": _write_or_verify(clock_path, clock_bytes),
        "report": _write_or_verify(report_path, _serialized_json(report)),
    }
    return report, statuses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, statuses = run(
        clock_path=Path(args.clock),
        report_path=Path(args.report),
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "decision": report["decision"],
                "manifest_hash": report["manifest_hash"],
                "outcomes_opened": report["outcomes_opened"],
                "statuses": statuses,
                "train_events": report["source_support"]["train"]["events"],
                "selection_events": report["source_support"]["selection"]["events"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
