"""Evaluate LVRT-72 source support and pure-clock novelty without market outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from training.evaluate_trollbox_semantic_disagreement_resolution_novelty import (
    ClockRow,
    exact_entry_jaccard,
    exposure_metrics,
    maximum_tolerant_matches,
)
from training.freeze_lvrt_comparator_cohort import (
    LIVE_CLOCK,
    MFIC_CLOCK,
    _load_afcs,
    _load_bafr,
    _load_pure_clock,
)


POLICY_ID = "LVRT-72"
PROTOCOL_VERSION = "liquidity_vacuum_replenishment_transition_source_gate_v1"
BAR = timedelta(minutes=5)
REFERENCE_BARS = 2_016
EPISODE_BARS = 12
HOLD_BARS = 72
FINALITY_BARS = 2
ENTRY_DELAY_BARS = 3
IMPLEMENTATION = Path(
    "training/evaluate_liquidity_vacuum_replenishment_transition_support.py"
)
MECHANISM_DOCUMENT = Path(
    "docs/liquidity-vacuum-replenishment-transition-mechanism-decision-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "9c2400a49b77a6e93594c65ae5bc8b17f6c676743a1fdfadf367979887dd77b9"
)
SOURCE = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
SOURCE_SHA256 = (
    "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
)
SOURCE_MANIFEST = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
COMPARATOR_FREEZE = Path("results/lvrt_comparator_cohort_freeze_2026-07-21.json")
COMPARATOR_FREEZE_SHA256 = (
    "ca1b65ab08b4e2b78454282dfdbde6db86ac7dd2972fe80c56e9e635040ecc1d"
)
COMPARATOR_FREEZE_MANIFEST_HASH = (
    "52eb4ae07b893081a9b259a4fc9e708051a2dbaf40a0b398db8ce10ee2f943fc"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/liquidity_vacuum_replenishment_transition_source_gate_2026-07-21.json"
)
DEFAULT_CLOCK_OUTPUT = Path("results/lvrt_pure_clocks_2026-07-21.csv.gz")
SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2020, 2, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2023, 1, 1, tzinfo=timezone.utc)
SELECTION_END = datetime(2024, 1, 1, tzinfo=timezone.utc)
SPLITS = (
    ("train", TRAIN_START, TRAIN_END),
    ("selection", TRAIN_END, SELECTION_END),
)
ALLOWLIST = (
    "date",
    "agg_trade_count",
    "event_notional_hhi",
    "normalized_effective_event_count",
    "signed_event_imbalance",
    "max_same_sign_run_share",
    "interarrival_burstiness",
)
FEATURES = (
    "agg_trade_count",
    "event_notional_hhi",
    "normalized_effective_event_count",
    "signed_event_imbalance",
    "max_same_sign_run_share",
    "interarrival_burstiness",
)
EXPECTED_GAP_DAYS = {
    "2020-04-15",
    "2021-02-09",
    "2021-02-24",
    "2021-05-19",
    "2022-09-06",
}
CLOCK_FIELDS = (
    "candidate_id",
    "split",
    "causal_origin",
    "decision_time",
    "availability_time",
    "entry_time",
    "exit_time",
    "side",
)


@dataclass(frozen=True)
class Candidate:
    setup_time: datetime
    confirmation_time: datetime
    availability_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    split: str | None = None


@dataclass(frozen=True)
class BuildAudit:
    setups: int
    confirmations: int
    expiries: int
    gap_cancellations: int
    active_at_end: bool


@dataclass(frozen=True)
class ScheduleAudit:
    raw_candidates: int
    split_contained_candidates: int
    split_boundary_drops: int
    overlap_suppressions: int
    accepted_candidates: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _source_gap_days(manifest: Mapping[str, Any]) -> set[str]:
    months = manifest.get("months")
    if not isinstance(months, list):
        raise ValueError("source manifest months changed")
    archives: list[dict[str, Any]] = []
    for month in months:
        if not isinstance(month, dict) or not isinstance(month.get("archives"), list):
            raise ValueError("source manifest archive schema changed")
        archives.extend(cast(list[dict[str, Any]], month["archives"]))
    gaps: set[str] = set()
    for archive in archives:
        first = int(archive["first_agg_trade_id"])
        last = int(archive["last_agg_trade_id"])
        rows = int(archive["agg_trade_rows"])
        if last - first + 1 - rows > 0:
            gaps.add(str(archive["date"]))
    for previous, current in zip(archives, archives[1:]):
        delta = int(current["first_agg_trade_id"]) - int(
            previous["last_agg_trade_id"]
        ) - 1
        if delta > 0:
            gaps.add(str(previous["date"]))
            gaps.add(str(current["date"]))
        elif delta < 0:
            raise ValueError("aggregate trade IDs overlap across source days")
    return gaps


def _strict_prior_midrank(
    values: pd.Series,
    valid: pd.Series,
    *,
    window: int = REFERENCE_BARS,
) -> pd.Series:
    if window < 1:
        raise ValueError("strict-prior rank window must be positive")
    clean = values.where(valid)
    rank_including_current = clean.rolling(
        window + 1,
        min_periods=window + 1,
    ).rank(method="average")
    return (rank_including_current - 1.0) / window


def load_source_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(MECHANISM_DOCUMENT) != MECHANISM_DOCUMENT_SHA256:
        raise ValueError("LVRT mechanism decision changed")
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise ValueError("LVRT source changed")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("LVRT source manifest changed")
    if sha256_file(COMPARATOR_FREEZE) != COMPARATOR_FREEZE_SHA256:
        raise ValueError("LVRT comparator freeze changed")

    manifest = load_json(SOURCE_MANIFEST)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("outcomes_opened") is not False:
        raise ValueError("LVRT source manifest opened outcomes")
    if manifest.get("combined_sha256") != SOURCE_SHA256:
        raise ValueError("LVRT source manifest binding changed")
    if manifest.get("rows") != 420732:
        raise ValueError("LVRT source row count changed")
    if manifest.get("columns") != [
        "date",
        "first_transact_time_ms",
        "last_transact_time_ms",
        "agg_trade_count",
        "underlying_trade_count",
        "base_volume",
        "quote_notional",
        "buy_quote_notional",
        "sell_quote_notional",
        "signed_quote_notional",
        "flow_coherence",
        "first_price",
        "last_price",
        "micro_log_return",
        "signed_price_response",
        "event_notional_mean",
        "event_notional_std",
        "event_notional_p50",
        "event_notional_p90",
        "event_notional_p99",
        "event_notional_max",
        "event_notional_hhi",
        "normalized_effective_event_count",
        "underlying_trades_per_agg_event",
        "signed_event_imbalance",
        "sign_flip_rate",
        "mean_same_sign_run_length",
        "max_same_sign_run_share",
        "interarrival_mean_ms",
        "interarrival_std_ms",
        "interarrival_burstiness",
        "buy_sell_event_size_log_ratio",
    ]:
        raise ValueError("LVRT source schema changed")
    gap_days = _source_gap_days(manifest)
    if gap_days != EXPECTED_GAP_DAYS:
        raise ValueError("LVRT source gap-day contract changed")

    frame = pd.read_csv(
        SOURCE,
        compression="gzip",
        usecols=cast(Any, list(ALLOWLIST)),
        dtype={feature: "float64" for feature in FEATURES},
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if len(frame) != 420732:
        raise ValueError("LVRT parsed source row count changed")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("LVRT source timestamps are invalid")
    if not bool((frame["date"].dt.second == 0).all()) or not bool(
        (frame["date"].dt.minute % 5 == 0).all()
    ):
        raise ValueError("LVRT source left the five-minute grid")
    values = frame[list(FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("LVRT source contains non-finite allowlisted values")
    counts = frame["agg_trade_count"].to_numpy(float)
    if bool((counts <= 0.0).any()) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("LVRT aggregate event counts are invalid")
    for field in ("event_notional_hhi", "normalized_effective_event_count"):
        if bool((frame[field] <= 0.0).any()) or bool((frame[field] > 1.0).any()):
            raise ValueError(f"LVRT {field} left (0, 1]")
    if bool((frame["max_same_sign_run_share"] <= 0.0).any()) or bool(
        (frame["max_same_sign_run_share"] > 1.0).any()
    ):
        raise ValueError("LVRT run share left (0, 1]")
    for field in ("signed_event_imbalance", "interarrival_burstiness"):
        if bool((frame[field] < -1.0).any()) or bool((frame[field] > 1.0).any()):
            raise ValueError(f"LVRT {field} left [-1, 1]")

    full_grid = pd.date_range(SOURCE_START, SELECTION_END, freq="5min", inclusive="left")
    frame = frame.set_index("date").reindex(full_grid)
    frame.index.name = "date"
    grid_index = cast(pd.DatetimeIndex, frame.index)
    present = frame["agg_trade_count"].notna()
    gap_mask = pd.Series(
        grid_index.strftime("%Y-%m-%d").isin(gap_days),
        index=grid_index,
        dtype=bool,
    )
    valid = present & ~gap_mask
    frame["source_present"] = present
    frame["gap_day"] = gap_mask
    frame["valid"] = valid

    rank_specs = {
        "r_count": "agg_trade_count",
        "r_hhi": "event_notional_hhi",
        "r_neff": "normalized_effective_event_count",
        "r_flow": "signed_event_imbalance",
        "r_run": "max_same_sign_run_share",
        "r_burst": "interarrival_burstiness",
    }
    for output, source in rank_specs.items():
        source_values = cast(pd.Series, frame[source])
        values_to_rank = source_values.abs() if output == "r_flow" else source_values
        frame[output] = _strict_prior_midrank(values_to_rank, valid)
    frame["vacuum_score"] = (
        frame["r_burst"]
        + frame["r_hhi"]
        + (1.0 - frame["r_neff"])
        + frame["r_run"]
    ) / 4.0
    frame["flow_sign"] = np.sign(frame["signed_event_imbalance"].fillna(0.0)).astype(
        np.int8
    )
    ranked = cast(
        pd.Series,
        frame[
            ["r_count", "r_hhi", "r_neff", "r_flow", "r_run", "r_burst"]
        ]
        .notna()
        .all(axis=1),
    )
    frame["ranked"] = ranked
    return frame, {
        "source_rows": 420732,
        "full_grid_rows": len(frame),
        "missing_grid_rows": int((~present).sum()),
        "gap_day_rows": int(gap_mask.sum()),
        "valid_rows": int(valid.sum()),
        "rank_ready_rows": int(ranked.sum()),
        "gap_days": sorted(gap_days),
        "source_columns_read": list(ALLOWLIST),
        "forbidden_source_columns_read": 0,
        "first_timestamp": cast(pd.Timestamp, grid_index[0]).isoformat(),
        "last_timestamp": cast(pd.Timestamp, grid_index[-1]).isoformat(),
    }


def _setup_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["ranked"]
        & frame["vacuum_score"].ge(0.90)
        & frame["r_flow"].ge(0.90)
        & frame["r_count"].le(0.25)
        & frame["flow_sign"].ne(0)
    ).to_numpy(bool)


def _replenishment_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["ranked"]
        & frame["r_flow"].ge(0.60)
        & frame["r_burst"].le(0.50)
        & frame["r_hhi"].le(0.50)
        & frame["r_neff"].ge(0.50)
        & frame["r_run"].le(0.50)
        & frame["r_count"].ge(0.50)
    ).to_numpy(bool)


def _new_candidate(
    setup_time: datetime,
    confirmation_time: datetime,
    side: int,
) -> Candidate:
    availability = confirmation_time + FINALITY_BARS * BAR
    entry = confirmation_time + ENTRY_DELAY_BARS * BAR
    return Candidate(
        setup_time=setup_time,
        confirmation_time=confirmation_time,
        availability_time=availability,
        entry_time=entry,
        exit_time=entry + HOLD_BARS * BAR,
        side=side,
    )


def build_relay_candidates(
    dates: Sequence[datetime],
    valid: np.ndarray,
    setup: np.ndarray,
    replenishment: np.ndarray,
    flow_sign: np.ndarray,
    *,
    require_flip: bool,
) -> tuple[list[Candidate], BuildAudit]:
    active_index: int | None = None
    active_sign = 0
    rows: list[Candidate] = []
    setups = 0
    expiries = 0
    gap_cancellations = 0
    for index, date in enumerate(dates):
        if not valid[index]:
            if active_index is not None:
                gap_cancellations += 1
            active_index = None
            active_sign = 0
            continue
        if active_index is not None:
            age = index - active_index
            sign_ok = flow_sign[index] == -active_sign if require_flip else True
            if age <= EPISODE_BARS and replenishment[index] and sign_ok:
                rows.append(_new_candidate(dates[active_index], date, -active_sign))
                active_index = None
                active_sign = 0
                continue
            if age >= EPISODE_BARS:
                expiries += 1
                active_index = None
                active_sign = 0
                continue
            continue
        if setup[index]:
            setups += 1
            active_index = index
            active_sign = int(flow_sign[index])
    return rows, BuildAudit(
        setups=setups,
        confirmations=len(rows),
        expiries=expiries,
        gap_cancellations=gap_cancellations,
        active_at_end=active_index is not None,
    )


def build_reverse_candidates(
    dates: Sequence[datetime],
    valid: np.ndarray,
    setup: np.ndarray,
    replenishment: np.ndarray,
    flow_sign: np.ndarray,
) -> tuple[list[Candidate], BuildAudit]:
    active_index: int | None = None
    rows: list[Candidate] = []
    starts = 0
    expiries = 0
    gaps = 0
    for index, date in enumerate(dates):
        if not valid[index]:
            if active_index is not None:
                gaps += 1
            active_index = None
            continue
        if active_index is not None:
            age = index - active_index
            if age <= EPISODE_BARS and setup[index]:
                rows.append(_new_candidate(dates[active_index], date, -int(flow_sign[index])))
                active_index = None
                continue
            if age >= EPISODE_BARS:
                expiries += 1
                active_index = None
                continue
            continue
        if replenishment[index] and flow_sign[index] != 0:
            starts += 1
            active_index = index
    return rows, BuildAudit(
        setups=starts,
        confirmations=len(rows),
        expiries=expiries,
        gap_cancellations=gaps,
        active_at_end=active_index is not None,
    )


def _split(candidate: Candidate) -> str | None:
    for name, start, end in SPLITS:
        if (
            candidate.setup_time >= start
            and candidate.confirmation_time >= start
            and candidate.availability_time >= start
            and candidate.entry_time >= start
            and candidate.exit_time <= end
        ):
            return name
    return None


def schedule_candidates(
    candidates: Iterable[Candidate],
    *,
    entry_delay_bars: int = ENTRY_DELAY_BARS,
) -> tuple[list[Candidate], ScheduleAudit]:
    ordered = sorted(
        candidates,
        key=lambda row: (row.entry_time, row.setup_time, row.confirmation_time, row.side),
    )
    for row in ordered:
        if row.side not in {-1, 1}:
            raise ValueError("LVRT candidate side changed")
        if row.availability_time - row.confirmation_time != FINALITY_BARS * BAR:
            raise ValueError("LVRT finality delay changed")
        if row.entry_time - row.confirmation_time != entry_delay_bars * BAR:
            raise ValueError("LVRT entry delay changed")
        if row.exit_time - row.entry_time != HOLD_BARS * BAR:
            raise ValueError("LVRT hold changed")
    contained = [replace(row, split=split) for row in ordered if (split := _split(row))]
    accepted: list[Candidate] = []
    prior_exit = datetime.min.replace(tzinfo=timezone.utc)
    overlaps = 0
    for row in contained:
        if row.entry_time < prior_exit:
            overlaps += 1
            continue
        accepted.append(row)
        prior_exit = row.exit_time
    return accepted, ScheduleAudit(
        raw_candidates=len(ordered),
        split_contained_candidates=len(contained),
        split_boundary_drops=len(ordered) - len(contained),
        overlap_suppressions=overlaps,
        accepted_candidates=len(accepted),
    )


def _direct_candidates(
    dates: Sequence[datetime],
    mask: np.ndarray,
    sides: np.ndarray,
) -> list[Candidate]:
    return [
        _new_candidate(dates[index], dates[index], int(sides[index]))
        for index in np.flatnonzero(mask)
        if int(sides[index]) in {-1, 1}
    ]


def _deterministic_side(entry: datetime) -> int:
    digest = hashlib.sha256(
        f"LVRT-72-random-side-20260721|{entry.isoformat()}".encode("ascii")
    ).digest()
    return 1 if digest[0] < 128 else -1


def _clock_rows(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "lvrt:primary",
            "split": row.split,
            "causal_origin": row.setup_time.isoformat(),
            "decision_time": row.confirmation_time.isoformat(),
            "availability_time": row.availability_time.isoformat(),
            "entry_time": row.entry_time.isoformat(),
            "exit_time": row.exit_time.isoformat(),
            "side": row.side,
        }
        for row in candidates
    ]


def _summary(candidates: Sequence[Candidate], split: str) -> dict[str, Any]:
    rows = [row for row in candidates if row.split == split]
    total = len(rows)
    months = Counter(f"{row.entry_time.year:04d}-{row.entry_time.month:02d}" for row in rows)
    weekdays = Counter(str(row.entry_time.weekday()) for row in rows)
    return {
        "accepted_events": total,
        "side_counts": {
            "LONG": sum(row.side == 1 for row in rows),
            "SHORT": sum(row.side == -1 for row in rows),
        },
        "year_counts": dict(sorted(Counter(str(row.entry_time.year) for row in rows).items())),
        "half_counts": dict(
            sorted(
                Counter(
                    f"{row.entry_time.year}-H{1 if row.entry_time.month <= 6 else 2}"
                    for row in rows
                ).items()
            )
        ),
        "quarter_counts": dict(
            sorted(
                Counter(
                    f"{row.entry_time.year}-Q{(row.entry_time.month - 1) // 3 + 1}"
                    for row in rows
                ).items()
            )
        ),
        "active_utc_weeks": len({row.entry_time.isocalendar()[:2] for row in rows}),
        "maximum_calendar_month_share": max(months.values()) / total if total else 0.0,
        "maximum_utc_entry_weekday_share": (
            max(weekdays.values()) / total if total else 0.0
        ),
    }


def _support_gate(primary: Mapping[str, Any]) -> dict[str, Any]:
    train = primary["splits"]["train"]
    selection = primary["splits"]["selection"]
    checks = {
        "train_total_between_100_and_360": 100 <= train["accepted_events"] <= 360,
        "train_each_year_between_25_and_160": all(
            25 <= train["year_counts"].get(str(year), 0) <= 160
            for year in (2020, 2021, 2022)
        ),
        "train_each_half_at_least_12": all(
            train["half_counts"].get(f"{year}-H{half}", 0) >= 12
            for year in (2020, 2021, 2022)
            for half in (1, 2)
        ),
        "train_long_at_least_35": train["side_counts"]["LONG"] >= 35,
        "train_short_at_least_35": train["side_counts"]["SHORT"] >= 35,
        "train_active_weeks_at_least_60": train["active_utc_weeks"] >= 60,
        "train_month_share_at_most_0_15": train["maximum_calendar_month_share"] <= 0.15,
        "train_weekday_share_at_most_0_22": (
            train["maximum_utc_entry_weekday_share"] <= 0.22
        ),
        "selection_total_between_45_and_180": (
            45 <= selection["accepted_events"] <= 180
        ),
        "selection_each_half_at_least_18": all(
            selection["half_counts"].get(f"2023-H{half}", 0) >= 18
            for half in (1, 2)
        ),
        "selection_each_quarter_at_least_8": all(
            selection["quarter_counts"].get(f"2023-Q{quarter}", 0) >= 8
            for quarter in range(1, 5)
        ),
        "selection_long_at_least_15": selection["side_counts"]["LONG"] >= 15,
        "selection_short_at_least_15": selection["side_counts"]["SHORT"] >= 15,
        "selection_active_weeks_at_least_25": selection["active_utc_weeks"] >= 25,
        "selection_month_share_at_most_0_20": (
            selection["maximum_calendar_month_share"] <= 0.20
        ),
        "selection_weekday_share_at_most_0_25": (
            selection["maximum_utc_entry_weekday_share"] <= 0.25
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _control_report(
    accepted: Sequence[Candidate],
    build_audit: BuildAudit | None,
    schedule_audit: ScheduleAudit | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "clock_hash": canonical_hash(_clock_rows(accepted)),
        "splits": {name: _summary(accepted, name) for name, _, _ in SPLITS},
    }
    if build_audit is not None:
        report["build_audit"] = asdict(build_audit)
    if schedule_audit is not None:
        report["schedule_audit"] = asdict(schedule_audit)
    return report


def _load_comparators() -> dict[str, list[ClockRow]]:
    freeze = load_json(COMPARATOR_FREEZE)
    if freeze.get("manifest_hash") != COMPARATOR_FREEZE_MANIFEST_HASH:
        raise ValueError("LVRT comparator freeze manifest changed")
    if freeze.get("all_required_members_available") is not True:
        raise ValueError("LVRT comparator cohort is incomplete")
    mfic_schema = [
        "candidate_id",
        "split",
        "causal_origin",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    live_schema = [
        "candidate_id",
        "split",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    tuples = [
        *_load_afcs(),
        *_load_bafr(),
        *_load_pure_clock(MFIC_CLOCK, expected_schema=mfic_schema),
        *_load_pure_clock(LIVE_CLOCK, expected_schema=live_schema),
    ]
    grouped: dict[str, list[ClockRow]] = {}
    for candidate_id, entry, exit_time, side in tuples:
        grouped.setdefault(candidate_id, []).append(
            ClockRow(
                candidate_id=candidate_id,
                split="comparator",
                causal_origin=entry,
                decision_time=entry,
                entry_time=entry,
                exit_time=exit_time,
                side=side,
            )
        )
    if set(grouped) != set(freeze["required_members"]):
        raise ValueError("LVRT comparator member set changed")
    return grouped


def _novelty(
    primary: Sequence[Candidate],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_clocks = [
        ClockRow(
            candidate_id="lvrt:primary",
            split=cast(str, row.split),
            causal_origin=row.setup_time,
            decision_time=row.confirmation_time,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
            side=row.side,
        )
        for row in primary
    ]
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for candidate_id, comparator in sorted(_load_comparators().items()):
        eligible = [row for row in comparator if row.entry_time < SELECTION_END]
        start = max(TRAIN_START, min(row.entry_time for row in eligible))
        end = min(SELECTION_END, max(row.exit_time for row in eligible))
        left = [
            row
            for row in primary_clocks
            if start <= row.entry_time < end and row.exit_time <= end
        ]
        right = [
            row
            for row in eligible
            if start <= row.entry_time < end and row.exit_time <= end
        ]
        if not left or not right:
            raise ValueError(f"empty LVRT novelty common coverage: {candidate_id}")
        jaccard, exact_matches = exact_entry_jaccard(left, right)
        tolerant_matches = maximum_tolerant_matches(left, right)
        correlation, position_jaccard = exposure_metrics(
            left,
            right,
            start=start,
            end=end,
        )
        item = {
            "common_start": start.isoformat(),
            "common_end_exclusive": end.isoformat(),
            "lvrt_events": len(left),
            "comparator_events": len(right),
            "exact_entry_matches": exact_matches,
            "exact_entry_jaccard": jaccard,
            "maximum_one_to_one_matches_within_6h": tolerant_matches,
            "lvrt_tolerant_match_coverage": tolerant_matches / len(left),
            "signed_occupied_exposure_correlation": correlation,
            "position_bar_jaccard": position_jaccard,
        }
        metrics[candidate_id] = item
        prefix = candidate_id.replace(":", "_")
        checks[f"{prefix}_exact_jaccard_at_most_0_20"] = jaccard <= 0.20
        checks[f"{prefix}_tolerant_coverage_at_most_0_35"] = (
            item["lvrt_tolerant_match_coverage"] <= 0.35
        )
        checks[f"{prefix}_absolute_exposure_correlation_at_most_0_40"] = (
            abs(correlation) <= 0.40
        )
    return metrics, {"checks": checks, "passed": all(checks.values())}


def _clock_bytes(rows: Sequence[Candidate]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(CLOCK_FIELDS)
    for row in rows:
        writer.writerow(
            [
                "lvrt:primary",
                row.split,
                row.setup_time.isoformat(),
                row.confirmation_time.isoformat(),
                row.availability_time.isoformat(),
                row.entry_time.isoformat(),
                row.exit_time.isoformat(),
                row.side,
            ]
        )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return output.getvalue()


def build_outputs() -> tuple[dict[str, Any], bytes | None]:
    frame, source_audit = load_source_frame()
    dates = [value.to_pydatetime() for value in cast(pd.DatetimeIndex, frame.index)]
    valid = frame["valid"].to_numpy(bool)
    setup = _setup_mask(frame)
    replenishment = _replenishment_mask(frame)
    flow_sign = frame["flow_sign"].to_numpy(np.int8)

    raw_primary, primary_build = build_relay_candidates(
        dates,
        valid,
        setup,
        replenishment,
        flow_sign,
        require_flip=True,
    )
    primary, primary_schedule = schedule_candidates(raw_primary)
    primary_report = _control_report(primary, primary_build, primary_schedule)
    support_gate = _support_gate(primary_report)

    vacuum_raw = _direct_candidates(dates, setup, -flow_sign)
    vacuum, vacuum_schedule = schedule_candidates(vacuum_raw)
    replenishment_raw = _direct_candidates(dates, replenishment, flow_sign)
    replenishment_only, replenishment_schedule = schedule_candidates(
        replenishment_raw
    )
    no_flip_raw, no_flip_build = build_relay_candidates(
        dates,
        valid,
        setup,
        replenishment,
        flow_sign,
        require_flip=False,
    )
    no_flip, no_flip_schedule = schedule_candidates(no_flip_raw)
    reverse_raw, reverse_build = build_reverse_candidates(
        dates,
        valid,
        setup,
        replenishment,
        flow_sign,
    )
    reverse, reverse_schedule = schedule_candidates(reverse_raw)
    direction_flip = [replace(row, side=-row.side) for row in primary]
    random_side = [replace(row, side=_deterministic_side(row.entry_time)) for row in primary]
    delayed_raw = [
        replace(
            row,
            entry_time=row.entry_time + BAR,
            exit_time=row.exit_time + BAR,
            split=None,
        )
        for row in raw_primary
    ]
    delayed, delayed_schedule = schedule_candidates(
        delayed_raw,
        entry_delay_bars=ENTRY_DELAY_BARS + 1,
    )
    controls = {
        "vacuum_only": _control_report(vacuum, None, vacuum_schedule),
        "replenishment_only": _control_report(
            replenishment_only, None, replenishment_schedule
        ),
        "no_flow_flip": _control_report(no_flip, no_flip_build, no_flip_schedule),
        "reverse_order": _control_report(reverse, reverse_build, reverse_schedule),
        "direction_flip": _control_report(direction_flip, None, None),
        "deterministic_random_side": _control_report(random_side, None, None),
        "one_bar_execution_delay": _control_report(delayed, None, delayed_schedule),
    }

    if support_gate["passed"]:
        novelty_metrics, novelty_gate = _novelty(primary)
        clock_bytes: bytes | None = _clock_bytes(primary)
    else:
        novelty_metrics = {}
        novelty_gate = {
            "checks": {},
            "passed": False,
            "skipped_reason": "source support failed",
        }
        clock_bytes = None
    passed = bool(support_gate["passed"] and novelty_gate["passed"])
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-21",
        "implementation_binding": {
            "path": str(IMPLEMENTATION),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "decision_binding": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "source_binding": {
            "path": str(SOURCE),
            "sha256": SOURCE_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "comparator_freeze_binding": {
            "path": str(COMPARATOR_FREEZE),
            "sha256": COMPARATOR_FREEZE_SHA256,
            "manifest_hash": COMPARATOR_FREEZE_MANIFEST_HASH,
        },
        "configuration": {
            "reference_bars": REFERENCE_BARS,
            "episode_bars": EPISODE_BARS,
            "hold_bars": HOLD_BARS,
            "finality_bars": FINALITY_BARS,
            "entry_delay_bars": ENTRY_DELAY_BARS,
            "exposure": 0.5,
            "train_start": TRAIN_START.isoformat(),
            "train_end_exclusive": TRAIN_END.isoformat(),
            "selection_end_exclusive": SELECTION_END.isoformat(),
        },
        "source_audit": source_audit,
        "primary": primary_report,
        "controls": controls,
        "support_gate": support_gate,
        "novelty_metrics": novelty_metrics,
        "novelty_gate": novelty_gate,
        "combined_gate_passed": passed,
        "pure_clock": (
            {
                "path": str(DEFAULT_CLOCK_OUTPUT),
                "sha256": hashlib.sha256(clock_bytes).hexdigest(),
                "gzip_mtime": 0,
                "schema": list(CLOCK_FIELDS),
                "rows": len(primary),
            }
            if clock_bytes is not None
            else None
        ),
        "outcome_boundary": {
            "lvrt_source_rows_read": source_audit["source_rows"],
            "source_columns_read": list(ALLOWLIST),
            "forbidden_source_columns_read": 0,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "strict_simulation_calls": 0,
            "post_2023_rows_loaded": 0,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
        "failure_action": None if passed else "retire_before_economic_evaluation",
        "next_action": (
            "freeze strict train evaluator"
            if passed
            else "retire LVRT-72 without threshold, side, or hold repair"
        ),
    }
    core["result_hash"] = canonical_hash(core)
    return {**core, "created_at": datetime.now(timezone.utc).isoformat()}, clock_bytes


def publish(
    report_path: Path,
    clock_path: Path,
    report: Mapping[str, Any],
    clock_bytes: bytes | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        if clock_bytes is not None:
            if report["pure_clock"]["path"] != str(clock_path):
                raise ValueError("LVRT pure-clock output path changed")
            clock_fd = os.open(clock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            created.append(clock_path)
            with os.fdopen(clock_fd, "wb") as handle:
                handle.write(clock_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        report_fd = os.open(
            report_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        created.append(report_path)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(report_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report, clock_bytes = build_outputs()
    publish(DEFAULT_REPORT_OUTPUT, DEFAULT_CLOCK_OUTPUT, report, clock_bytes)
    failed_support = [
        name for name, passed in report["support_gate"]["checks"].items() if not passed
    ]
    failed_novelty = [
        name for name, passed in report["novelty_gate"]["checks"].items() if not passed
    ]
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT_OUTPUT),
                "clock": str(DEFAULT_CLOCK_OUTPUT) if clock_bytes is not None else None,
                "result_hash": report["result_hash"],
                "passed": report["combined_gate_passed"],
                "failed_support": failed_support,
                "failed_novelty": failed_novelty,
                "train_events": report["primary"]["splits"]["train"][
                    "accepted_events"
                ],
                "selection_events": report["primary"]["splits"]["selection"][
                    "accepted_events"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
