"""Build FLNSR-2016 source-only clocks, support, and novelty diagnostics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import federal_liquidity_component_concordance_clock as flcc
from training import preregister_federal_liquidity_narrative_sponsorship_relay as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "252952438eb2a87dc5f85fbe887a4f99a5f3a7a8a7e764feac414fac2929fd6d"
)
DEFAULT_OUTPUT = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_"
    "support_2026-07-23.json"
)
DEFAULT_CLOCK = Path(
    "data/federal_liquidity_narrative_sponsorship_relay_"
    "clocks_2020_2023.csv.gz"
)
FLCC_COMPARATOR = Path(
    "results/federal_liquidity_component_concordance_"
    "preregistered_clock_2026-07-17.csv.gz"
)
TRAIN_START = pd.Timestamp("2020-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = TRAIN_END
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
BAR = pd.Timedelta(minutes=5)
HOLD = 2_016 * BAR
DAY = pd.Timedelta(days=1)
CLOCK_COLUMNS = [
    "clock_name",
    "release_date",
    "narrative_source_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_inputs() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("FLNSR preregistration file changed")
    frozen = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(frozen)
    if frozen["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("FLNSR policy changed after preregistration")
    inputs = (
        (prereg.H41_SOURCE, frozen["h41_source_contract"]["source_sha256"]),
        (
            prereg.H41_BUILD_MANIFEST,
            frozen["h41_source_contract"]["build_manifest_sha256"],
        ),
        (
            prereg.H41_SOURCE_MANIFEST,
            frozen["h41_source_contract"]["source_manifest_sha256"],
        ),
        (prereg.GDELT_SOURCE, frozen["gdelt_source_contract"]["source_sha256"]),
        (
            prereg.GDELT_MANIFEST,
            frozen["gdelt_source_contract"]["manifest_file_sha256"],
        ),
        (
            prereg.GDELT_ACCESS_SEAL,
            frozen["gdelt_source_contract"]["access_seal_sha256"],
        ),
        (
            FLCC_COMPARATOR,
            frozen["source_only_novelty_gate"]["comparator_sha256"],
        ),
    )
    for path, expected in inputs:
        if sha256_file(path) != expected:
            raise RuntimeError(f"FLNSR frozen input changed: {path}")
    manifest = json.loads(Path(prereg.GDELT_MANIFEST).read_text())
    if manifest.get("manifest_hash") != frozen["gdelt_source_contract"]["manifest_hash"]:
        raise RuntimeError("FLNSR GDELT internal manifest hash changed")
    return frozen


def load_h41(path: str | Path = prereg.H41_SOURCE) -> pd.DataFrame:
    rows = flcc.read_source(path)
    if len(rows) != 313:
        raise RuntimeError(f"FLNSR H.4.1 row count changed: {len(rows)}")
    frame = pd.DataFrame(asdict(row) for row in rows)
    frame["release_date"] = pd.to_datetime(frame["release_date"], utc=True)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], utc=True)
    frame["available_at_utc"] = pd.to_datetime(frame["available_at_utc"], utc=True)
    if not frame["release_date"].is_monotonic_increasing:
        raise RuntimeError("FLNSR H.4.1 release order changed")
    return frame


def load_gdelt(path: str | Path = prereg.GDELT_SOURCE) -> pd.DataFrame:
    required = prereg._core_manifest()["gdelt_source_contract"]["required_columns"]
    frame = pd.read_csv(path, usecols=required).loc[:, required]
    if len(frame) != 1_461:
        raise RuntimeError(f"FLNSR GDELT row count changed: {len(frame)}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    expected = pd.date_range("2020-01-01", "2024-01-01", freq="D", inclusive="left", tz="UTC")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise RuntimeError("FLNSR GDELT daily grid changed")
    if not frame["available_at"].equals(frame["date"] + pd.Timedelta(hours=48, minutes=15)):
        raise RuntimeError("FLNSR GDELT availability clock changed")
    count_columns = [column for column in required if column.endswith("_article_count")]
    for column in count_columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
            raise RuntimeError(f"FLNSR invalid GDELT count: {column}")
        frame[column] = numeric.astype(np.int64)
    for column in ("failure_article_count", "constraint_article_count", "adoption_article_count"):
        if (frame[column] > frame["broad_article_count"]).any():
            raise RuntimeError(f"FLNSR GDELT subset count exceeds broad: {column}")
    all_zero = frame.loc[frame[count_columns].eq(0).all(axis=1), "date"].dt.strftime("%Y-%m-%d").tolist()
    if all_zero != ["2020-10-20", "2023-03-23"]:
        raise RuntimeError(f"FLNSR GDELT outage set changed: {all_zero}")
    return frame


def midrank_numerator(current: int, prior: Iterable[int], expected: int = 104) -> int:
    values = list(prior)
    if len(values) != expected:
        raise RuntimeError(f"FLNSR midrank expected {expected} values, got {len(values)}")
    return 2 * sum(value < current for value in values) + sum(
        value == current for value in values
    )


def _side_from_rank(numerator: int, policy: prereg.Policy) -> str:
    if numerator >= policy.liquidity_upper_rank_numerator:
        return "LONG"
    if numerator <= policy.liquidity_lower_rank_numerator:
        return "SHORT"
    return ""


def narrative_quality(frame: pd.DataFrame, pseudocount: float = 0.5) -> pd.Series:
    adoption = frame["adoption_article_count"].to_numpy(float)
    stress = (
        frame["failure_article_count"].to_numpy(float)
        + frame["constraint_article_count"].to_numpy(float)
    )
    return pd.Series(
        np.log((adoption + pseudocount) / (stress + 2.0 * pseudocount)),
        index=frame.index,
        dtype=float,
    )


def build_release_features(
    h41: pd.DataFrame,
    gdelt: pd.DataFrame,
    policy: prereg.Policy = prereg.Policy(),
) -> pd.DataFrame:
    impulses = h41["net_liquidity_usd_millions"].diff()
    quality = narrative_quality(gdelt, policy.narrative_pseudocount)
    narrative_available = pd.DatetimeIndex(gdelt["available_at"])
    records: list[dict[str, Any]] = []
    first_index = policy.liquidity_delta_releases + policy.liquidity_rank_lookback_releases
    for index in range(first_index, len(h41)):
        current_impulse = int(impulses.iloc[index])
        prior = impulses.iloc[index - policy.liquidity_rank_lookback_releases : index]
        if prior.isna().any():
            raise RuntimeError("FLNSR H.4.1 impulse warmup failed")
        rank_numerator = midrank_numerator(
            current_impulse,
            (int(value) for value in prior),
            policy.liquidity_rank_lookback_releases,
        )
        liquidity_side = _side_from_rank(rank_numerator, policy)
        signal_time = pd.Timestamp(h41.at[index, "available_at_utc"])
        narrative_index = int(narrative_available.searchsorted(signal_time, side="right") - 1)
        required_days = policy.narrative_recent_days + policy.narrative_baseline_days
        if narrative_index + 1 < required_days:
            continue
        start = narrative_index + 1 - required_days
        window_dates = gdelt["date"].iloc[start : narrative_index + 1]
        expected_dates = pd.date_range(
            window_dates.iloc[0], periods=required_days, freq="D", tz="UTC"
        )
        if not pd.DatetimeIndex(window_dates).equals(expected_dates):
            raise RuntimeError("FLNSR GDELT narrative window is not consecutive")
        baseline_end = narrative_index + 1 - policy.narrative_recent_days
        baseline = float(quality.iloc[start:baseline_end].mean())
        recent = float(quality.iloc[baseline_end : narrative_index + 1].mean())
        rotation = recent - baseline
        narrative_side = "LONG" if rotation > 0 else "SHORT" if rotation < 0 else ""
        entry_time = signal_time + pd.Timedelta(minutes=policy.entry_delay_minutes)
        records.append(
            {
                "release_date": pd.Timestamp(h41.at[index, "release_date"]),
                "narrative_source_date": pd.Timestamp(gdelt.at[narrative_index, "date"]),
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": entry_time + policy.hold_bars * BAR,
                "liquidity_impulse": current_impulse,
                "liquidity_rank_numerator": rank_numerator,
                "liquidity_side": liquidity_side,
                "narrative_rotation": rotation,
                "narrative_side": narrative_side,
                "primary": bool(liquidity_side and liquidity_side == narrative_side),
            }
        )
    features = pd.DataFrame(records)
    if features.empty:
        return features.assign(previous_narrative_side=pd.Series(dtype=str))
    features["previous_narrative_side"] = features["narrative_side"].shift(1).fillna("")
    return features


def _contained(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        frame["signal_time"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].le(end)
    ].copy()


def _schedule(frame: pd.DataFrame, clock_name: str, side_column: str) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in frame.sort_values("entry_time", kind="mergesort").itertuples(index=False):
        side = str(getattr(row, side_column))
        if side not in {"LONG", "SHORT"}:
            continue
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.append(
            {
                "clock_name": clock_name,
                "release_date": pd.Timestamp(row.release_date),
                "narrative_source_date": pd.Timestamp(row.narrative_source_date),
                "signal_time": pd.Timestamp(row.signal_time),
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
            }
        )
        previous_exit = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def schedule_across_splits(
    frame: pd.DataFrame, clock_name: str, side_column: str
) -> pd.DataFrame:
    parts = [
        _schedule(_contained(frame, TRAIN_START, TRAIN_END), clock_name, side_column),
        _schedule(
            _contained(frame, SELECTION_START, SELECTION_END), clock_name, side_column
        ),
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    ).reset_index(drop=True)


def _exact_variant(primary: pd.DataFrame, clock_name: str, sides: Iterable[str]) -> pd.DataFrame:
    output = primary.copy()
    output["clock_name"] = clock_name
    output["side"] = list(sides)
    return output.loc[:, CLOCK_COLUMNS]


def build_clocks(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary_rows = features.loc[features["primary"]].copy()
    primary = schedule_across_splits(primary_rows, "primary", "liquidity_side")
    disagreement = features.loc[
        features["liquidity_side"].ne("")
        & features["narrative_side"].ne("")
        & features["liquidity_side"].ne(features["narrative_side"])
    ]
    stale = features.loc[
        features["liquidity_side"].ne("")
        & features["previous_narrative_side"].ne("")
        & features["liquidity_side"].eq(features["previous_narrative_side"])
    ].copy()
    clocks = {
        "primary": primary,
        "liquidity_only": schedule_across_splits(
            features.loc[features["liquidity_side"].ne("")],
            "liquidity_only",
            "liquidity_side",
        ),
        "narrative_only": schedule_across_splits(
            features.loc[features["narrative_side"].ne("")],
            "narrative_only",
            "narrative_side",
        ),
        "disagreement": schedule_across_splits(
            disagreement, "disagreement", "liquidity_side"
        ),
        "one_release_stale_narrative": schedule_across_splits(
            stale, "one_release_stale_narrative", "liquidity_side"
        ),
    }
    clocks["exact_side_flip"] = _exact_variant(
        primary,
        "exact_side_flip",
        ["SHORT" if side == "LONG" else "LONG" for side in primary["side"]],
    )
    seed = int.from_bytes(
        hashlib.sha256(b"FLNSR-2016|side").digest()[:8], "big", signed=False
    )
    rng = np.random.default_rng(seed)
    random_sides = np.where(rng.integers(0, 2, len(primary)) == 0, "SHORT", "LONG")
    clocks["deterministic_random_side"] = _exact_variant(
        primary, "deterministic_random_side", random_sides
    )
    return clocks


def _window(clock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return clock.loc[clock["entry_time"].ge(start) & clock["exit_time"].le(end)]


def _longest_run(sides: Iterable[str]) -> int:
    best = current = 0
    previous = None
    for side in sides:
        if side == previous:
            current += 1
        else:
            previous = side
            current = 1
        best = max(best, current)
    return best


def clock_stats(clock: pd.DataFrame) -> dict[str, Any]:
    ordered = clock.sort_values("entry_time", kind="mergesort")
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
            "maximum_calendar_gap_days": None,
            "maximum_same_side_run": 0,
        }
    sides = ordered["side"].value_counts().to_dict()
    naive = ordered["entry_time"].dt.tz_convert(None)
    months = naive.dt.to_period("M").astype(str).value_counts()
    quarters = naive.dt.to_period("Q").astype(str).value_counts()
    gaps = ordered["entry_time"].diff().dropna()
    return {
        "events": total,
        "long": int(sides.get("LONG", 0)),
        "short": int(sides.get("SHORT", 0)),
        "long_share": float(sides.get("LONG", 0) / total),
        "short_share": float(sides.get("SHORT", 0) / total),
        "active_months": int(len(months)),
        "maximum_month_share": float(months.max() / total),
        "maximum_quarter_share": float(quarters.max() / total),
        "maximum_calendar_gap_days": float(gaps.max() / DAY) if len(gaps) else None,
        "maximum_same_side_run": _longest_run(ordered["side"]),
    }


def _normalize_side(value: Any) -> str:
    if str(value) in {"1", "1.0", "LONG"}:
        return "LONG"
    if str(value) in {"-1", "-1.0", "SHORT"}:
        return "SHORT"
    raise RuntimeError(f"FLNSR comparator has invalid side: {value!r}")


def one_to_one_overlap(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=15),
) -> dict[str, Any]:
    left = primary.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    right = comparator.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for left_index, left_row in left.iterrows():
        candidates: list[tuple[pd.Timedelta, pd.Timestamp, int]] = []
        for right_index, right_row in right.iterrows():
            if right_index in used:
                continue
            delta = abs(pd.Timestamp(right_row["entry_time"]) - pd.Timestamp(left_row["entry_time"]))
            if delta <= tolerance:
                candidates.append((delta, pd.Timestamp(right_row["entry_time"]), right_index))
        if not candidates:
            continue
        _, _, selected = min(candidates)
        used.add(selected)
        matches.append((left_index, selected))
    matched = len(matches)
    same_side = sum(
        str(left.at[i, "side"]) == _normalize_side(right.at[j, "side"])
        for i, j in matches
    )
    union = len(left) + len(right) - matched
    return {
        "primary_events": len(left),
        "comparator_events": len(right),
        "matched": matched,
        "same_side_matched": same_side,
        "jaccard": float(matched / union) if union else 0.0,
        "flnsr_containment": float(matched / len(left)) if len(left) else 0.0,
        "same_side_flnsr_containment": float(same_side / len(left)) if len(left) else 0.0,
    }


def novelty_report(primary: pd.DataFrame) -> dict[str, Any]:
    comparator = pd.read_csv(FLCC_COMPARATOR)
    required = {"candidate_id", "clock_name", "entry_time", "side"}
    if not required.issubset(comparator.columns):
        raise RuntimeError("FLNSR FLCC comparator schema changed")
    comparator = comparator.loc[comparator["clock_name"].eq("primary")].copy()
    comparator["entry_time"] = pd.to_datetime(comparator["entry_time"], utc=True)
    frozen = prereg._core_manifest()["source_only_novelty_gate"]
    candidates: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for candidate_id, group in comparator.groupby("candidate_id", sort=True):
        stats = one_to_one_overlap(primary, group)
        candidates[str(candidate_id)] = stats
        checks[f"{candidate_id}:nonempty"] = stats["comparator_events"] > 0
        checks[f"{candidate_id}:jaccard"] = stats["jaccard"] <= frozen[
            "jaccard_max_each_flcc_candidate"
        ]
        checks[f"{candidate_id}:containment"] = stats["flnsr_containment"] <= frozen[
            "flnsr_containment_max_each_flcc_candidate"
        ]
        checks[f"{candidate_id}:same_side_containment"] = stats[
            "same_side_flnsr_containment"
        ] <= frozen["same_side_flnsr_containment_max_each_flcc_candidate"]
    if len(candidates) != 4:
        checks["exactly_four_flcc_candidates"] = False
    else:
        checks["exactly_four_flcc_candidates"] = True
    return {
        "comparator_path": str(FLCC_COMPARATOR),
        "comparator_sha256": sha256_file(FLCC_COMPARATOR),
        "tolerance_minutes": 15,
        "candidate_metrics": candidates,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _combine_clocks(clocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nonempty = [clock for clock in clocks.values() if not clock.empty]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    ).reset_index(drop=True)


def write_clock(path: str | Path, clocks: dict[str, pd.DataFrame]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = _combine_clocks(clocks)
    if list(combined.columns) != CLOCK_COLUMNS:
        raise RuntimeError("FLNSR clock schema changed")
    if not combined["entry_time"].eq(combined["signal_time"] + pd.Timedelta(minutes=10)).all():
        raise RuntimeError("FLNSR entry-delay contract changed")
    if not combined["exit_time"].eq(combined["entry_time"] + HOLD).all():
        raise RuntimeError("FLNSR hold contract changed")
    csv_bytes = combined.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ").encode()
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            zipped.write(csv_bytes)


def _support_report(
    h41: pd.DataFrame,
    gdelt: pd.DataFrame,
    features: pd.DataFrame,
    clocks: dict[str, pd.DataFrame],
    clock_path: Path,
) -> dict[str, Any]:
    primary = clocks["primary"]
    windows = {
        "all": clock_stats(primary),
        "train_2020_2022": clock_stats(_window(primary, TRAIN_START, TRAIN_END)),
        "2020": clock_stats(_window(primary, TRAIN_START, pd.Timestamp("2021-01-01T00:00:00Z"))),
        "2021": clock_stats(_window(primary, pd.Timestamp("2021-01-01T00:00:00Z"), pd.Timestamp("2022-01-01T00:00:00Z"))),
        "2022": clock_stats(_window(primary, pd.Timestamp("2022-01-01T00:00:00Z"), TRAIN_END)),
        "selection_2023": clock_stats(_window(primary, SELECTION_START, SELECTION_END)),
        "selection_2023_h1": clock_stats(_window(primary, SELECTION_START, pd.Timestamp("2023-07-01T00:00:00Z"))),
        "selection_2023_h2": clock_stats(_window(primary, pd.Timestamp("2023-07-01T00:00:00Z"), SELECTION_END)),
    }
    gate = prereg._core_manifest()["source_support_gate"]
    low, high = gate["each_side_share_range_all_train_selection"]

    def side_ok(stats: dict[str, Any]) -> bool:
        return bool(
            stats["events"]
            and stats["long_share"] is not None
            and low <= stats["long_share"] <= high
            and low <= stats["short_share"] <= high
        )

    all_stats = windows["all"]
    support_checks = {
        "h41_rows_exact": len(h41) == 313,
        "gdelt_rows_exact": len(gdelt) == 1_461,
        "train_events_min": windows["train_2020_2022"]["events"] >= gate["train_events_min"],
        "each_train_year_events_min": all(
            windows[str(year)]["events"] >= gate["each_train_year_events_min"]
            for year in (2020, 2021, 2022)
        ),
        "selection_events_min": windows["selection_2023"]["events"] >= gate["selection_events_min"],
        "each_selection_half_events_min": all(
            windows[name]["events"] >= gate["each_selection_half_events_min"]
            for name in ("selection_2023_h1", "selection_2023_h2")
        ),
        "side_share_all_train_selection": all(
            side_ok(windows[name]) for name in ("all", "train_2020_2022", "selection_2023")
        ),
        "active_months_min": all_stats["active_months"] >= gate["active_months_min"],
        "maximum_single_month_share": bool(
            all_stats["maximum_month_share"] is not None
            and all_stats["maximum_month_share"] <= gate["maximum_single_month_share"]
        ),
        "maximum_single_quarter_share": bool(
            all_stats["maximum_quarter_share"] is not None
            and all_stats["maximum_quarter_share"] <= gate["maximum_single_quarter_share"]
        ),
        "maximum_calendar_gap_days": bool(
            all_stats["maximum_calendar_gap_days"] is not None
            and all_stats["maximum_calendar_gap_days"] <= gate["maximum_calendar_gap_days"]
        ),
        "maximum_same_side_run": all_stats["maximum_same_side_run"] <= gate["maximum_same_side_run"],
        "clock_schema_outcome_free": list(_combine_clocks(clocks).columns) == CLOCK_COLUMNS,
    }
    novelty = novelty_report(primary)
    failed_support = [name for name, passed in support_checks.items() if not passed]
    failed_novelty = [name for name, passed in novelty["checks"].items() if not passed]
    combined = _combine_clocks(clocks)
    passed = not failed_support and not failed_novelty
    return {
        "protocol_version": "federal_liquidity_narrative_sponsorship_support_v1",
        "policy_id": prereg.Policy().policy_id,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "support_passed": passed,
        "authorized_next_stage": "freeze_strict_evaluator" if passed else None,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(PREREGISTRATION),
            "manifest_hash": json.loads(PREREGISTRATION.read_text())["manifest_hash"],
        },
        "sources": {
            "h41": {"path": prereg.H41_SOURCE, "sha256": sha256_file(prereg.H41_SOURCE), "rows": len(h41)},
            "gdelt": {"path": prereg.GDELT_SOURCE, "sha256": sha256_file(prereg.GDELT_SOURCE), "rows": len(gdelt)},
        },
        "feature_funnel": {
            "release_features": len(features),
            "liquidity_non_neutral": int(features["liquidity_side"].ne("").sum()),
            "narrative_long": int(features["narrative_side"].eq("LONG").sum()),
            "narrative_short": int(features["narrative_side"].eq("SHORT").sum()),
            "raw_agreement": int(features["primary"].sum()),
            "raw_disagreement": int(
                (
                    features["liquidity_side"].ne("")
                    & features["narrative_side"].ne("")
                    & features["liquidity_side"].ne(features["narrative_side"])
                ).sum()
            ),
        },
        "windows": windows,
        "controls": {name: clock_stats(clock) for name, clock in clocks.items()},
        "support_checks": support_checks,
        "novelty": novelty,
        "failed_support_checks": failed_support,
        "failed_novelty_checks": failed_novelty,
        "clock": {
            "path": str(clock_path),
            "rows": len(combined),
            "columns": CLOCK_COLUMNS,
            "sha256": sha256_file(clock_path),
        },
        "leakage_guard": {
            "h41_reference_exactly_104_prior_impulses": True,
            "current_h41_impulse_excluded_from_rank": True,
            "gdelt_latest_row_available_not_after_h41_release": True,
            "gdelt_window_exactly_28_consecutive_source_days": True,
            "no_btc_market_or_funding_source_loaded": True,
            "no_post_entry_price_access": True,
            "no_return_or_pnl_field_written": True,
            "comparator_timestamps_and_sides_only": True,
            "llm_or_model_trained": False,
        },
    }


def build_support(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK,
) -> dict[str, Any]:
    validate_frozen_inputs()
    h41 = load_h41()
    gdelt = load_gdelt()
    features = build_release_features(h41, gdelt)
    clocks = build_clocks(features)
    clock_path = Path(clock_output)
    write_clock(clock_path, clocks)
    report = _support_report(h41, gdelt, features, clocks, clock_path)
    core = {key: value for key, value in report.items() if key != "created_at"}
    report["report_manifest_hash"] = canonical_hash(core)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    args = parser.parse_args()
    report = build_support(output=args.output, clock_output=args.clock_output)
    print(
        json.dumps(
            {
                "policy_id": report["policy_id"],
                "support_passed": report["support_passed"],
                "failed_support_checks": report["failed_support_checks"],
                "failed_novelty_checks": report["failed_novelty_checks"],
                "windows": report["windows"],
                "feature_funnel": report["feature_funnel"],
                "clock": report["clock"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
