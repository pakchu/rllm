"""Build IVFHR-72 source-only clocks and support diagnostics without outcomes."""
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

from training import build_intrinsic_volume_latent_impact_relay_support as ivlir
from training import preregister_intrinsic_volume_flow_handoff_relay as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "e01e7f5af034adf98c0eef1e086ed1265c02998641f39d8cddd5137089f4153e"
)
DEFAULT_OUTPUT = Path(
    "results/intrinsic_volume_flow_handoff_relay_support_2026-07-23.json"
)
DEFAULT_CLOCK = Path(
    "data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz"
)
SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_START = SOURCE_START
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = TRAIN_END
SELECTION_END = SOURCE_END
BAR = pd.Timedelta(minutes=5)
DAY = pd.Timedelta(days=1)
CLOCK_COLUMNS = [
    "clock_name",
    "source_day",
    "decision_time",
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
        raise RuntimeError("IVFHR preregistration file changed")
    frozen = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(frozen)
    if frozen["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("IVFHR policy changed after preregistration")
    for path, expected in (
        (prereg.MARKET_MANIFEST, frozen["source_contract"]["market_manifest_sha256"]),
        (prereg.MARKET_SOURCE, frozen["source_contract"]["market_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"IVFHR frozen source changed: {path}")
    inherited = ivlir.prereg.Policy()
    policy = prereg.Policy()
    shared = {
        "utc_day_volume_lookback_days": inherited.utc_day_volume_lookback_days,
        "utc_day_volume_min_days": inherited.utc_day_volume_min_days,
        "intrinsic_volume_fraction": inherited.intrinsic_volume_fraction,
        "latest_anchor_minute_utc": inherited.latest_anchor_minute_utc,
        "event_reference_days": inherited.event_reference_days,
        "event_reference_min_days": inherited.event_reference_min_days,
        "entry_delay_bars": inherited.entry_delay_bars,
        "hold_bars": inherited.hold_bars,
    }
    expected_shared = {key: getattr(policy, key) for key in shared}
    if shared != expected_shared:
        raise RuntimeError("IVFHR inherited equal-notional clock contract changed")
    if inherited.absolute_flow_quantile != policy.current_flow_quantile:
        raise RuntimeError("IVFHR inherited q60 flow contract changed")
    return frozen


def load_source(path: str | Path = prereg.MARKET_SOURCE) -> pd.DataFrame:
    """Load the exact grid through IVLIR's already-tested causal source validator."""
    return ivlir.load_source(path)


def build_anchor_features(
    frame: pd.DataFrame,
    *,
    fixed_noon: bool = False,
) -> pd.DataFrame:
    """Reuse the frozen equal-notional anchor math, then expose only causal fields."""
    anchors = ivlir.build_anchor_features(
        frame, ivlir.prereg.Policy(), fixed_noon=fixed_noon
    )
    referenced = ivlir.apply_causal_event_references(
        anchors, ivlir.prereg.Policy()
    )
    return annotate_handoff_state(referenced)


def annotate_handoff_state(
    referenced: pd.DataFrame,
    policy: prereg.Policy = prereg.Policy(),
) -> pd.DataFrame:
    """Add calendar-consecutive state transitions without reading a future row."""
    if referenced.empty:
        empty = referenced.copy()
        for column, dtype in (
            ("calendar_consecutive", bool),
            ("prior_state_side", object),
            ("prior_state_run_length", int),
            ("handoff", bool),
            ("persistence", bool),
            ("price_lag", bool),
            ("strong_new_flow", bool),
            ("primary", bool),
        ):
            empty[column] = pd.Series(dtype=dtype)
        return empty
    ordered = referenced.sort_values("source_day", kind="mergesort").reset_index(
        drop=True
    )
    # The reused IVLIR reference helper carries its own rejected-event label.
    # Drop that label before materializing the independently frozen IVFHR identity.
    ordered = ordered.drop(columns=["primary"], errors="ignore")
    if ordered["source_day"].duplicated().any():
        raise RuntimeError("IVFHR has multiple anchors for one UTC day")
    records: list[dict[str, Any]] = []
    previous_day: pd.Timestamp | None = None
    previous_side = ""
    previous_run = 0
    for row in ordered.itertuples(index=False):
        day = pd.Timestamp(row.source_day)
        side = str(row.side)
        if side not in {"LONG", "SHORT"}:
            raise RuntimeError("IVFHR anchor has invalid flow side")
        consecutive = previous_day is not None and day == previous_day + DAY
        prior_state_side = previous_side if consecutive else ""
        prior_state_run = previous_run if consecutive else 0
        handoff = bool(consecutive and side != previous_side)
        persistence = bool(consecutive and side == previous_side)
        price_lag = bool(float(row.directional_return) <= 0.0)
        strong_new_flow = bool(row.reference_ready and row.flow_pass)
        primary = bool(
            row.reference_ready
            and handoff
            and prior_state_run >= policy.prior_state_min_anchors
            and strong_new_flow
            and price_lag
        )
        records.append(
            {
                "calendar_consecutive": consecutive,
                "prior_state_side": prior_state_side,
                "prior_state_run_length": prior_state_run,
                "handoff": handoff,
                "persistence": persistence,
                "price_lag": price_lag,
                "strong_new_flow": strong_new_flow,
                "primary": primary,
            }
        )
        if consecutive and side == previous_side:
            previous_run += 1
        else:
            previous_run = 1
        previous_side = side
        previous_day = day
    return pd.concat([ordered, pd.DataFrame(records)], axis=1)


def _contained(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        frame["entry_time"].ge(start) & frame["exit_time"].le(end)
    ].copy()


def _greedy_schedule(frame: pd.DataFrame, clock_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    selected: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in frame.sort_values("entry_time", kind="mergesort").itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.append(
            {
                "clock_name": clock_name,
                "source_day": pd.Timestamp(row.source_day),
                "decision_time": entry,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": str(row.side),
            }
        )
        previous_exit = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def schedule_across_splits(frame: pd.DataFrame, clock_name: str) -> pd.DataFrame:
    parts = [
        _greedy_schedule(_contained(frame, TRAIN_START, TRAIN_END), clock_name),
        _greedy_schedule(
            _contained(frame, SELECTION_START, SELECTION_END), clock_name
        ),
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    ).reset_index(drop=True)


def _exact_clock_variant(
    primary: pd.DataFrame, clock_name: str, sides: Iterable[str]
) -> pd.DataFrame:
    variant = primary.copy()
    variant["clock_name"] = clock_name
    variant["side"] = list(sides)
    return variant.loc[:, CLOCK_COLUMNS]


def build_clocks(
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    reference_ready = features["reference_ready"]
    run_ready = features["prior_state_run_length"].ge(
        prereg.Policy().prior_state_min_anchors
    )
    primary = schedule_across_splits(features.loc[features["primary"]], "primary")
    clocks = {
        "primary": primary,
        "any_handoff": schedule_across_splits(
            features.loc[reference_ready & features["handoff"] & features["price_lag"]],
            "any_handoff",
        ),
        "no_price_lag": schedule_across_splits(
            features.loc[
                reference_ready
                & features["handoff"]
                & run_ready
                & features["strong_new_flow"]
            ],
            "no_price_lag",
        ),
        "no_flow_strength": schedule_across_splits(
            features.loc[
                reference_ready
                & features["handoff"]
                & run_ready
                & features["price_lag"]
            ],
            "no_flow_strength",
        ),
        "persistence_level": schedule_across_splits(
            features.loc[
                reference_ready
                & features["persistence"]
                & run_ready
                & features["strong_new_flow"]
                & features["price_lag"]
            ],
            "persistence_level",
        ),
        "fixed_noon_handoff": schedule_across_splits(
            fixed_noon_features.loc[fixed_noon_features["primary"]],
            "fixed_noon_handoff",
        ),
    }
    clocks["exact_side_flip"] = _exact_clock_variant(
        primary,
        "exact_side_flip",
        ["SHORT" if side == "LONG" else "LONG" for side in primary["side"]],
    )
    seed = int.from_bytes(
        hashlib.sha256(b"IVFHR-72|side").digest()[:8], "big", signed=False
    )
    rng = np.random.default_rng(seed)
    random_side = np.where(rng.integers(0, 2, len(primary)) == 0, "SHORT", "LONG")
    clocks["deterministic_random_side"] = _exact_clock_variant(
        primary, "deterministic_random_side", random_side
    )
    return clocks


def _window(clock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return clock.loc[clock["entry_time"].ge(start) & clock["exit_time"].le(end)]


def _longest_run(sides: Iterable[str]) -> int:
    best = 0
    current = 0
    previous = None
    for side in sides:
        if side == previous:
            current += 1
        else:
            current = 1
            previous = side
        best = max(best, current)
    return best


def _maximum_gap_days(times: pd.Series) -> float | None:
    ordered = times.sort_values(kind="mergesort")
    if len(ordered) < 2:
        return None
    return float(ordered.diff().dropna().max() / DAY)


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
    side_counts = ordered["side"].value_counts().to_dict()
    calendar_time = ordered["entry_time"].dt.tz_convert(None)
    month_counts = calendar_time.dt.to_period("M").astype(str).value_counts()
    quarter_counts = calendar_time.dt.to_period("Q").astype(str).value_counts()
    return {
        "events": total,
        "long": int(side_counts.get("LONG", 0)),
        "short": int(side_counts.get("SHORT", 0)),
        "long_share": float(side_counts.get("LONG", 0) / total) if total else None,
        "short_share": float(side_counts.get("SHORT", 0) / total) if total else None,
        "active_months": int(len(month_counts)),
        "maximum_month_share": float(month_counts.max() / total) if total else None,
        "maximum_quarter_share": float(quarter_counts.max() / total) if total else None,
        "maximum_calendar_gap_days": _maximum_gap_days(ordered["entry_time"]),
        "maximum_same_side_run": _longest_run(ordered["side"]),
    }


def _combine_clocks(clocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nonempty = [clock for clock in clocks.values() if not clock.empty]
    if not nonempty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    ).reset_index(drop=True)


def _support_report(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
    clocks: dict[str, pd.DataFrame],
    clock_path: Path,
) -> dict[str, Any]:
    primary = clocks["primary"]
    windows = {
        "all": clock_stats(primary),
        "train_2020_2022": clock_stats(_window(primary, TRAIN_START, TRAIN_END)),
        "2020": clock_stats(
            _window(primary, TRAIN_START, pd.Timestamp("2021-01-01T00:00:00Z"))
        ),
        "2021": clock_stats(
            _window(
                primary,
                pd.Timestamp("2021-01-01T00:00:00Z"),
                pd.Timestamp("2022-01-01T00:00:00Z"),
            )
        ),
        "2022": clock_stats(
            _window(primary, pd.Timestamp("2022-01-01T00:00:00Z"), TRAIN_END)
        ),
        "selection_2023": clock_stats(
            _window(primary, SELECTION_START, SELECTION_END)
        ),
        "selection_2023_h1": clock_stats(
            _window(
                primary, SELECTION_START, pd.Timestamp("2023-07-01T00:00:00Z")
            )
        ),
        "selection_2023_h2": clock_stats(
            _window(
                primary, pd.Timestamp("2023-07-01T00:00:00Z"), SELECTION_END
            )
        ),
    }
    frozen_gate = prereg._core_manifest()["source_support_gate"]
    low_share, high_share = frozen_gate["each_side_share_range_all_train_selection"]

    def side_ok(stats: dict[str, Any]) -> bool:
        return bool(
            stats["events"]
            and stats["long_share"] is not None
            and low_share <= stats["long_share"] <= high_share
            and low_share <= stats["short_share"] <= high_share
        )

    all_stats = windows["all"]
    checks = {
        "source_exact_grid": len(frame) == 420_768,
        "train_events_min": windows["train_2020_2022"]["events"]
        >= frozen_gate["train_events_min"],
        "each_train_year_events_min": all(
            windows[str(year)]["events"] >= frozen_gate["each_train_year_events_min"]
            for year in (2020, 2021, 2022)
        ),
        "selection_events_min": windows["selection_2023"]["events"]
        >= frozen_gate["selection_events_min"],
        "each_selection_half_events_min": all(
            windows[name]["events"] >= frozen_gate["each_selection_half_events_min"]
            for name in ("selection_2023_h1", "selection_2023_h2")
        ),
        "side_share_all_train_selection": all(
            side_ok(windows[name])
            for name in ("all", "train_2020_2022", "selection_2023")
        ),
        "active_months_min": all_stats["active_months"]
        >= frozen_gate["active_months_min"],
        "maximum_single_month_share": bool(
            all_stats["maximum_month_share"] is not None
            and all_stats["maximum_month_share"]
            <= frozen_gate["maximum_single_month_share"]
        ),
        "maximum_single_quarter_share": bool(
            all_stats["maximum_quarter_share"] is not None
            and all_stats["maximum_quarter_share"]
            <= frozen_gate["maximum_single_quarter_share"]
        ),
        "maximum_calendar_gap_days": bool(
            all_stats["maximum_calendar_gap_days"] is not None
            and all_stats["maximum_calendar_gap_days"]
            <= frozen_gate["maximum_calendar_gap_days"]
        ),
        "maximum_same_side_run": all_stats["maximum_same_side_run"]
        <= frozen_gate["maximum_same_side_run"],
        "clock_has_no_market_value_or_outcome_columns": list(
            _combine_clocks(clocks).columns
        )
        == CLOCK_COLUMNS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    all_clocks = _combine_clocks(clocks)
    return {
        "protocol_version": "intrinsic_volume_flow_handoff_relay_support_v1",
        "policy_id": prereg.Policy().policy_id,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "support_passed": not failed,
        "authorized_next_stage": "freeze_strict_evaluator" if not failed else None,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(PREREGISTRATION),
            "manifest_hash": json.loads(PREREGISTRATION.read_text())["manifest_hash"],
        },
        "source": {
            "path": prereg.MARKET_SOURCE,
            "sha256": sha256_file(prereg.MARKET_SOURCE),
            "rows": len(frame),
            "first_timestamp": frame["date"].iloc[0].isoformat(),
            "last_timestamp": frame["date"].iloc[-1].isoformat(),
            "complete_utc_days": int(frame["date"].dt.floor("D").nunique()),
        },
        "feature_funnel": {
            "first_passage_anchors": len(features),
            "reference_ready": int(features["reference_ready"].sum()),
            "calendar_consecutive": int(features["calendar_consecutive"].sum()),
            "prior_state_ready": int(
                features["prior_state_run_length"].ge(
                    prereg.Policy().prior_state_min_anchors
                ).sum()
            ),
            "handoff": int(features["handoff"].sum()),
            "strong_new_flow": int(features["strong_new_flow"].sum()),
            "price_lag": int(features["price_lag"].sum()),
            "raw_primary": int(features["primary"].sum()),
            "fixed_noon_anchors": len(fixed_noon_features),
            "fixed_noon_raw_primary": int(fixed_noon_features["primary"].sum()),
        },
        "windows": windows,
        "controls": {name: clock_stats(clock) for name, clock in clocks.items()},
        "support_checks": checks,
        "failed_checks": failed,
        "clock": {
            "path": str(clock_path),
            "rows": len(all_clocks),
            "columns": CLOCK_COLUMNS,
            "sha256": sha256_file(clock_path),
        },
        "leakage_guard": {
            "daily_volume_reference_shifted_one_complete_day": True,
            "event_reference_strictly_prior": True,
            "current_anchor_excluded_from_quantiles": True,
            "state_uses_only_prior_calendar_consecutive_anchors": True,
            "invalid_calendar_day_resets_state": True,
            "signals_use_only_day_start_through_completed_anchor": True,
            "decision_time_equals_next_bar_open_after_anchor_close": True,
            "no_post_entry_price_access": True,
            "no_return_or_pnl_field_written": True,
            "llm_or_model_trained": False,
        },
    }


def write_clock(path: str | Path, clocks: dict[str, pd.DataFrame]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = _combine_clocks(clocks)
    if list(combined.columns) != CLOCK_COLUMNS:
        raise RuntimeError("IVFHR clock schema changed")
    if not combined["decision_time"].eq(combined["entry_time"]).all():
        raise RuntimeError("IVFHR decision/entry timestamp contract changed")
    if not combined["exit_time"].eq(combined["entry_time"] + 72 * BAR).all():
        raise RuntimeError("IVFHR hold contract changed")
    csv_bytes = combined.to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    ).encode()
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            zipped.write(csv_bytes)


def build_support(
    *,
    source: str | Path = prereg.MARKET_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK,
) -> dict[str, Any]:
    validate_frozen_inputs()
    if str(source) != prereg.MARKET_SOURCE:
        raise RuntimeError("IVFHR support must use the frozen source path")
    frame = load_source(source)
    features = build_anchor_features(frame)
    fixed_noon_features = build_anchor_features(frame, fixed_noon=True)
    clocks = build_clocks(features, fixed_noon_features)
    clock_path = Path(clock_output)
    write_clock(clock_path, clocks)
    report = _support_report(frame, features, fixed_noon_features, clocks, clock_path)
    core = {key: value for key, value in report.items() if key != "created_at"}
    report["report_manifest_hash"] = canonical_hash(core)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=prereg.MARKET_SOURCE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    args = parser.parse_args()
    report = build_support(
        source=args.source, output=args.output, clock_output=args.clock_output
    )
    print(
        json.dumps(
            {
                "policy_id": report["policy_id"],
                "support_passed": report["support_passed"],
                "failed_checks": report["failed_checks"],
                "windows": report["windows"],
                "feature_funnel": report["feature_funnel"],
                "clock": report["clock"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
