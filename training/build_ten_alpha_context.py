"""Materialize exact ten-sleeve 5m context for alpha-combination search.

This builder is plumbing only: it starts from the frozen six-sleeve
``research/short_complement_context`` NPZ files and appends four fixed sleeves
without optimizing weights:

1. ``oi_pullback`` from the saved OI divergence pullback candidate, evaluated on
   ``evaluate_g9_macro_historical.merged_market()`` with the original global
   stride-6 phase-143 clock, next-bar entry, hold-96, no self-overlap, and
   window-end force exit.
2. ``regional_trend`` from ``evaluate_regional_trend_fresh.position`` over the
   existing completed-hour base+macro features, aligned to the next 5m open.
3. ``dollar_rally_short`` from saved legacy short trade schedules, original
   variant, local indices, fixed no-barrier exits.
4. ``failed_rebound_short`` from saved independent-short finalist schedules,
   local indices and exact barrier prices.

The first six target/event/barrier columns are copied byte-for-byte from the
source contexts.  Added sleeves receive zero default weight so the six-sleeve
parent baseline is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from training import evaluate_g9_macro_historical as hist
from training import evaluate_oi_divergence_fresh as oi_fresh
from training import evaluate_regional_trend_fresh as regional_fresh
from training import g9_joint_net_ledger as ledger
from training import search_macro_flow_alpha_combinations as macro_features
from training import search_meaningful_alpha_combinations as base

SOURCE = base.ROOT / "research/short_complement_context"
OUT = base.ROOT / "research/ten_alpha_context"
LEGACY_TRADES = base.ROOT / "research/legacy_dollar_rally_short/trades.json"
LEGACY_REPORT = base.ROOT / "research/legacy_dollar_rally_short/report.json"
FAILED_TRADES = base.ROOT / "research/independent_short_candidates/finalist_trades.json"
FAILED_REPORT = base.ROOT / "research/independent_short_candidates/report.json"
FAILED_NAME = "failed_rebound_t0.5_h24_tp0.02_sl0.015"

VERSION = 1
WINDOWS: dict[str, tuple[str, str, str]] = {
    "2024": ("2024", "2024-01-01", "2025-01-01"),
    "2025": ("2025", "2025-01-01", "2026-01-01"),
    "full2026H1": ("2026H1", "2026-01-01", "2026-07-01"),
}
NAMES = [
    "fresh_kimchi_fx",
    "frozen_annual_rank7",
    "rex_taker_low_range_position",
    "cand_rex_veto_7",
    "markov_transition_long",
    "macro_flow",
    "oi_pullback",
    "regional_trend",
    "dollar_rally_short",
    "failed_rebound_short",
]
SOURCE_SIX = NAMES[:6]
WEIGHTS = np.array([1.0, 1.5, 0.2, 0.8, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
WEIGHT_LABEL = "g9_macro1_plus_zero_added"
COSTS = (0.0006, 0.001)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(base.ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _load_npz(window: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE / f"{window}_context.npz", allow_pickle=False) as src:
        return {k: src[k] for k in src.files}


def _period_data(src: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(src[k]) for k in ["open", "end", "high", "low", "funding", "date", "end_date"]}


def _event_counts(targets: np.ndarray, events: np.ndarray, barriers: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        name: {
            "nonzero_target_bars": int(np.count_nonzero(targets[:, i])),
            "event_count": int(np.count_nonzero(events[:, i])),
            "barrier_count": int(np.count_nonzero(np.isfinite(barriers[:, i]))),
        }
        for i, name in enumerate(NAMES)
    }


def _set_fixed_trade(target: np.ndarray, event: np.ndarray, barrier_arr: np.ndarray, *, entry: int, exit_: int,
                     side: float, n: int, barrier: bool = False, exit_price: float | None = None) -> None:
    if entry < 0 or entry >= n:
        raise ValueError(f"entry out of local bounds: {entry} not in [0,{n})")
    if exit_ < entry:
        raise ValueError(f"exit before entry: {entry}>{exit_}")
    stop = min(exit_ + int(barrier), n)
    target[entry:stop] = side
    event[entry] = True
    if barrier:
        if exit_ >= n:
            raise ValueError("barrier exit beyond window")
        if exit_price is None or not np.isfinite(exit_price) or exit_price <= 0:
            raise ValueError("invalid barrier exit price")
        barrier_arr[exit_] = float(exit_price)
        if exit_ + 1 < n:
            event[exit_ + 1] = True
    elif exit_ < n:
        event[exit_] = True


def _short_arrays_from_legacy(window: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    trades = json.loads(LEGACY_TRADES.read_text())[window]["original"]
    target = np.zeros(n, dtype=float)
    event = np.zeros(n, dtype=bool)
    barriers = np.full(n, np.nan, dtype=float)
    for t in trades:
        if bool(t.get("barrier", False)):
            raise ValueError("legacy dollar rally original unexpectedly has barrier exits")
        _set_fixed_trade(target, event, barriers, entry=int(t["entry_index"]), exit_=int(t["exit_index"]), side=-1.0, n=n)
    return target, event, barriers, len(trades)


def _short_arrays_from_failed(source_window: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    trades = json.loads(FAILED_TRADES.read_text())[source_window][FAILED_NAME]
    target = np.zeros(n, dtype=float)
    event = np.zeros(n, dtype=bool)
    barriers = np.full(n, np.nan, dtype=float)
    for t in trades:
        _set_fixed_trade(
            target,
            event,
            barriers,
            entry=int(t["entry"]),
            exit_=int(t["exit"]),
            side=-1.0,
            n=n,
            barrier=bool(t["barrier"]),
            exit_price=float(t["exit_price"]),
        )
    return target, event, barriers, len(trades)


def _make_oi_global_schedule(market: pd.DataFrame) -> list[tuple[int, int]]:
    payload = json.loads(oi_fresh.CONFIG.read_text())
    signal = payload["signal"]
    candidate = {**signal, "hold_bars": int(signal["hold_bars_5m"]), "stride_bars": int(signal["stride_bars_5m"])}
    feat = oi_fresh.oi_eval._feature_frame(market, window_size=144)
    active = oi_fresh.oi_eval._candidate_active(feat, candidate)
    active &= pd.to_numeric(market.open_interest_available, errors="coerce").fillna(0).to_numpy() > 0.5
    hold = int(candidate["hold_bars"])
    stride = int(candidate["stride_bars"])
    slots = np.arange(143, len(market) - hold - 2, stride, dtype=np.int64)
    scheduled: list[tuple[int, int]] = []
    next_allowed = 0
    for signal_pos in slots[active[slots]]:
        signal_i = int(signal_pos)
        if signal_i < next_allowed:
            continue
        entry = signal_i + 1
        exit_ = entry + hold
        if exit_ >= len(market):
            continue
        scheduled.append((entry, exit_))
        next_allowed = exit_ + 1
    return scheduled


def _oi_arrays_for_period(market_dates: pd.DatetimeIndex, schedule: list[tuple[int, int]], start: str, end: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    target = np.zeros(n, dtype=float)
    event = np.zeros(n, dtype=bool)
    barriers = np.full(n, np.nan, dtype=float)
    count = 0
    for entry_g, exit_g in schedule:
        entry_ts = market_dates[entry_g]
        if not (start_ts <= entry_ts < end_ts):
            continue
        entry = int((entry_ts - start_ts) / pd.Timedelta("5min"))
        exit_ts = min(market_dates[exit_g], end_ts)
        exit_ = int((exit_ts - start_ts) / pd.Timedelta("5min"))
        _set_fixed_trade(target, event, barriers, entry=entry, exit_=exit_, side=1.0, n=n)
        count += 1
    return target, event, barriers, count


def _regional_series_by_window(market: pd.DataFrame, funding: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    feature_base = base.features(market, funding)
    feature_base, hourly_data, receipt = base.execution_blocks(market, funding, feature_base)
    macro_cols = ["date", "dxy", "usdkrw", "kimchi_premium", "dxy_available", "usdkrw_available", "kimchi_available"]
    feature_frame = pd.concat([feature_base, macro_features.macro_features(market[macro_cols], feature_base.index)], axis=1)
    positions, raw = regional_fresh.position(feature_frame)
    series = pd.Series(positions, index=pd.DatetimeIndex(hourly_data["date"]))
    diag = {"engine_receipt": receipt, "nonzero_decision_hours_full": int(np.count_nonzero(raw))}
    return series, diag


def _regional_arrays_for_dates(series: pd.Series, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    target = series.reindex(series.index.union(dates)).ffill().reindex(dates).fillna(0).to_numpy(float)
    event = dates.isin(series.index)
    if len(event):
        event[0] = True
    barriers = np.full(len(dates), np.nan, dtype=float)
    return target, event, barriers, int(np.count_nonzero(series.reindex(dates.intersection(series.index)).to_numpy(float)))


def _simulate_one(d: dict[str, np.ndarray], target: np.ndarray, event: np.ndarray, barriers: np.ndarray, cost: float) -> dict[str, Any]:
    rows, _ = ledger.simulate(d, target[:, None], event[:, None], barriers[:, None], np.ones((1, 1)), ["independent_short"], cost=cost, net_cap=4.5)
    return rows[0]


def _compare_metric_rows(actual: Mapping[str, Any], expected: Mapping[str, Any], *, ignore: set[str] | None = None) -> dict[str, Any]:
    ignore = ignore or set()
    mismatches: dict[str, Any] = {}
    for key, wanted in expected.items():
        if key in ignore or key not in actual:
            continue
        got = actual[key]
        if isinstance(wanted, Mapping):
            if dict(got) != dict(wanted):
                mismatches[key] = {"actual": got, "expected": wanted}
        elif isinstance(wanted, bool):
            if bool(got) != bool(wanted):
                mismatches[key] = {"actual": got, "expected": wanted}
        elif isinstance(wanted, int):
            if int(got) != int(wanted):
                mismatches[key] = {"actual": got, "expected": wanted}
        elif isinstance(wanted, (float, int)):
            if not np.isclose(float(got), float(wanted), rtol=0.0, atol=1e-9):
                mismatches[key] = {"actual": got, "expected": wanted}
    return mismatches


def build_context() -> dict[str, Any]:
    market, funding, market_receipt = hist.merged_market()
    market_dates = pd.DatetimeIndex(market.date)
    oi_schedule = _make_oi_global_schedule(market)
    regional_series, regional_diag = _regional_series_by_window(market, funding)

    legacy_report = json.loads(LEGACY_REPORT.read_text())["reports"]
    failed_report = json.loads(FAILED_REPORT.read_text())["reports"]

    periods: dict[str, dict[str, Any]] = {}
    validation: dict[str, Any] = {}
    counts: dict[str, Any] = {}

    for out_window, (source_window, start, end) in WINDOWS.items():
        src = _load_npz(source_window)
        n = len(src["date"])
        dates = pd.DatetimeIndex(src["date"])
        if dates[0] != pd.Timestamp(start) or src["end_date"][-1] != np.datetime64(pd.Timestamp(end)):
            raise ValueError(f"{out_window}: source context date range mismatch")
        if src["sleeve_names"].tolist() != SOURCE_SIX:
            raise ValueError(f"{out_window}: source sleeve order mismatch")

        oi_t, oi_e, oi_b, oi_count = _oi_arrays_for_period(market_dates, oi_schedule, start, end, n)
        reg_t, reg_e, reg_b, reg_count = _regional_arrays_for_dates(regional_series, dates)
        legacy_t, legacy_e, legacy_b, legacy_count = _short_arrays_from_legacy(out_window, n)
        failed_t, failed_e, failed_b, failed_count = _short_arrays_from_failed(source_window, n)

        targets = np.column_stack([src["targets"], oi_t, reg_t, legacy_t, failed_t])
        events = np.column_stack([src["events"], oi_e, reg_e, legacy_e, failed_e])
        barriers = np.column_stack([src["barriers"], oi_b, reg_b, legacy_b, failed_b])

        # Validation: appended-zero weights must leave parent six-sleeve result unchanged.
        d = _period_data(src)
        parent_rows, _ = ledger.simulate(d, src["targets"], src["events"], src["barriers"], src["weights"][None, :], SOURCE_SIX, cost=0.0006)
        ten_rows, _ = ledger.simulate(d, targets, events, barriers, WEIGHTS[None, :], NAMES, cost=0.0006)
        parent_mismatch = _compare_metric_rows(ten_rows[0], parent_rows[0], ignore={"weights_notional"})
        if parent_mismatch:
            raise RuntimeError(f"{out_window}: zero-added parent mismatch {parent_mismatch}")

        short_checks: dict[str, Any] = {}
        for sleeve_name, t, e, b, report_obj in [
            ("dollar_rally_short", legacy_t, legacy_e, legacy_b, legacy_report[out_window]["original"]["metrics"]),
            ("failed_rebound_short", failed_t, failed_e, failed_b, failed_report[source_window][FAILED_NAME]),
        ]:
            short_checks[sleeve_name] = {}
            for cost in COSTS:
                actual = _simulate_one(d, t, e, b, cost)
                expected = report_obj[str(cost)]
                mismatches = _compare_metric_rows(actual, expected, ignore={"win_rate"})
                if mismatches:
                    raise RuntimeError(f"{out_window} {sleeve_name} cost {cost} mismatch: {mismatches}")
                short_checks[sleeve_name][str(cost)] = {"matched": True, "row": actual}

        validation[out_window] = {
            "source_first_six_array_hashes": {
                key: array_sha256(src[key]) for key in ["targets", "events", "barriers"]
            },
            "output_first_six_array_hashes": {
                "targets": array_sha256(targets[:, :6]),
                "events": array_sha256(events[:, :6]),
                "barriers": array_sha256(barriers[:, :6]),
            },
            "first_six_byte_exact": bool(
                np.array_equal(targets[:, :6], src["targets"])
                and np.array_equal(events[:, :6], src["events"])
                and np.array_equal(barriers[:, :6], src["barriers"], equal_nan=True)
                and targets[:, :6].tobytes() == np.ascontiguousarray(src["targets"]).tobytes()
                and events[:, :6].tobytes() == np.ascontiguousarray(src["events"]).tobytes()
                and np.nan_to_num(barriers[:, :6], nan=9.87654321e123).tobytes()
                == np.nan_to_num(np.ascontiguousarray(src["barriers"]), nan=9.87654321e123).tobytes()
            ),
            "zero_added_parent_cost_0.0006": {"matched": True, "row": ten_rows[0]},
            "standalone_short_checks": short_checks,
        }
        if not validation[out_window]["first_six_byte_exact"]:
            raise RuntimeError(f"{out_window}: first six arrays not byte-exact")

        counts[out_window] = {
            "oi_pullback_trades": oi_count,
            "regional_nonzero_decision_hours": reg_count,
            "dollar_rally_short_trades": legacy_count,
            "failed_rebound_short_trades": failed_count,
            "bounds_by_sleeve": _event_counts(targets, events, barriers),
        }
        periods[out_window] = {
            "source_window": source_window,
            "source": src,
            "targets": targets,
            "events": events,
            "barriers": barriers,
        }

    return {
        "periods": periods,
        "validation": validation,
        "counts": counts,
        "receipts": {
            "market": market_receipt,
            "oi_global_schedule_count": len(oi_schedule),
            "regional": regional_diag,
        },
    }


def write_artifacts(out: Path = OUT) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    context = build_context()
    artifacts: dict[str, Any] = {}
    source_artifacts: dict[str, Any] = {}

    for out_window, period in context["periods"].items():
        src = period["source"]
        arrays = {k: src[k] for k in src.keys() if k not in {"targets", "events", "barriers", "sleeve_names", "weights", "weight_label"}}
        arrays.update({
            "targets": np.asarray(period["targets"], dtype=float),
            "events": np.asarray(period["events"], dtype=bool),
            "barriers": np.asarray(period["barriers"], dtype=float),
            "sleeve_names": np.array(NAMES, dtype="U"),
            "names": np.array(NAMES, dtype="U"),
            "weights": WEIGHTS,
            "weight_label": np.array(WEIGHT_LABEL, dtype="U"),
        })
        path = out / f"{out_window}_context.npz"
        np.savez_compressed(path, **arrays)
        artifacts[out_window] = {
            "path": _artifact_path(path),
            "sha256": sha256_file(path),
            "rows_5m": int(len(arrays["date"])),
            "shape_targets": list(arrays["targets"].shape),
            "shape_events": list(arrays["events"].shape),
            "shape_barriers": list(arrays["barriers"].shape),
            "array_hashes": {name: array_sha256(value) for name, value in arrays.items() if value.dtype.kind not in {"U", "O"}},
        }
        src_path = SOURCE / f"{period['source_window']}_context.npz"
        source_artifacts[out_window] = {"path": _artifact_path(src_path), "sha256": sha256_file(src_path)}

    # Compatibility alias for the existing optimizer, whose historical-window key
    # is ``2026H1`` while this artifact is explicitly labelled full2026H1.
    if "full2026H1" in artifacts:
        artifacts["2026H1"] = dict(artifacts["full2026H1"])
        source_artifacts["2026H1"] = dict(source_artifacts["full2026H1"])

    report = {
        "version": VERSION,
        "purpose": "Exact ten-sleeve 5m target/event/barrier context for downstream alpha-combination weighting; no optimizer.",
        "windows": {
            **{k: {"source_window": v[0], "start": v[1], "end_exclusive": v[2]} for k, v in WINDOWS.items()},
            **({"2026H1": {"source_window": "2026H1", "start": "2026-01-01", "end_exclusive": "2026-07-01", "alias_for": "full2026H1"}} if "full2026H1" in artifacts else {}),
        },
        "sleeve_names": NAMES,
        "weights": {WEIGHT_LABEL: WEIGHTS.tolist()},
        "arrays": ["open", "end", "high", "low", "funding", "date", "end_date", "targets", "events", "barriers", "names"],
        "preserved_source_arrays": "targets/events/barriers first six columns copied from short_complement_context and validated byte-exact in-memory before write.",
        "source_artifacts": source_artifacts,
        "source_hashes": {
            "builder": sha256_file(Path(__file__)),
            "short_context_report": sha256_file(SOURCE / "report.json"),
            "oi_config": sha256_file(oi_fresh.CONFIG),
            "legacy_trades": sha256_file(LEGACY_TRADES),
            "failed_trades": sha256_file(FAILED_TRADES),
            "hist_evaluator": sha256_file(Path(hist.__file__)),
            "regional_evaluator": sha256_file(Path(regional_fresh.__file__)),
            "oi_feature_module": sha256_file(Path(oi_fresh.oi_eval.__file__)),
        },
        "construction": {
            "oi_pullback": "evaluate_oi_divergence_fresh.oi_eval features on evaluate_g9_macro_historical.merged_market; saved fixed four-gate long config; global np.arange phase143 stride6; next-bar entry; hold96; no self-overlap; force exit at window end.",
            "regional_trend": "evaluate_regional_trend_fresh.position over base.features plus macro_features completed-hour rows; targets aligned to hourly T+5m 5m bars.",
            "dollar_rally_short": "saved legacy_dollar_rally_short/trades.json original schedules; local indices; short; no barrier/TP/SL.",
            "failed_rebound_short": f"saved independent_short_candidates/finalist_trades.json {FAILED_NAME}; local indices; short; exact barrier prices.",
        },
        "receipts": context["receipts"],
        "counts": context["counts"],
        "validation": context["validation"],
        "artifacts": artifacts,
        "live_enabled": False,
        "limitations": [
            "Materialization only; root process is expected to optimize weights downstream.",
            "Default added-sleeve weights are zero to prove parent baseline invariance.",
            "Historical contexts are exposed research artifacts and make no pristine OOS claim.",
        ],
    }
    report_path = out / "report.json"
    base.write_json(report_path, _json_safe(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = write_artifacts(args.out)
    for window, artifact in report["artifacts"].items():
        print(window, artifact["path"], artifact["sha256"])


if __name__ == "__main__":
    main()
