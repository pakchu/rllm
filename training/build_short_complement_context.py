"""Materialize reusable G9+macro historical context for short-complement search.

This is context plumbing only.  It reuses the frozen reconstruction APIs from
``training.evaluate_g9_macro_historical`` and writes per-period NPZ files with:

* five-minute ledger arrays: data dict(open,end,high,low,funding,date,end_date)
* G9 + macro_flow target/event/barrier arrays for six sleeves
* causal hourly feature rows aligned to their decision timestamp and next 5m open

No signal tuning or optimizer is performed here; 2024 is marked as the only
selection window and 2025/2026H1 as report windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import evaluate_g9_macro_historical as hist
from training import evaluate_macro_flow_fixed_fresh as macro
from training import optimize_g9_plus_added_alphas as joint
from training import portfolio_opt_added_alpha_update as portfolio
from training import search_macro_flow_alpha_combinations as macro_features
from training import search_meaningful_alpha_combinations as base

OUT = base.ROOT / "research/short_complement_context"
VERSION = 1
NAMES = hist.NAMES
WEIGHTS = np.array([[1.0, 1.5, 0.2, 0.8, 1.0, 1.0]], dtype=float)
WEIGHT_LABEL = "g9_macro1"
COST = 0.0006
SELECTION_WINDOWS = ["2024"]
REPORT_WINDOWS = ["2025", "2026H1"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(base.ROOT))
    except ValueError:
        return str(path)

def array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _historical_trades(m: pd.DataFrame, funding: pd.DataFrame) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Reconstruct exactly the trade clocks used by the frozen evaluator."""
    historical = json.loads(hist.BARRIERS.read_text())
    rex = hist.fixed_historical_clocks()
    native, native_diag = hist.native_2026(m, funding)

    old = m[m.date < pd.Timestamp("2026-01-01")].reset_index(drop=True)
    features = portfolio.feature_frame(old)
    markov = portfolio.markov_active(old, features)

    by_window: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for window, (start, end) in hist.WINDOWS.items():
        if window == "2026H1":
            trades = {name: list(native[name]) for name in joint.G9}
        else:
            trades = {name: [] for name in joint.G9}
            for name in ["fresh_kimchi_fx", "frozen_annual_rank7"]:
                trades[name] = list(historical["windows"][window]["sleeves"][name]["trades"])
            for name, clock in rex.items():
                for row in clock[clock["split"] == window].itertuples():
                    exit_time = pd.Timestamp(row.exit_time).tz_convert(None)
                    trades[name].append(
                        {
                            "entry_date": str(row.entry_time),
                            "exit_date": str(exit_time),
                            "side": "LONG" if row.side > 0 else "SHORT",
                            "exit_kind": "open",
                            "exit_price": float(m.loc[m.date == exit_time, "open"].iloc[0]),
                        }
                    )

            slots = np.arange(143, len(old) - 578, 12)
            next_slot = 0
            for slot in slots:
                if slot < next_slot or not markov[slot]:
                    continue
                endpos = slot + 577
                if not (pd.Timestamp(start) <= old.date.iloc[slot] and old.date.iloc[endpos] < pd.Timestamp(end)):
                    continue
                trades["markov_transition_long"].append(
                    {
                        "entry_date": str(old.date.iloc[slot + 1]),
                        "exit_date": str(old.date.iloc[endpos]),
                        "side": "LONG",
                        "exit_kind": "open",
                        "exit_price": float(old.open.iloc[endpos]),
                    }
                )
                next_slot = endpos + 1

        if window == "2024":
            for name in joint.G9:
                expected = hist.exporter.EXPECTED_COUNTS[name]["test2024"]
                if len(trades[name]) != expected:
                    raise RuntimeError(f"Historical2024 clock mismatch {name}: {len(trades[name])} != {expected}")
        by_window[window] = trades
    return by_window, {"rank7_2026_diagnostics": native_diag}


def build_context() -> dict[str, Any]:
    """Return all in-memory arrays before artifact writing."""
    registration = hist.register()
    market, funding, receipt = hist.merged_market()

    feature_base = base.features(market, funding)
    feature_base, hourly_data, execution_receipt = base.execution_blocks(market, funding, feature_base)
    macro_cols = ["date", "dxy", "usdkrw", "kimchi_premium", "dxy_available", "usdkrw_available", "kimchi_available"]
    feature_frame = pd.concat(
        [feature_base, macro_features.macro_features(market[macro_cols], feature_base.index)],
        axis=1,
    )
    macro_positions, _ = macro.fixed_positions(feature_frame)
    macro_series = pd.Series(
        macro_positions["dollar_flow_plus_regime_switch"],
        index=pd.DatetimeIndex(hourly_data["date"]),
    )

    trades_by_window, trade_receipts = _historical_trades(market, funding)
    periods: dict[str, dict[str, Any]] = {}
    baseline: dict[str, Any] = {}
    trade_counts: dict[str, dict[str, int]] = {}

    for window, (start, end) in hist.WINDOWS.items():
        _, data = hist.blocks(market, funding, start, end)
        dates = pd.DatetimeIndex(data["date"])
        trades = trades_by_window[window]
        targets, events, barriers = joint.clock_arrays({"sleeves": {n: {"trades": trades[n]} for n in joint.G9}}, dates)
        macro_target = macro_series.reindex(macro_series.index.union(dates)).ffill().reindex(dates).fillna(0).to_numpy()
        targets = np.column_stack([targets, macro_target])
        events = np.column_stack([events, dates.isin(macro_series.index)])
        events[0] = True
        barriers = np.column_stack([barriers, np.full(len(dates), np.nan)])

        rows, _ = hist.ledger.simulate(data, targets, events, barriers, WEIGHTS, NAMES, cost=COST)
        baseline[window] = rows[0]
        trade_counts[window] = {name: len(trades[name]) for name in trades}

        feature_next5m = pd.DatetimeIndex(feature_frame.index + pd.Timedelta("5min"))
        feature_mask = (feature_next5m >= pd.Timestamp(start)) & (feature_next5m < pd.Timestamp(end))
        period_features = feature_frame.loc[feature_mask]
        period_next5m = feature_next5m[feature_mask]
        next5m_pos = dates.get_indexer(period_next5m)
        if np.any(next5m_pos < 0):
            raise RuntimeError(f"{window}: feature next5m open not present in 5m period")
        row_for_5m = np.full(len(dates), -1, dtype=np.int64)
        row_for_5m[next5m_pos] = np.arange(len(period_features), dtype=np.int64)

        periods[window] = {
            "data": data,
            "targets": targets,
            "events": events,
            "barriers": barriers,
            "feature_names": np.array(period_features.columns.astype(str), dtype="U"),
            "feature_date": period_features.index.to_numpy(),
            "feature_next5m_date": period_next5m.to_numpy(),
            "features": period_features.to_numpy(dtype=float),
            "feature_row_for_5m": row_for_5m,
        }

    return {
        "registration": registration,
        "receipt": receipt,
        "execution_receipt": execution_receipt,
        "trade_receipts": trade_receipts,
        "trade_counts": trade_counts,
        "baseline": baseline,
        "periods": periods,
    }


def write_artifacts(out: Path = OUT) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    context = build_context()
    expected = json.loads(hist.OUT.joinpath("report.json").read_text())["reports"]

    artifacts = {}
    baseline_checks = {}
    for window, period in context["periods"].items():
        data = period["data"]
        path = out / f"{window}_context.npz"
        arrays = {
            "open": np.asarray(data["open"], dtype=float),
            "end": np.asarray(data["end"], dtype=float),
            "high": np.asarray(data["high"], dtype=float),
            "low": np.asarray(data["low"], dtype=float),
            "funding": np.asarray(data["funding"], dtype=float),
            "date": np.asarray(data["date"]),
            "end_date": np.asarray(data["end_date"]),
            "targets": np.asarray(period["targets"], dtype=float),
            "events": np.asarray(period["events"], dtype=bool),
            "barriers": np.asarray(period["barriers"], dtype=float),
            "sleeve_names": np.array(NAMES, dtype="U"),
            "weights": WEIGHTS[0],
            "weight_label": np.array(WEIGHT_LABEL, dtype="U"),
            "feature_names": period["feature_names"],
            "feature_date": period["feature_date"],
            "feature_next5m_date": period["feature_next5m_date"],
            "features": np.asarray(period["features"], dtype=float),
            "feature_row_for_5m": period["feature_row_for_5m"],
        }
        np.savez_compressed(path, **arrays)
        artifacts[window] = {
            "path": _artifact_path(path),
            "sha256": sha256_file(path),
            "rows_5m": int(len(arrays["date"])),
            "feature_rows_hourly": int(len(arrays["feature_date"])),
            "shape_targets": list(arrays["targets"].shape),
            "shape_events": list(arrays["events"].shape),
            "shape_barriers": list(arrays["barriers"].shape),
            "shape_features": list(arrays["features"].shape),
            "array_hashes": {name: array_sha256(value) for name, value in arrays.items() if value.dtype.kind not in {"U", "O"}},
        }

        actual = context["baseline"][window]
        wanted = expected[window][str(COST)][WEIGHT_LABEL]
        mismatches = {}
        for key, wanted_value in wanted.items():
            actual_value = actual[key]
            if isinstance(wanted_value, dict):
                if actual_value != wanted_value:
                    mismatches[key] = {"actual": actual_value, "expected": wanted_value}
            elif isinstance(wanted_value, bool):
                if bool(actual_value) != wanted_value:
                    mismatches[key] = {"actual": actual_value, "expected": wanted_value}
            elif isinstance(wanted_value, int):
                if int(actual_value) != wanted_value:
                    mismatches[key] = {"actual": actual_value, "expected": wanted_value}
            else:
                if not np.isclose(float(actual_value), float(wanted_value), rtol=0, atol=1e-9):
                    mismatches[key] = {"actual": actual_value, "expected": wanted_value}
        if mismatches:
            raise RuntimeError(f"{window} {WEIGHT_LABEL} cost {COST} baseline mismatch: {mismatches}")
        baseline_checks[window] = {"matched": True, "cost": COST, "label": WEIGHT_LABEL, "row": actual}

    report = {
        "version": VERSION,
        "purpose": "Reusable causal context for short-complement search; no optimizer and no signal tuning.",
        "selection_windows": SELECTION_WINDOWS,
        "report_windows": REPORT_WINDOWS,
        "windows": hist.WINDOWS,
        "sleeve_names": NAMES,
        "weights": {WEIGHT_LABEL: WEIGHTS[0].tolist()},
        "cost": COST,
        "source": {
            "context_builder_sha256": sha256_file(Path(__file__)),
            "frozen_evaluator": str(Path(hist.__file__).relative_to(base.ROOT)),
            "baseline_report": _artifact_path(hist.OUT.joinpath("report.json")),
            "reconstruction": "Imports and reuses evaluate_g9_macro_historical helpers; frozen file unmodified.",
        },
        "receipts": {
            "registration": context["registration"],
            "market": context["receipt"],
            "hourly_execution": context["execution_receipt"],
            **context["trade_receipts"],
        },
        "trade_counts": context["trade_counts"],
        "artifacts": artifacts,
        "baseline_checks": baseline_checks,
        "limitations": [
            "Context plumbing only; downstream short search must select on 2024 only.",
            "2025 and 2026H1 are report/evaluation windows and are not tuned here.",
            "Causal hourly features are labelled by completed-hour decision time and aligned to next 5m open.",
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
