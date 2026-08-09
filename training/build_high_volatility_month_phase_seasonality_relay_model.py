"""Fit and freeze the pre-2023 HVMPS-12 month-phase model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_month_phase_seasonality_relay as prereg


PREREG_SHA = "2e1f9577317c7ec77c49a35b6e0ae781ebc322beb5a8d207296c8ae63f3b734e"
MODEL = Path("data/high_volatility_month_phase_seasonality_relay_model_2026-08-10.json")
RESULT = Path("results/high_volatility_month_phase_seasonality_relay_model_freeze_2026-08-10.json")
BUILDER = Path("training/build_high_volatility_month_phase_seasonality_relay_model.py")
FIT_START = pd.Timestamp("2020-01-01T00:00:00Z")
FIT_END = pd.Timestamp("2023-01-01T00:00:00Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_labels(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy(); frame["date"] = pd.to_datetime(frame.date, utc=True); frame = frame.sort_values("date").set_index("date")
    opens = pd.to_numeric(frame.open, errors="coerce")
    decisions = pd.date_range(FIT_START, FIT_END, freq="1D", inclusive="left")
    rows = []
    for decision in decisions:
        day = decision.day
        if day > 28:
            continue
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        expected = pd.date_range(entry, exit_time, freq="5min")
        path = opens.reindex(expected)
        valid = len(path) == 145 and np.isfinite(path).all() and path.gt(0).all()
        if valid:
            rows.append({"decision_time": decision, "day_of_month": day, "label": float(np.log(path.iloc[-1] / path.iloc[0]))})
    return pd.DataFrame(rows)


def fit_model(labels: pd.DataFrame) -> dict[str, Any]:
    stats = labels.groupby("day_of_month").label.agg(["count", "mean"]).reset_index()
    eligible = stats[stats["count"].ge(30)].copy()
    positive = eligible[eligible["mean"].gt(0)].sort_values(["mean", "day_of_month"], ascending=[False, True]).head(4)
    negative = eligible[eligible["mean"].lt(0)].sort_values(["mean", "day_of_month"], ascending=[True, True]).head(4)
    if len(positive) != 4 or len(negative) != 4:
        raise RuntimeError("HVMPS insufficient signed month-phase slots")
    selected = [
        {"day_of_month": int(row.day_of_month), "side": 1, "fit_count": int(row.count), "fit_mean_log_return": float(row.mean)}
        for row in positive.itertuples(index=False)
    ] + [
        {"day_of_month": int(row.day_of_month), "side": -1, "fit_count": int(row.count), "fit_mean_log_return": float(row.mean)}
        for row in negative.itertuples(index=False)
    ]
    selected.sort(key=lambda item: item["day_of_month"])
    return {"fit_window": [FIT_START.isoformat(), FIT_END.isoformat()], "slot_floor": 30, "positive_slots": 4, "negative_slots": 4, "selected": selected, "all_slot_stats": [{"day_of_month": int(row.day_of_month), "fit_count": int(row.count), "fit_mean_log_return": float(row.mean)} for row in stats.itertuples(index=False)]}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA or sha(prereg.MARKET) != prereg.MARKET_SHA:
        raise RuntimeError("HVMPS predecessor drift")
    market = pd.read_csv(prereg.MARKET, usecols=["date", "open"])
    labels = fit_labels(market)
    model_core = {"protocol_version": "hvmps_12_frozen_model_v1", "policy_id": prereg.POLICY_ID, "preregistration_sha256": PREREG_SHA, "pretraining_outcomes_opened": True, "oos_source_incidence_opened": False, "oos_outcomes_opened": False, **fit_model(labels)}
    model = {**model_core, "manifest_hash": prereg.canonical_hash(model_core)}
    MODEL.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    core = {"protocol_version": "hvmps_12_model_freeze_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA}, "historical_market": {"path": str(prereg.MARKET), "sha256": prereg.MARKET_SHA}, "fit_labels": len(labels), "fit_last_decision": str(labels.decision_time.max()), "pretraining_outcomes_opened": True, "oos_source_incidence_opened": False, "oos_outcomes_opened": False, "gross9_rows_opened": False, "model": {"path": str(MODEL), "sha256": sha(MODEL), "manifest_hash": model["manifest_hash"]}, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "decision": "freeze_model_before_oos_incidence"}
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps(report["model"], sort_keys=True))
