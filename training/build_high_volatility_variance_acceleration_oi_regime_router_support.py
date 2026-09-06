"""Deterministic source-only union for HVCVAROIR-8."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from training import preregister_high_volatility_variance_acceleration_oi_regime_router as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "fd6c0149039435ba1c15a780a5d8015c516089986cf6dfa6a5e9dbf81a15ccbc"
CLOCK = Path("data/high_volatility_variance_acceleration_oi_regime_router_clocks_2020_2026.csv.gz")
RESULT = Path("results/high_volatility_variance_acceleration_oi_regime_router_support_2026-08-18.json")
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
FIELDS = ("split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "oi_change")

def load_branch(name: str) -> pd.DataFrame:
    path = prereg.COMPONENTS[name]["clock"]["path"]
    frame = pd.read_csv(path, usecols=list(FIELDS))
    for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    frame["oi_change"] = pd.to_numeric(frame["oi_change"], errors="raise")
    if frame["decision_time"].duplicated().any() or not frame["side"].isin((-1, 1)).all():
        raise RuntimeError(f"invalid branch {name}")
    expected_positive = name == "expansion_continuation"
    if not (frame["oi_change"].gt(0) if expected_positive else frame["oi_change"].lt(0)).all():
        raise RuntimeError(f"OI sign drift {name}")
    return frame

def combine(branches: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in branches.items():
        item = frame.copy(); item["branch"] = name; rows.append(item)
    output = pd.concat(rows, ignore_index=True).sort_values("decision_time", kind="stable").reset_index(drop=True)
    if output["decision_time"].duplicated().any():
        raise RuntimeError("duplicate regime-router decision")
    if not output["entry_time"].eq(output["decision_time"] + pd.Timedelta("5m")).all() or not output["exit_time"].eq(output["entry_time"] + pd.Timedelta("8h")).all():
        raise RuntimeError("router timing drift")
    output.insert(0, "candidate", prereg.POLICY_ID); output.insert(1, "control", "primary")
    return output

def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame = clock.loc[clock["split"].eq(split)]; count = len(frame)
    longs = int(frame["side"].eq(1).sum()); shorts = int(frame["side"].eq(-1).sum())
    month = float(frame["entry_time"].dt.strftime("%Y-%m").value_counts().max() / count) if count else 0.0
    return {"events": count, "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / count if count else 0.0, "max_month_share": month}

def run(clock_path: Path = CLOCK, result_path: Path = RESULT) -> dict[str, Any]:
    if prereg.sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    branches = {name: load_branch(name) for name in prereg.COMPONENTS}
    clock = combine(branches); clock_path.parent.mkdir(parents=True, exist_ok=True); _write_gzip_csv(clock, clock_path)
    support = {split: stats(clock, split) for split in MINIMUM_EVENTS}; checks = {}
    for split, values in support.items():
        checks[f"{split}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[split]
        checks[f"{split}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{split}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvcvaroir_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "component_rows": {name: len(frame) for name, frame in branches.items()},
        "combined_incidence_opened": True, "combined_postentry_returns_or_pnl_opened": False,
        "funding_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(clock_path), "sha256": prereg.sha256(clock_path), "rows": len(clock)},
        "branch_counts": clock["branch"].value_counts().sort_index().to_dict(),
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}; result_path.parent.mkdir(parents=True, exist_ok=True); result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result

if __name__ == "__main__":
    report = run(); print(json.dumps({"passed": report["support_passed"], "branches": report["branch_counts"], "support": report["support"]}, indent=2))
