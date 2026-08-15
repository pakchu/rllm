"""Deterministic source-only support for HVTVDAC-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_temporal_variance_decay_acceptance_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "6e1cd4a837cd50691f35f5cdc9d1588d928b42301ec07fd3abc2a089a8956e60"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]
ROOT = Path("data/high_volatility_temporal_variance_decay_acceptance_continuation_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_temporal_variance_decay_acceptance_continuation_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_temporal_variance_decay_acceptance_continuation_split_clocks_2023_2026")
RESULT = Path("results/high_volatility_temporal_variance_decay_acceptance_continuation_support_2026-08-16.json")
QUERY = """SELECT ts,open,high,low,close FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-POLICY["history_cycles"] :], dtype=float)
        if math.isfinite(value) and len(prior) >= POLICY["minimum_history_cycles"]:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            frame = pd.read_sql_query(text(QUERY), connection, params={"start": START, "end": END})
    finally:
        database.dispose()
    return frame


def build_panel(frame: pd.DataFrame) -> pd.DataFrame:
    expected = ["ts", "open", "high", "low", "close"]
    if frame.columns.tolist() != expected:
        raise RuntimeError("HVTVDAC-8 source schema drift")
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    for column in expected[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["ts"].duplicated().any():
        raise RuntimeError("HVTVDAC-8 duplicate timestamps")
    frame = frame.set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left"):
        index = pd.date_range(decision - pd.Timedelta("8h"), decision, freq="1min", inclusive="left")
        window = frame.reindex(index)
        valid = (
            len(window) == 480
            and np.isfinite(window).all().all()
            and window[["open", "high", "low", "close"]].gt(0).all().all()
            and window["high"].ge(window[["open", "close"]].max(axis=1)).all()
            and window["low"].le(window[["open", "close"]].min(axis=1)).all()
            and window["high"].ge(window["low"]).all()
        )
        if valid:
            returns = np.log(window["close"].to_numpy(float) / window["open"].to_numpy(float))
            first, second = returns[:240], returns[240:]
            first_return, second_return = float(first.sum()), float(second.sum())
            first_qv, second_qv = float(np.square(first).sum()), float(np.square(second).sum())
            first_efficiency = abs(first_return) / math.sqrt(first_qv) if first_qv > 0 else math.nan
            second_efficiency = abs(second_return) / math.sqrt(second_qv) if second_qv > 0 else math.nan
            variation = math.sqrt(first_qv + second_qv) if first_qv + second_qv > 0 else math.nan
            valid = all(math.isfinite(value) for value in (first_return, second_return, first_qv, second_qv, first_efficiency, second_efficiency, variation)) and first_qv > 0 and second_qv > 0
        if not valid:
            first_return = second_return = first_qv = second_qv = first_efficiency = second_efficiency = variation = math.nan
        rows.append({
            "decision_time": decision,
            "source_valid": bool(valid),
            "first_half_return": first_return,
            "second_half_return": second_return,
            "first_half_qv": first_qv,
            "second_half_qv": second_qv,
            "first_half_efficiency": first_efficiency,
            "second_half_efficiency": second_efficiency,
            "realized_variation": variation,
        })
    panel = pd.DataFrame(rows)
    panel["variation_rank"] = prior_rank(panel["realized_variation"].where(panel["source_valid"]))
    agreement = panel["first_half_return"].ne(0) & panel["second_half_return"].ne(0) & np.sign(panel["first_half_return"]).eq(np.sign(panel["second_half_return"]))
    panel["eligible"] = panel["source_valid"] & agreement & panel["second_half_qv"].lt(panel["first_half_qv"]) & panel["second_half_efficiency"].gt(panel["first_half_efficiency"]) & panel["variation_rank"].ge(POLICY["variation_rank_min"])
    panel["onset"] = panel["eligible"] & ~panel["eligible"].shift(1, fill_value=False) & panel["source_valid"].shift(1, fill_value=False)
    panel["feature_available_time"] = panel["decision_time"]
    return panel


def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next((name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end), None)


def build_clock(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for row in panel.loc[panel["onset"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time)
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_ = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = stage_for(entry, exit_)
        if split is None:
            continue
        side = int(np.sign(row.second_half_return))
        if side not in (-1, 1):
            raise RuntimeError("HVTVDAC-8 side drift")
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "first_half_return": row.first_half_return, "second_half_return": row.second_half_return,
            "first_half_qv": row.first_half_qv, "second_half_qv": row.second_half_qv,
            "first_half_efficiency": row.first_half_efficiency,
            "second_half_efficiency": row.second_half_efficiency,
            "realized_variation": row.realized_variation, "variation_rank": row.variation_rank,
        })
    columns = ["candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "first_half_return", "second_half_return", "first_half_qv", "second_half_qv", "first_half_efficiency", "second_half_efficiency", "realized_variation", "variation_rank"]
    return pd.DataFrame(rows, columns=columns)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset["side"].eq(1).sum()), int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVTVDAC-8 preregistration hash drift")
    prereg.validate(json.loads(prereg.DEFAULT_OUTPUT.read_text()))
    raw = load_source()
    panel = build_panel(raw)
    clock = build_clock(panel)
    ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(clock, CLOCK)
    split_clocks = {name: clock.loc[clock["split"].eq(name)].copy() for name in STAGES}
    for name, value in split_clocks.items():
        _write_gzip_csv(value, SPLIT_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvtvdac_8_sources_v1", "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "tables": ["bars_binance"],
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "panel": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "execution_prices_opened": False, "funding_opened": False,
        "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    support = {name: support_stats(clock, name) for name in STAGES}
    checks = {key: passed for name, values in support.items() for key, passed in ((f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]), (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]), (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]))}
    passed = all(checks.values())
    core = {
        "protocol_version": "hvtvdac_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": REGISTRATION["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "held_interval_funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(clock)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(value)} for name, value in split_clocks.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
