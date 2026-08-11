"""Build source-only HVCOM-355M clocks before Gross9 or economic outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_new_york_opening_momentum_relay as prereg
PREREG_SHA = "45f5aba85e0b8eff2dfa119cff1b5f39e26be3b89a5b5f899b3c7e24b01d3ea4"
ENV_FILE = "/home/pakchu/rllm/.env"
NY = "America/New_York"
START = pd.Timestamp("2023-01-01T00:00:00Z"); FINAL_END = pd.Timestamp("2026-08-01T00:00:00Z"); END = pd.Timestamp("2026-08-02T00:00:00Z")
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
STATE = Path("data/high_volatility_new_york_opening_momentum_relay_sources_2023_2026/weekday_states.csv.gz")
CLOCK = Path("data/high_volatility_new_york_opening_momentum_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_new_york_opening_momentum_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_new_york_opening_momentum_relay_support_2026-08-11.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), FINAL_END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = tuple(prereg.build()["diagnostic_controls"]["names"])
COLUMNS = ("candidate", "control", "split", "anchor_time", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "reaction_return", "first_fifteen_return", "pre_anchor_variation", "variation_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weekday_anchors(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    left = start.tz_convert(NY).normalize()
    right = end.tz_convert(NY).normalize() + pd.Timedelta(days=1)
    days = pd.date_range(left, right, freq="D", inclusive="left")
    anchors = pd.DatetimeIndex(days[days.weekday < 5] + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    return anchors[(anchors >= start) & (anchors < end)]


def _valid(window: pd.DataFrame) -> bool:
    values = window[["open", "high", "low", "close"]]
    return bool(np.isfinite(values).all(axis=1).all() and values.gt(0).all(axis=1).all() and window.high.ge(window[["open", "close"]].max(axis=1)).all() and window.low.le(window[["open", "close"]].min(axis=1)).all() and window.high.ge(window.low).all())


def score_states(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy(); frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame = frame.sort_values("date").set_index("date")
    anchors = weekday_anchors(frame.index.min() + pd.Timedelta(days=1), frame.index.max() + pd.Timedelta(days=1))
    rows, prior = [], []
    for anchor in anchors:
        variation_index = pd.date_range(anchor - pd.Timedelta(hours=24), anchor, freq="5min", inclusive="left")
        reaction_index = pd.date_range(anchor, anchor + pd.Timedelta(minutes=30), freq="5min", inclusive="left")
        variation_window, reaction = frame.reindex(variation_index), frame.reindex(reaction_index)
        if len(variation_window) != 288 or len(reaction) != 6 or not _valid(variation_window) or not _valid(reaction):
            continue
        closes = variation_window.close.to_numpy(dtype=float)
        variation = float(np.sqrt(np.square(np.log(variation_window.close.to_numpy(float)/variation_window.open.to_numpy(float))).sum()))
        reaction_return = float(np.log(float(reaction.close.iloc[-1]) / float(reaction.open.iloc[0])))
        first_fifteen_return = float(np.log(float(reaction.close.iloc[2]) / float(reaction.open.iloc[0])))
        history = np.asarray(prior[-180:], dtype=float)
        rank = float(((history < variation).sum() + .5 * (history == variation).sum()) / len(history)) if len(history) >= 120 else np.nan
        rows.append({"anchor_time": anchor, "decision_time": anchor + pd.Timedelta(minutes=30), "reaction_return": reaction_return, "first_fifteen_return": first_fifteen_return, "pre_anchor_variation": variation, "variation_rank": rank})
        prior.append(variation)
    return pd.DataFrame(rows)


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    rows = []
    for i, row in states.iterrows():
        reaction = float(row.reaction_return)
        decision = pd.Timestamp(row.decision_time)
        ranked = bool(np.isfinite(row.variation_rank) and row.variation_rank >= .65)
        if control == "first_fifteen_minute_direction":
            reaction = float(row.first_fifteen_return)
        elif control == "one_weekday_stale_reaction":
            if i == 0:
                continue
            reaction = float(states.iloc[i - 1].reaction_return)
        if not np.isfinite(reaction) or reaction == 0 or (not ranked and control != "no_variation_gate"):
            continue
        side = int(np.sign(reaction))
        if control == "direction_flip":side = -side
        elif control == "forced_long":
            side = 1
        local_day=pd.Timestamp(row.anchor_time).tz_convert(NY).normalize();entry=(local_day+pd.Timedelta(hours=10,minutes=5)).tz_convert("UTC");exit_=(local_day+pd.Timedelta(hours=16)).tz_convert("UTC")
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "anchor_time": row.anchor_time, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_, "side": side, "reaction_return": reaction, "first_fifteen_return":float(row.first_fifteen_return), "pre_anchor_variation": float(row.pre_anchor_variation), "variation_rank": float(row.variation_rank)})
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = frame[frame.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0., "max_month_share": 0.}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file,postgres_url_from_env
    load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})

def load_market():
    from sqlalchemy import text
    db=postgres_engine()
    try:
        with db.connect() as c:raw=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
    finally:db.dispose()
    raw["ts"]=pd.to_datetime(raw.ts,utc=True);g=raw.set_index("ts").groupby(raw.ts.dt.floor("5min").to_numpy());market=pd.DataFrame({"date":g.open.first().index,"open":g.open.first().to_numpy(),"high":g.high.max().to_numpy(),"low":g.low.min().to_numpy(),"close":g.close.last().to_numpy(),"rows":g.close.count().to_numpy()});market=market[market.rows.eq(5)].drop(columns="rows");return market,raw

def csv_gz(frame):
    b=io.BytesIO();raw=frame.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
    with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
    return b.getvalue()

def immutable(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=content:raise RuntimeError(f"refusing overwrite {path}")
    path.write_bytes(content)

def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCOM prereg drift")
    market,raw = load_market(); states = score_states(market)
    primary = build_clock(states); controls = {name: build_clock(states, name) for name in CONTROLS}
    STATE.parent.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    immutable(STATE,csv_gz(states)); immutable(CLOCK,csv_gz(primary))
    for name, frame in controls.items():
        immutable(CONTROL_DIR / f"{name}.csv.gz",csv_gz(frame))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, item in support.items() for key, value in ((f"{name}_minimum_events", item["events"] >= MINIMUM[name]), (f"{name}_side_balance", item["minority_side_share"] >= .2), (f"{name}_month_concentration", item["max_month_share"] <= .45))}
    passed = all(checks.values()); registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvcom_355m_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "calendar_audit": {"timezone": NY, "weekday": "Monday-Friday", "local_anchor": "09:30", "dst_rule": "IANA zoneinfo via pandas", "exchange_holiday_filter":False}, "source": {"query":QUERY,"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"physical_rows":len(raw),"outcomes_opened":False}, "source_state": {"path": str(STATE), "sha256": sha(STATE), "rows": len(states)}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    report = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
