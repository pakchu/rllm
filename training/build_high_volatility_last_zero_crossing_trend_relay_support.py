"""Build outcome-blind source support for frozen HVLZCTR-6."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_last_zero_crossing_trend_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "93d66de2501271297ad9f27a5dfdb2324b7e6fc45ac5d95abc596d2554ee101a"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
PERP_QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_last_zero_crossing_trend_relay_sources_2023_2026")
PANEL = ROOT / "twelve_hour_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_last_zero_crossing_trend_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_last_zero_crossing_trend_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_last_zero_crossing_trend_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_last_zero_crossing_trend_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_start", "feature_available_time", "source_valid", "perp_rows", "terminal_return",
    "last_passage_fraction", "last_passage_rank", "realized_variation", "variation_rank",
    "last_passage_tail", "variation_tail", "occupation_share", "entry_side", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_start", "feature_available_time",
    "entry_time", "exit_time", "side", "terminal_return", "last_passage_fraction",
    "last_passage_rank", "realized_variation", "variation_rank", "last_passage_tail",
    "variation_tail", "occupation_share", "entry_side", "eligible",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prior_rank(series: pd.Series, continuity_valid: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    valid = continuity_valid.to_numpy(bool)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        if not valid[index]:
            continue
        prior = np.asarray(history[-P["history_blocks"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_history_blocks"]:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def last_passage(opening: float, closes: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(closes, float)
    if (
        values.shape != (P["block_minutes"],)
        or not math.isfinite(opening)
        or opening <= 0
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        return (math.nan,) * 3
    displacement = np.log(values / opening)
    terminal = float(displacement[-1])
    if terminal == 0 or not math.isfinite(terminal):
        return (math.nan,) * 3
    side = 1 if terminal > 0 else -1
    opposed = np.flatnonzero(displacement * side <= 0)
    fraction = 0.0 if len(opposed) == 0 else float((opposed[-1] + 1) / len(values))
    occupation = float(np.mean(displacement * side > 0))
    return terminal, fraction, occupation


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
            params = {"start": START.to_pydatetime(), "end": END.to_pydatetime()}
            perp = pd.read_sql_query(text(PERP_QUERY), connection, params=params)
            return perp
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVLZCTR source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVLZCTR invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1)) & source.high.ge(source.low)
    )
    return source.set_index("ts").sort_index()


def build_panel(perp_raw: pd.DataFrame) -> pd.DataFrame:
    perp = prepare(perp_raw)
    rows = []
    for source_start in pd.date_range(START, END, freq="12h", inclusive="left"):
        expected = pd.date_range(source_start, source_start + pd.Timedelta("12h"), freq="1min", inclusive="left")
        perp_block = perp.reindex(expected);perp_rows=int(perp_block.row_valid.eq(True).sum())
        valid = perp_rows == P["block_minutes"] and bool(perp_block.row_valid.eq(True).all())
        if valid:
            opening=float(perp_block.open.iloc[0]);closes=perp_block.close.to_numpy(float);returns=np.log(perp_block.close.to_numpy(float)/perp_block.open.to_numpy(float))
            terminal,last_fraction,occupation=last_passage(opening,closes);variation=math.sqrt(float(np.square(returns).sum()))
            valid=np.isfinite([terminal,last_fraction,occupation,variation]).all() and variation>0
        else:
            terminal=last_fraction=occupation=variation=math.nan
        rows.append({"source_start":source_start,"source_valid":valid,"perp_rows":perp_rows,"terminal_return":terminal,"last_passage_fraction":last_fraction,"realized_variation":variation,"occupation_share":occupation})
    panel = pd.DataFrame(rows)
    panel["feature_available_time"] = panel.source_start + pd.Timedelta("12h")
    panel["last_passage_rank"] = prior_rank(panel.last_passage_fraction.where(panel.source_valid), panel.source_valid)
    panel["variation_rank"] = prior_rank(panel.realized_variation.where(panel.source_valid), panel.source_valid)
    panel["last_passage_tail"] = panel.last_passage_rank.le(P["last_passage_rank_max"])
    panel["variation_tail"] = panel.variation_rank.ge(P["variation_rank_min"])
    panel["entry_side"] = np.sign(panel.terminal_return).fillna(0).astype(int)
    panel["eligible"] = panel.source_valid & panel.last_passage_tail & panel.variation_tail & panel.entry_side.ne(0)
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    valid = used.source_valid.eq(True)
    side = pd.to_numeric(used.entry_side, errors="coerce").fillna(0).astype(int)
    state = used.eligible.eq(True) & side.ne(0)
    if control == "no_last_passage_gate":
        state = valid & used.variation_tail & side.ne(0)
    elif control == "no_variation_gate":
        state = valid & used.last_passage_tail & side.ne(0)
    elif control == "occupation_share":
        state = valid & used.occupation_share.ge(1-P["last_passage_rank_max"]) & used.variation_tail & side.ne(0)
    elif control == "one_block_stale_state":
        state = state.shift(1, fill_value=False)
        side = side.shift(1, fill_value=0)
    elif control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=side.index, dtype=int)
    return state & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activity, side, used = active(panel, control)
    rows = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[activity]:
        decision = pd.Timestamp(panel.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=P["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=P["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "source_start": pd.Timestamp(used.at[index, "source_start"]), "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            **{column: bool(used.at[index, column]) if column in ("last_passage_tail", "variation_tail", "eligible") else float(used.at[index, column]) for column in CLOCK_COLUMNS[8:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def csv_gz(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(raw)
    return buffer.getvalue()


def immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing overwrite {path}")
    path.write_bytes(content)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVLZCTR prereg drift")
    perp_raw = load_source()
    panel = build_panel(perp_raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel)); immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items(): immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items(): immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hvlzctr_6_source_v1", "queries": {"perpetual": PERP_QUERY},
        "query_sha256": {"perpetual": hashlib.sha256(PERP_QUERY.encode()).hexdigest()},
        "window": [START.isoformat(), END.isoformat()], "physical_rows": {"perpetual": len(perp_raw)},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable(MANIFEST, json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in (
        (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
        (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
        (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
    )}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvlzctr_6_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({"passed": run()["support_passed"], "result": str(RESULT)}))
