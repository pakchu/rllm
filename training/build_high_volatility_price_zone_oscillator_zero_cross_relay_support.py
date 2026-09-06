"""Build outcome-blind source support for frozen HVPZO-6."""
from __future__ import annotations

import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_price_zone_oscillator_zero_cross_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-06-20T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "b093498375e36a2b91b375a1cccefff4c4d9fedd47cb3df65f155ca2b76daac4"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, bounds)) for name, bounds in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_price_zone_oscillator_zero_cross_relay_sources_2023_2026")
PANEL = ROOT / "quarter_hour_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_price_zone_oscillator_zero_cross_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_price_zone_oscillator_zero_cross_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_price_zone_oscillator_zero_cross_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_price_zone_oscillator_zero_cross_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = ("decision_time", "feature_available_time", "source_valid", "bar_rows", "bar_open", "bar_high", "bar_low", "bar_close", "signed_close", "signed_close_ema", "close_ema", "pzo", "pzo_cross", "cross_side", "trailing_variation", "variation_rank")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "pzo", "trailing_variation", "variation_rank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-P["variation_history_decisions"]:], dtype=float)
        if math.isfinite(value) and len(prior) >= P["minimum_variation_history_decisions"]:
            result[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(result, index=series.index)


def seeded_ema(values: pd.Series, valid: pd.Series, periods: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    flags = valid.to_numpy(bool)
    result = np.full(len(numeric), np.nan)
    alpha = 2.0 / (periods + 1.0)
    seed: list[float] = []
    previous = math.nan
    for index, (value, okay) in enumerate(zip(numeric, flags)):
        if not okay or not math.isfinite(value):
            seed = []
            previous = math.nan
            continue
        if not math.isfinite(previous):
            seed.append(float(value))
            if len(seed) == periods:
                previous = float(np.mean(seed))
                result[index] = previous
            continue
        previous = alpha * float(value) + (1.0 - alpha) * previous
        result[index] = previous
    return pd.Series(result, index=values.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        engine.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVPZO source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVPZO invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1) & source.high.ge(prices[["open", "close"]].max(axis=1)) & source.low.le(prices[["open", "close"]].min(axis=1)) & source.high.ge(source.low)
    return source.set_index("ts").sort_index()


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    minute_index = pd.date_range(START, END, freq="1min", inclusive="left")
    source = prepare(raw).reindex(minute_index)
    valid = source.row_valid.eq(True)
    source["minute_sq"] = np.square(np.log(source.close.astype(float) / source.open.astype(float))).where(valid)
    groups = source.resample("15min", label="left", closed="left")
    bars = pd.DataFrame({
        "bar_rows": groups.row_valid.sum(),
        "bar_open": groups.open.first(),
        "bar_high": groups.high.max(),
        "bar_low": groups.low.min(),
        "bar_close": groups.close.last(),
    })
    bars["bar_valid"] = bars.bar_rows.eq(15) & np.isfinite(bars[["bar_open", "bar_high", "bar_low", "bar_close"]]).all(axis=1) & bars[["bar_open", "bar_high", "bar_low", "bar_close"]].gt(0).all(axis=1)
    direction = np.sign(bars.bar_close - bars.bar_close.shift(1))
    contiguous = bars.bar_valid & bars.bar_valid.shift(1, fill_value=False)
    bars["signed_close"] = (direction * bars.bar_close).where(contiguous)
    bars["signed_close_ema"] = seeded_ema(bars.signed_close, contiguous, P["ema_periods"])
    bars["close_ema"] = seeded_ema(bars.bar_close, contiguous, P["ema_periods"])
    bars["pzo"] = 100.0 * bars.signed_close_ema / bars.close_ema
    previous = bars.pzo.shift(1)
    bars["cross_side"] = np.select([(bars.pzo > 0) & (previous <= 0), (bars.pzo < 0) & (previous >= 0)], [1, -1], default=0).astype(int)
    bars["pzo_cross"] = bars.cross_side.ne(0) & np.isfinite(previous)
    trailing_rows = valid.astype(int).rolling(1440, min_periods=1440).sum()
    trailing_sq = source.minute_sq.rolling(1440, min_periods=1440).sum()
    decisions = bars.index + pd.Timedelta("15min")
    panel = bars.reset_index(drop=True)
    panel["decision_time"] = decisions
    panel["feature_available_time"] = decisions
    lookup = decisions - pd.Timedelta("1min")
    panel["trailing_variation"] = np.sqrt(trailing_sq.reindex(lookup).to_numpy(float))
    complete_variation = trailing_rows.reindex(lookup).to_numpy(float) == 1440
    panel["source_valid"] = bars.bar_valid.to_numpy(bool) & np.isfinite(panel[["pzo", "trailing_variation"]]).all(axis=1) & complete_variation & panel.trailing_variation.gt(0)
    panel["variation_rank"] = prior_rank(panel.trailing_variation.where(panel.source_valid))
    panel = panel[(panel.decision_time >= START + pd.Timedelta("24h")) & (panel.decision_time < END)].reset_index(drop=True)
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    valid = panel.source_valid.eq(True)
    side = panel.cross_side.astype(int)
    event = panel.pzo_cross.eq(True)
    variation = panel.variation_rank.ge(P["variation_rank_min"])
    if control == "no_variation_gate":
        selected = valid & event
    elif control == "unsmoothed_signed_close_cross":
        raw_side = np.sign(panel.signed_close).fillna(0).astype(int)
        event = raw_side.ne(0) & raw_side.ne(raw_side.shift(1))
        side = raw_side
        selected = valid & event & variation
    elif control == "one_bar_stale_cross":
        side = side.shift(1, fill_value=0)
        selected = valid & event.shift(1, fill_value=False) & variation
    else:
        selected = valid & event & variation
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=panel.index, dtype=int)
    return selected & side.ne(0), side


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    selected, side = active(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[selected]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=P["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=P["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({"candidate": prereg.POLICY_ID, "control": control, "split": split, "decision_time": decision, "feature_available_time": panel.at[index, "feature_available_time"], "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "pzo": float(panel.at[index, "pzo"]), "trailing_variation": float(panel.at[index, "trailing_variation"]), "variation_rank": float(panel.at[index, "variation_rank"])})
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
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(raw)
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
        raise RuntimeError("HVPZO preregistration drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel))
    immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items(): immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items(): immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {"protocol_version": "hvpzo_6_source_v1", "query": QUERY, "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw), "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())}, "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True}
    manifest = {**source_core, "manifest_hash": chash(source_core)}
    immutable(MANIFEST, json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in ((f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]), (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]), (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]))}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "hvpzo_6_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "candidate_incidence_opened": True, "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": chash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
