"""Build outcome-blind source support for frozen HVLVPR-24."""
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

from training import preregister_high_volatility_liquidity_volatility_premium_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "6363ae89b522e038d025c1f6705a086802f1899fc94e3149e9dff1cfa1238ea4"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close,quote_asset_volume FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_liquidity_volatility_premium_relay_sources_2023_2026")
PANEL = ROOT / "daily_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_liquidity_volatility_premium_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_liquidity_volatility_premium_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_liquidity_volatility_premium_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_liquidity_volatility_premium_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_day", "feature_available_time", "decision_time", "source_valid",
    "daily_liquidity_volatility", "log_liquidity_volatility", "daily_mean_amihud",
    "reference_count", "reference_log_liquidity_volatility_median", "innovation",
    "innovation_rank", "mean_amihud_innovation", "mean_amihud_innovation_rank",
    "realized_variation", "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "feature_available_time",
    "decision_time", "entry_time", "exit_time", "side", "daily_liquidity_volatility",
    "log_liquidity_volatility", "daily_mean_amihud", "reference_count",
    "reference_log_liquidity_volatility_median", "innovation", "innovation_rank",
    "mean_amihud_innovation", "mean_amihud_innovation_rank", "realized_variation",
    "variation_rank", "eligible",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-P["history_days"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_history_days"]:
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
            return pd.read_sql_query(
                text(QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close", "quote_asset_volume"]:
        raise RuntimeError("HVLVPR source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "quote_asset_volume"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVLVPR invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1))
        & source.high.ge(source.low)
    )
    source["row_valid"] &= np.isfinite(source.quote_asset_volume) & source.quote_asset_volume.ge(0)
    source["minute_return"] = np.log(source.close / source.open).where(source.row_valid)
    return source.set_index("ts").sort_index()


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source = prepare(raw).reindex(pd.date_range(START, END, freq="1min", inclusive="left"))
    groups = source.groupby(source.index.floor("5min"), sort=True)
    bars = pd.DataFrame({"rows": groups.row_valid.sum(), "open": groups.open.first(), "close": groups.close.last(), "quote_turnover": groups.quote_asset_volume.sum(min_count=5)})
    bars["valid"] = bars.rows.eq(5) & np.isfinite(bars[["open", "close", "quote_turnover"]]).all(axis=1) & bars.open.gt(0) & bars.close.gt(0) & bars.quote_turnover.gt(0)
    bars["return"] = np.log(bars.close / bars.open).where(bars.valid)
    bars["amihud_piece"] = (bars["return"].abs() / bars.quote_turnover).where(bars.valid)
    daily_group = bars.groupby(bars.index.floor("1d"), sort=True)
    daily = pd.DataFrame({
        "valid_groups": daily_group.valid.sum(),
        "daily_mean_amihud": daily_group.amihud_piece.mean(),
        "daily_liquidity_volatility": daily_group.amihud_piece.std(ddof=1),
        "realized_variation": np.sqrt(daily_group["return"].apply(lambda values: float(np.square(values.to_numpy(float)).sum()))),
    })
    daily["source_valid"] = daily.valid_groups.eq(P["intraday_groups"]) & np.isfinite(daily[["daily_mean_amihud", "daily_liquidity_volatility", "realized_variation"]]).all(axis=1) & daily.daily_mean_amihud.gt(0) & daily.daily_liquidity_volatility.gt(0) & daily.realized_variation.gt(0)
    daily["log_liquidity_volatility"] = np.log(daily.daily_liquidity_volatility).where(daily.source_valid)
    daily["log_mean_amihud"] = np.log(daily.daily_mean_amihud).where(daily.source_valid)
    prior_lv = daily.log_liquidity_volatility.shift(1); prior_mean = daily.log_mean_amihud.shift(1)
    daily["reference_log_liquidity_volatility_median"] = prior_lv.rolling(P["reference_days"], min_periods=P["reference_days"]).median()
    daily["reference_log_mean_amihud_median"] = prior_mean.rolling(P["reference_days"], min_periods=P["reference_days"]).median()
    daily["reference_count"] = prior_lv.notna().rolling(P["reference_days"], min_periods=P["reference_days"]).sum()
    daily["innovation"] = daily.log_liquidity_volatility - daily.reference_log_liquidity_volatility_median
    daily["mean_amihud_innovation"] = daily.log_mean_amihud - daily.reference_log_mean_amihud_median
    daily["source_valid"] &= daily.reference_count.eq(P["reference_days"]) & np.isfinite(daily[["reference_log_liquidity_volatility_median", "innovation", "mean_amihud_innovation"]]).all(axis=1) & daily.innovation.ne(0)
    panel = daily.reset_index(names="source_day"); panel["feature_available_time"] = panel.source_day + pd.Timedelta("1d")
    panel["innovation_rank"] = prior_rank(panel.innovation.where(panel.source_valid)); panel["mean_amihud_innovation_rank"] = prior_rank(panel.mean_amihud_innovation.where(panel.source_valid)); panel["variation_rank"] = prior_rank(panel.realized_variation.where(panel.source_valid))
    panel["eligible"] = panel.source_valid & (panel.innovation_rank.ge(P["innovation_rank_long_min"]) | panel.innovation_rank.le(P["innovation_rank_short_max"])) & panel.variation_rank.ge(P["variation_rank_min"])
    panel["decision_time"] = panel.feature_available_time
    return panel.loc[:, PANEL_COLUMNS]


def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    used=panel.copy(); valid=used.source_valid.eq(True); rank=pd.to_numeric(used.innovation_rank,errors="coerce"); tail=rank.ge(P["innovation_rank_long_min"])|rank.le(P["innovation_rank_short_max"]); variation=used.variation_rank.ge(P["variation_rank_min"]); side=pd.Series(np.where(rank.ge(P["innovation_rank_long_min"]),1,np.where(rank.le(P["innovation_rank_short_max"]),-1,0)),index=used.index,dtype=int); state=valid&side.ne(0)&tail&variation
    if control=="no_liquidity_volatility_tail": side=np.sign(pd.to_numeric(used.innovation,errors="coerce")).fillna(0).astype(int);state=valid&side.ne(0)&variation
    elif control=="no_variation_gate": state=valid&side.ne(0)&tail
    elif control=="daily_mean_amihud_innovation":
        raw=pd.to_numeric(used.mean_amihud_innovation_rank,errors="coerce");side=pd.Series(np.where(raw.ge(P["innovation_rank_long_min"]),1,np.where(raw.le(P["innovation_rank_short_max"]),-1,0)),index=used.index,dtype=int);state=valid&side.ne(0)&variation
    elif control=="one_day_stale_innovation": state=state.shift(1,fill_value=False);side=side.shift(1,fill_value=0)
    if control=="direction_flip":side=-side
    if control=="forced_long":side=pd.Series(1,index=side.index,dtype=int)
    return state&side.ne(0),side,used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activity, side, used = active(panel, control)
    rows = []
    for index in panel.index[activity]:
        day = pd.Timestamp(panel.at[index, "source_day"])
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta("5min")
        exit_time = entry + pd.Timedelta("24h")
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "source_day": day,
                "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{
                    column: bool(used.at[index, column]) if column == "eligible" else float(used.at[index, column])
                    for column in CLOCK_COLUMNS[9:]
                },
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


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
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVLVPR prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel))
    immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items():
        immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items():
        immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hvlvpr_24_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable(MANIFEST, json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: passed
        for name, values in support.items()
        for key, passed in (
            (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvlvpr_24_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({"passed": run()["support_passed"], "result": str(RESULT)}))
