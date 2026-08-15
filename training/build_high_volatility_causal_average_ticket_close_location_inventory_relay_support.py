"""Deterministic outcome-blind source support for HVCATCLIR-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_causal_average_ticket_close_location_inventory_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "9058d54ff2cad323358b048a9a1baea6a0881b411f96a04774c02528a4e4faca"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]

OI_QUERY = """SELECT ts AS decision_time,sum_open_interest FROM open_interest_binance WHERE symbol='BTCUSDT' AND period='5m' AND ts>=:start AND ts<:end AND extract(minute FROM ts)=0 AND extract(second FROM ts)=0 AND extract(hour FROM ts) IN (0,8,16) ORDER BY ts"""
BAR_QUERY = """SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '8 hours' AS decision_time,sum(quote_asset_volume) FILTER (WHERE ts<date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '4 hours') AS first_quote_turnover,sum(number_of_trades) FILTER (WHERE ts<date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '4 hours') AS first_trade_count,sum(quote_asset_volume) FILTER (WHERE ts>=date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '4 hours') AS second_quote_turnover,sum(number_of_trades) FILTER (WHERE ts>=date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '4 hours') AS second_trade_count,(array_agg(close ORDER BY ts DESC))[1] AS final_close,max(high) AS cycle_high,min(low) AS cycle_low,sum(ln(close/open)) AS completed_return,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high) AND quote_asset_volume>=0 AND number_of_trades>0 AND number_of_trades=floor(number_of_trades)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""

ROOT = Path("data/high_volatility_causal_average_ticket_close_location_inventory_relay_sources_2020_2026")
PANEL = ROOT / "settlement_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_causal_average_ticket_close_location_inventory_relay_clocks_2020_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_causal_average_ticket_close_location_inventory_relay_split_clocks_2020_2026")
RESULT = Path("results/high_volatility_causal_average_ticket_close_location_inventory_relay_support_2026-08-16.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "sum_open_interest",
    "previous_open_interest", "oi_change", "first_half_quote_turnover", "second_half_quote_turnover",
    "first_half_trade_count", "second_half_trade_count", "first_half_average_ticket", "second_half_average_ticket", "close_location", "completed_return",
    "average_ticket_acceleration", "average_ticket_acceleration_rank", "realized_variation",
    "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "oi_change", "first_half_quote_turnover", "second_half_quote_turnover",
    "first_half_trade_count", "second_half_trade_count", "first_half_average_ticket", "second_half_average_ticket", "close_location", "completed_return",
    "average_ticket_acceleration", "average_ticket_acceleration_rank", "realized_variation", "variation_rank",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_midrank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-POLICY["history_cycles"] :], dtype=float)
        if math.isfinite(value) and len(prior) >= POLICY["minimum_history_cycles"]:
            output[index] = float(
                (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
            )
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            oi = pd.read_sql_query(
                text(OI_QUERY), connection, params={"start": START, "end": END}
            )
            bars = pd.read_sql_query(
                text(BAR_QUERY), connection, params={"start": START, "end": END}
            )
    finally:
        database.dispose()
    return oi, bars


def prepare(raw: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    oi, bars = raw
    if oi.columns.tolist() != ["decision_time", "sum_open_interest"]:
        raise RuntimeError("HVCATCLIR-8 OI schema drift")
    expected_bars = [
        "decision_time", "first_quote_turnover", "first_trade_count", "second_quote_turnover", "second_trade_count", "final_close", "cycle_high", "cycle_low", "completed_return", "minute_squared_return",
        "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    ]
    if bars.columns.tolist() != expected_bars:
        raise RuntimeError("HVCATCLIR-8 bar schema drift")

    oi = oi.copy()
    oi["decision_time"] = pd.to_datetime(oi["decision_time"], utc=True, errors="raise")
    oi["sum_open_interest"] = pd.to_numeric(oi["sum_open_interest"], errors="raise")
    if oi["decision_time"].duplicated().any() or not np.isfinite(oi["sum_open_interest"]).all():
        raise RuntimeError("HVCATCLIR-8 invalid OI rows")

    bars = bars.copy()
    for column in ("decision_time", "first_ts", "last_ts"):
        bars[column] = pd.to_datetime(bars[column], utc=True, errors="raise")
    numeric = ("first_quote_turnover", "first_trade_count", "second_quote_turnover", "second_trade_count", "final_close", "cycle_high", "cycle_low", "completed_return", "minute_squared_return", "source_rows", "distinct_rows")
    for column in numeric:
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    if bars["decision_time"].duplicated().any():
        raise RuntimeError("HVCATCLIR-8 duplicate eight-hour bar")
    start = bars["decision_time"] - pd.Timedelta("8h")
    bars["bar_valid"] = (
        np.isfinite(bars[list(numeric)]).all(axis=1)
        & bars["first_quote_turnover"].gt(0)
        & bars["first_trade_count"].gt(0)
        & bars["second_quote_turnover"].gt(0)
        & bars["second_trade_count"].gt(0)
        & bars["cycle_high"].gt(bars["cycle_low"])
        & bars["minute_squared_return"].gt(0)
        & bars["source_rows"].eq(480)
        & bars["distinct_rows"].eq(480)
        & bars["first_ts"].eq(start)
        & bars["last_ts"].eq(bars["decision_time"] - pd.Timedelta("1m"))
        & bars["coherent"].eq(True)
    )
    bars["realized_variation"] = np.sqrt(bars["minute_squared_return"])
    bars = bars.sort_values("decision_time").reset_index(drop=True)
    bars["first_half_quote_turnover"] = bars["first_quote_turnover"]
    bars["second_half_quote_turnover"] = bars["second_quote_turnover"]
    bars["first_half_trade_count"] = bars["first_trade_count"]
    bars["second_half_trade_count"] = bars["second_trade_count"]
    bars["close_location"] = np.nan
    positive_range = bars["cycle_high"].gt(bars["cycle_low"])
    bars.loc[positive_range, "close_location"] = (
        (bars.loc[positive_range, "final_close"] - bars.loc[positive_range, "cycle_low"])
        / (bars.loc[positive_range, "cycle_high"] - bars.loc[positive_range, "cycle_low"])
    )
    bars["first_half_average_ticket"] = bars["first_quote_turnover"] / bars["first_trade_count"]
    bars["second_half_average_ticket"] = bars["second_quote_turnover"] / bars["second_trade_count"]
    bars["average_ticket_acceleration"] = np.log(bars["second_half_average_ticket"] / bars["first_half_average_ticket"])
    return oi.sort_values("decision_time"), bars.set_index("decision_time").sort_index()


def build_panel(raw: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    oi, bars = prepare(raw)
    panel = oi.copy()
    panel["previous_time"] = panel["decision_time"].shift(1)
    panel["previous_open_interest"] = panel["sum_open_interest"].shift(1)
    positive_oi = panel["sum_open_interest"].gt(0) & panel["previous_open_interest"].gt(0)
    panel["oi_change"] = np.nan
    panel.loc[positive_oi, "oi_change"] = np.log(
        panel.loc[positive_oi, "sum_open_interest"]
        / panel.loc[positive_oi, "previous_open_interest"]
    )
    panel = panel.join(
        bars[["bar_valid", "first_half_quote_turnover", "second_half_quote_turnover", "first_half_trade_count", "second_half_trade_count", "first_half_average_ticket", "second_half_average_ticket", "close_location", "completed_return", "average_ticket_acceleration", "realized_variation"]],
        on="decision_time",
    )
    panel["source_valid"] = (
        panel["previous_time"].notna()
        & panel["decision_time"].sub(panel["previous_time"]).eq(pd.Timedelta(hours=8))
        & panel["bar_valid"].eq(True)
        & np.isfinite(
            panel[["sum_open_interest", "previous_open_interest", "oi_change", "first_half_quote_turnover", "second_half_quote_turnover", "first_half_trade_count", "second_half_trade_count", "first_half_average_ticket", "second_half_average_ticket", "close_location", "completed_return", "average_ticket_acceleration", "realized_variation"]]
        ).all(axis=1)
        & panel["sum_open_interest"].gt(0)
        & panel["previous_open_interest"].gt(0)
        & panel["realized_variation"].gt(0)
        & panel["first_half_trade_count"].gt(0)
        & panel["second_half_trade_count"].gt(0)
        & panel["completed_return"].ne(0)
        & panel["close_location"].ge(0)
        & panel["close_location"].le(1)
        & panel["first_half_quote_turnover"].gt(0)
        & panel["second_half_quote_turnover"].gt(0)
        & panel["first_half_average_ticket"].gt(0)
        & panel["second_half_average_ticket"].gt(0)
        & panel["average_ticket_acceleration"].gt(0)
    )
    valid = panel["source_valid"].eq(True)
    panel["average_ticket_acceleration_rank"] = causal_midrank(panel["average_ticket_acceleration"].where(valid))
    panel["variation_rank"] = causal_midrank(panel["realized_variation"].where(valid))
    panel["eligible"] = (
        valid
        & panel["oi_change"].gt(0)
        & (
            (panel["completed_return"].gt(0) & panel["close_location"].gt(1 - POLICY["close_location_outer_fraction"]))
            | (panel["completed_return"].lt(0) & panel["close_location"].lt(POLICY["close_location_outer_fraction"]))
        )
        & panel["average_ticket_acceleration_rank"].ge(POLICY["average_ticket_acceleration_rank_min"])
        & panel["variation_rank"].ge(POLICY["variation_rank_min"])
    )
    panel["feature_available_time"] = panel["decision_time"]
    return panel.loc[:, PANEL_COLUMNS]


def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next(
        (name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end),
        None,
    )


def build_clock(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for row in panel.loc[panel["eligible"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time)
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_ = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = stage_for(entry, exit_)
        if split is None:
            continue
        side = int(np.sign(row.completed_return))
        if side not in (-1, 1) or row.feature_available_time > entry:
            raise RuntimeError("HVCATCLIR-8 side or availability drift")
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "oi_change": float(row.oi_change),
            "first_half_quote_turnover": float(row.first_half_quote_turnover),
            "second_half_quote_turnover": float(row.second_half_quote_turnover),
            "first_half_trade_count": float(row.first_half_trade_count),
            "second_half_trade_count": float(row.second_half_trade_count),
            "first_half_average_ticket": float(row.first_half_average_ticket),
            "second_half_average_ticket": float(row.second_half_average_ticket),
            "close_location": float(row.close_location),
            "completed_return": float(row.completed_return),
            "average_ticket_acceleration": float(row.average_ticket_acceleration),
            "average_ticket_acceleration_rank": float(row.average_ticket_acceleration_rank),
            "realized_variation": float(row.realized_variation),
            "variation_rank": float(row.variation_rank),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum()); shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCATCLIR-8 preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    raw = load_source()
    panel = build_panel(raw)
    clock = build_clock(panel)
    split_clocks = {name: clock.loc[clock["split"].eq(name)].copy() for name in STAGES}
    common.immutable(PANEL, common.csv_gz(panel))
    common.immutable(CLOCK, common.csv_gz(clock))
    for name, frame in split_clocks.items():
        common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(frame))
    source_core = {
        "protocol_version": "hvcatclir_8_sources_v1",
        "queries": {"open_interest": OI_QUERY, "bars": BAR_QUERY},
        "query_sha256": {"open_interest": hashlib.sha256(OI_QUERY.encode()).hexdigest(), "bars": hashlib.sha256(BAR_QUERY.encode()).hexdigest()},
        "tables": ["open_interest_binance", "bars_binance"], "symbol": "BTCUSDT",
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {"open_interest": len(raw[0]), "bars": len(raw[1])},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha256_file(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "execution_prices_opened": False,
        "held_interval_funding_opened": False, "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    common.immutable(MANIFEST, common.json_bytes(manifest))
    support = {name: support_stats(clock, name) for name in STAGES}
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
    core = {
        "protocol_version": "hvcatclir_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "held_interval_funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256_file(CLOCK), "rows": len(clock)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256_file(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in split_clocks.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    common.immutable(RESULT, common.json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
