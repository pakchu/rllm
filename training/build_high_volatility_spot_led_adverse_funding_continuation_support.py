"""Deterministic outcome-blind source support for HVSLAF-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_spot_led_adverse_funding_continuation as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "9aad58ad6eec0cac3931ab2b158c19e1e96ca5c1cafc5e9fae37ffe98dcaf5fb"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]

FUNDING_QUERY = """SELECT funding_time AS decision_time,funding_rate FROM funding_rates_binance WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"""
PERP_QUERY = """SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '8 hours' AS decision_time,sum(ln(close/open)) AS perpetual_return,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
SPOT_QUERY = """SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '8 hours' AS decision_time,sum(ln(close/open)) AS spot_return,sum(2*taker_buy_quote-quote_asset_volume) AS flow_numerator,sum(quote_asset_volume) AS quote_turnover,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high) AND quote_asset_volume>0 AND taker_buy_quote>=0 AND taker_buy_quote<=quote_asset_volume) AS coherent FROM bars_binance_spot WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""

ROOT = Path("data/high_volatility_spot_led_adverse_funding_continuation_sources_2023_2026")
PANEL = ROOT / "settlement_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_spot_led_adverse_funding_continuation_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_spot_led_adverse_funding_continuation_split_clocks_2023_2026")
RESULT = Path("results/high_volatility_spot_led_adverse_funding_continuation_support_2026-08-16.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "funding_rate",
    "perpetual_return", "spot_return", "spot_flow_share", "realized_variation",
    "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "funding_rate", "perpetual_return",
    "spot_return", "spot_flow_share", "realized_variation", "variation_rank",
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
            output[index] = float((np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior))
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            params = {"start": START, "end": END}
            funding = pd.read_sql_query(text(FUNDING_QUERY), connection, params=params)
            perpetual = pd.read_sql_query(text(PERP_QUERY), connection, params=params)
            spot = pd.read_sql_query(text(SPOT_QUERY), connection, params=params)
    finally:
        database.dispose()
    return funding, perpetual, spot


def _prepare_blocks(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    expected = (
        ["decision_time", "perpetual_return", "minute_squared_return", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent"]
        if kind == "perpetual"
        else ["decision_time", "spot_return", "flow_numerator", "quote_turnover", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent"]
    )
    if frame.columns.tolist() != expected:
        raise RuntimeError(f"HVSLAF-8 {kind} schema drift")
    result = frame.copy()
    for column in ("decision_time", "first_ts", "last_ts"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    numeric = [column for column in expected if column not in {"decision_time", "first_ts", "last_ts", "coherent"}]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    start = result["decision_time"] - pd.Timedelta("8h")
    result[f"{kind}_valid"] = (
        np.isfinite(result[numeric]).all(axis=1)
        & result["source_rows"].eq(480)
        & result["distinct_rows"].eq(480)
        & result["first_ts"].eq(start)
        & result["last_ts"].eq(result["decision_time"] - pd.Timedelta("1m"))
        & result["coherent"].eq(True)
    )
    if kind == "perpetual":
        result["realized_variation"] = np.sqrt(result["minute_squared_return"])
        return result.set_index("decision_time")[["perpetual_return", "realized_variation", "perpetual_valid"]]
    result["spot_flow_share"] = result["flow_numerator"] / result["quote_turnover"]
    return result.set_index("decision_time")[["spot_return", "spot_flow_share", "spot_valid"]]


def prepare(raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    funding, perpetual, spot = raw
    if funding.columns.tolist() != ["decision_time", "funding_rate"]:
        raise RuntimeError("HVSLAF-8 funding schema drift")
    funding = funding.copy()
    funding["decision_time"] = pd.to_datetime(funding["decision_time"], utc=True, errors="raise")
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    if funding["decision_time"].duplicated().any() or not np.isfinite(funding["funding_rate"]).all():
        raise RuntimeError("HVSLAF-8 invalid funding rows")
    return funding.sort_values("decision_time"), _prepare_blocks(perpetual, "perpetual"), _prepare_blocks(spot, "spot")


def build_panel(raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    funding, perpetual, spot = prepare(raw)
    panel = funding.join(perpetual, on="decision_time").join(spot, on="decision_time")
    finite = ["funding_rate", "perpetual_return", "spot_return", "spot_flow_share", "realized_variation"]
    panel["source_valid"] = (
        panel["perpetual_valid"].eq(True)
        & panel["spot_valid"].eq(True)
        & np.isfinite(panel[finite]).all(axis=1)
        & panel[finite[:-1]].ne(0).all(axis=1)
        & panel["realized_variation"].gt(0)
    )
    valid = panel["source_valid"].eq(True)
    panel["variation_rank"] = causal_midrank(panel["realized_variation"].where(valid))
    side = np.sign(panel["spot_return"])
    cash_leadership = (
        np.sign(panel["perpetual_return"]).eq(side)
        & np.sign(panel["spot_flow_share"]).eq(side)
        & panel["spot_return"].abs().gt(panel["perpetual_return"].abs())
    )
    adverse_funding = np.sign(panel["funding_rate"]).eq(-side)
    panel["eligible"] = valid & cash_leadership & adverse_funding & panel["variation_rank"].ge(POLICY["variation_rank_min"])
    panel["feature_available_time"] = panel["decision_time"]
    return panel.loc[:, PANEL_COLUMNS]


def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next((name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end), None)


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
        side = int(np.sign(row.spot_return))
        if side not in (-1, 1) or row.feature_available_time > entry:
            raise RuntimeError("HVSLAF-8 side or availability drift")
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "funding_rate": float(row.funding_rate), "perpetual_return": float(row.perpetual_return),
            "spot_return": float(row.spot_return), "spot_flow_share": float(row.spot_flow_share),
            "realized_variation": float(row.realized_variation), "variation_rank": float(row.variation_rank),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSLAF-8 preregistration hash drift")
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
    query_map = {"funding": FUNDING_QUERY, "perpetual": PERP_QUERY, "spot": SPOT_QUERY}
    source_core = {
        "protocol_version": "hvslaf_8_sources_v1", "queries": query_map,
        "query_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in query_map.items()},
        "tables": ["funding_rates_binance", "bars_binance", "bars_binance_spot"], "symbol": "BTCUSDT",
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {"funding": len(raw[0]), "perpetual": len(raw[1]), "spot": len(raw[2])},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha256_file(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "execution_prices_opened": False,
        "held_interval_funding_opened": False, "gross9_rows_opened": False, "no_imputation": True,
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
        "protocol_version": "hvslaf_8_source_support_v1", "policy_id": prereg.POLICY_ID,
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
