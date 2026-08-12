"""Outcome-blind source support for frozen HVCAFPT-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_cross_alt_flow_price_transmission_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "5bbd1d6729474b4c2a6f7167557b2eebb169eb422740ac351603767379c2dd8c"
REG = prereg.build()
P = REG["policy"]
SPLITS = {key: tuple(map(pd.Timestamp, value)) for key, value in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
SYMBOLS = ("BTCUSDT", *prereg.ALTS)
QUERY = """WITH tagged AS (
 SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 02:00:00+00') AS block_start,
        ts,symbol,open,high,low,close,quote_asset_volume,taker_buy_quote
 FROM bars_binance
 WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end
)
SELECT block_start,symbol,
 sum(quote_asset_volume) FILTER (WHERE ts<block_start+INTERVAL '4 hours') AS first_quote,
 sum(taker_buy_quote) FILTER (WHERE ts<block_start+INTERVAL '4 hours') AS first_buy_quote,
 (array_agg(open ORDER BY ts) FILTER (WHERE ts>=block_start+INTERVAL '4 hours'))[1] AS second_open,
 (array_agg(close ORDER BY ts DESC) FILTER (WHERE ts>=block_start+INTERVAL '4 hours'))[1] AS second_close,
 sum(power(ln(close/open),2)) AS minute_squared_return,
 count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,
 bool_and(open>0 AND high>0 AND low>0 AND close>0 AND quote_asset_volume>=0 AND
          taker_buy_quote>=0 AND taker_buy_quote<=quote_asset_volume AND
          high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent
FROM tagged GROUP BY 1,2 ORDER BY 1,2"""
ROOT = Path("data/high_volatility_cross_alt_flow_price_transmission_relay_sources_2023_2026")
PANEL = ROOT / "block_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_cross_alt_flow_price_transmission_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_cross_alt_flow_price_transmission_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_cross_alt_flow_price_transmission_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_cross_alt_flow_price_transmission_relay_support_2026-08-13.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "side", "breadth",
    "transmission_score", "transmission_rank", "btc_realized_variation", "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time",
    "exit_time", "side", "breadth", "transmission_score", "transmission_rank",
    "btc_realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    return prereg.canonical_hash(value)


def causal(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-P["history_decisions"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_history_decisions"]:
            result[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(result, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    db = postgres_engine()
    try:
        with db.connect() as connection:
            return pd.read_sql_query(
                text(QUERY), connection,
                params={"symbols": list(SYMBOLS), "start": START, "end": END},
            )
    finally:
        db.dispose()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "block_start", "symbol", "first_quote", "first_buy_quote", "second_open", "second_close",
        "minute_squared_return", "source_rows", "distinct_rows", "first_ts", "last_ts", "coherent",
    ]
    if raw.columns.tolist() != required:
        raise RuntimeError("HVCAFPT source schema drift")
    value = raw.copy()
    for column in ("block_start", "first_ts", "last_ts"):
        value[column] = pd.to_datetime(value[column], utc=True, errors="coerce")
    numeric = required[2:9]
    for column in numeric:
        value[column] = pd.to_numeric(value[column], errors="coerce")
    if value[["block_start", "symbol"]].isna().any().any() or value.duplicated(["block_start", "symbol"]).any():
        raise RuntimeError("HVCAFPT invalid source key")
    finite = np.isfinite(value[numeric]).all(axis=1)
    value["row_valid"] = (
        finite & value.first_quote.gt(0) & value.second_open.gt(0) & value.second_close.gt(0)
        & value.minute_squared_return.gt(0) & value.source_rows.eq(480) & value.distinct_rows.eq(480)
        & value.first_ts.eq(value.block_start) & value.last_ts.eq(value.block_start + pd.Timedelta("479m"))
        & value.coherent.eq(True)
    )
    value["first_flow"] = (2 * value.first_buy_quote - value.first_quote) / value.first_quote
    value["second_return"] = np.log(value.second_close / value.second_open)
    value["decision_time"] = value.block_start + pd.Timedelta("8h")
    return value.set_index(["decision_time", "symbol"]).sort_index()


def transmission(flow: np.ndarray, returns: np.ndarray) -> tuple[int, int, float]:
    flow_sign = np.sign(flow)
    return_sign = np.sign(returns)
    transmitting = np.isfinite(flow) & np.isfinite(returns) & (flow_sign != 0) & (flow_sign == return_sign)
    positive = int(np.sum(transmitting & (return_sign > 0)))
    negative = int(np.sum(transmitting & (return_sign < 0)))
    if max(positive, negative) < P["minimum_directional_breadth"] or positive == negative:
        return 0, max(positive, negative), math.nan
    side = 1 if positive > negative else -1
    chosen = transmitting & (return_sign == side)
    score = float(np.sum(np.abs(flow[chosen]) * np.abs(returns[chosen])))
    return side, int(np.sum(chosen)), score


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    value = prepare(raw)
    decisions = pd.date_range(START + pd.Timedelta("2h"), END, freq="8h", inclusive="left")
    full = value.reindex(pd.MultiIndex.from_product([decisions, SYMBOLS], names=["decision_time", "symbol"]))
    valid = full.row_valid.unstack("symbol").reindex(columns=SYMBOLS)
    flows = full.first_flow.unstack("symbol").reindex(columns=SYMBOLS)
    returns = full.second_return.unstack("symbol").reindex(columns=SYMBOLS)
    squared = full.minute_squared_return.unstack("symbol").reindex(columns=SYMBOLS)
    btc_variation = np.sqrt(squared.BTCUSDT.where(valid.BTCUSDT).rolling(3, min_periods=3).sum())
    rows = []
    for decision in decisions:
        ok = bool(valid.loc[decision].eq(True).all() and math.isfinite(float(btc_variation.loc[decision])))
        side, breadth, score = transmission(
            flows.loc[decision, list(prereg.ALTS)].to_numpy(float),
            returns.loc[decision, list(prereg.ALTS)].to_numpy(float),
        ) if ok else (0, 0, math.nan)
        rows.append({
            "decision_time": decision, "feature_available_time": decision, "source_valid": ok,
            "side": side, "breadth": breadth, "transmission_score": score,
            "btc_realized_variation": float(btc_variation.loc[decision]) if ok else math.nan,
        })
    panel = pd.DataFrame(rows)
    panel["transmission_rank"] = causal(panel.transmission_score.where(panel.source_valid & panel.breadth.gt(0)))
    panel["variation_rank"] = causal(panel.btc_realized_variation.where(panel.source_valid))
    panel["eligible"] = (
        panel.source_valid & panel.breadth.ge(P["minimum_directional_breadth"])
        & panel.transmission_rank.ge(P["transmission_rank_min"])
        & panel.variation_rank.ge(P["variation_rank_min"])
    )
    return panel.loc[:, PANEL_COLUMNS]


def onset(state: pd.Series, valid: pd.Series) -> pd.Series:
    result = pd.Series(False, index=state.index)
    prior = None
    for index in state.index:
        if not bool(valid.at[index]):
            continue
        if bool(state.at[index]) and prior is not None:
            result.at[index] = not bool(state.at[prior])
        prior = index
    return result


def active(panel: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = panel.copy()
    side = used.side.copy()
    score = used.transmission_score.copy()
    rank = used.transmission_rank.copy()
    if control == "contemporaneous_flow_return":
        return pd.Series(False, index=used.index), side, used
    if control == "one_block_stale_transmission":
        side, score, rank = side.shift(1), score.shift(1), rank.shift(1)
    valid = used.source_valid & side.ne(0) & np.isfinite(score)
    state = valid & rank.ge(P["transmission_rank_min"]) & used.variation_rank.ge(P["variation_rank_min"])
    if control == "no_transmission_tail":
        state = valid & used.variation_rank.ge(P["variation_rank_min"])
    elif control == "no_variation_gate":
        state = valid & rank.ge(P["transmission_rank_min"])
    selected = onset(state, used.source_valid)
    side = pd.to_numeric(side, errors="coerce").fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return selected & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    selected, side, used = active(panel, control)
    rows = []
    reserved = None
    for index in panel.index[selected]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=P["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=P["hold_hours"])
        if reserved is not None and entry < reserved:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": used.at[index, "feature_available_time"],
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "breadth": float(used.at[index, "breadth"]),
            "transmission_score": float(used.at[index, "transmission_score"]),
            "transmission_rank": float(used.at[index, "transmission_rank"]),
            "btc_realized_variation": float(used.at[index, "btc_realized_variation"]),
            "variation_rank": float(used.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    value = clock[clock.split.eq(split)]
    if value.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(value.side.eq(1).sum())
    shorts = int(value.side.eq(-1).sum())
    months = pd.to_datetime(value.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(value), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(value),
        "max_month_share": int(months.max()) / len(value),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCAFPT prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    common.immutable(PANEL, common.csv_gz(panel))
    common.immutable(CLOCK, common.csv_gz(primary))
    for name, value in controls.items():
        common.immutable(CONTROL_DIR / f"{name}.csv.gz", common.csv_gz(value))
    for name, value in splits.items():
        common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(value))
    source_core = {
        "protocol_version": "hvcafpt_8_sources_v1", "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "table": "bars_binance",
        "symbols": list(SYMBOLS), "window": [START.isoformat(), END.isoformat()],
        "physical_rows": len(raw), "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": chash(source_core)}
    common.immutable(MANIFEST, common.json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for name, summary in support.items()
        for key, value in (
            (f"{name}_minimum_events", summary["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", summary["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", summary["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvcafpt_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False,
        "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(value)} for name, value in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    common.immutable(RESULT, common.json_bytes(result))
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
