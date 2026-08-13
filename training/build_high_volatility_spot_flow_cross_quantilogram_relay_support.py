"""Materialize source-only HVSFCQ-8 clocks before novelty or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_spot_flow_cross_quantilogram_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "dac611d36a71b6536c0d5d96a4e18c2bcc1106f6ee29db24ea252e931baee76f"
START = pd.Timestamp("2023-01-01T02:00:00Z")
END = pd.Timestamp("2026-08-01T02:00:00Z")
SOURCE_DIR = Path("data/high_volatility_spot_flow_cross_quantilogram_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_cross_quantilogram_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_spot_flow_cross_quantilogram_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_spot_flow_cross_quantilogram_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_flow_cross_quantilogram_relay_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = tuple(prereg.build()["diagnostic_controls"]["names"])
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "lower_cross_quantilogram", "upper_cross_quantilogram",
    "active_score", "active_score_rank", "final_hour_spot_flow",
    "final_spot_return", "final_perp_return", "full_variation", "variation_rank",
)
QUERY = """
SELECT ts,open,high,low,close{flow_columns}
FROM {table}
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def strict_prior_midrank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-270:], dtype=float)
        if math.isfinite(current) and len(prior) >= 180:
            result.at[index] = ((prior < current).sum() + 0.5 * (prior == current).sum()) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    result["ts"] = pd.to_datetime(result["ts"], utc=True)
    for column in ("open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"):
        if column not in result:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates("ts", keep=False).set_index("ts").sort_index()


def _valid(window: pd.DataFrame, flow: bool = False) -> bool:
    finite = np.isfinite(window[["open", "high", "low", "close"]]).all(axis=1)
    positive = window[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = (
        window.high.ge(window[["open", "close"]].max(axis=1))
        & window.low.le(window[["open", "close"]].min(axis=1))
        & window.high.ge(window.low)
    )
    valid = len(window) == 480 and bool((finite & positive & coherent).all())
    if flow:
        valid = valid and bool(
            np.isfinite(window[["quote_asset_volume", "taker_buy_quote"]]).all(axis=1).all()
            and window.quote_asset_volume.gt(0).all()
            and window.taker_buy_quote.ge(0).all()
            and window.taker_buy_quote.le(window.quote_asset_volume).all()
        )
    return valid


def cross_quantilogram(predictor: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(predictor, float), np.asarray(response, float)
    if len(x) != 479 or len(y) != 479 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return math.nan, math.nan
    x25, x75 = np.quantile(x, (0.25, 0.75)); y25, y75 = np.quantile(y, (0.25, 0.75))
    if not x25 < x75 or not y25 < y75:
        return math.nan, math.nan
    def score(left, right):
        denominator = float(np.sqrt(np.square(left).sum() * np.square(right).sum()))
        return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan
    lower = score((x < x25).astype(float) - .25, (y < y25).astype(float) - .25)
    upper = score((x > x75).astype(float) - .25, (y > y75).astype(float) - .25)
    return lower, upper


def block_metrics(spot: pd.DataFrame, perp: pd.DataFrame, same_minute: bool = False) -> dict[str, float | bool]:
    flow = (2 * spot.taker_buy_quote.to_numpy(float) - spot.quote_asset_volume.to_numpy(float)) / spot.quote_asset_volume.to_numpy(float)
    perp_return = np.log(perp.close.to_numpy(float) / perp.open.to_numpy(float))
    predictor = flow[:-1]
    response = perp_return[:-1] if same_minute else perp_return[1:]
    lower, upper = cross_quantilogram(predictor, response)
    final_flow = float((2 * spot.taker_buy_quote.iloc[-60:].sum() - spot.quote_asset_volume.iloc[-60:].sum()) / spot.quote_asset_volume.iloc[-60:].sum())
    final_spot = float(np.log(spot.close.iloc[-1] / spot.open.iloc[-60])); final_perp = float(np.log(perp.close.iloc[-1] / perp.open.iloc[-60]))
    variation = float(np.sqrt(np.square(perp_return).sum())); active_score = upper if final_flow > 0 else lower if final_flow < 0 else math.nan
    valid = bool(np.isfinite([lower, upper, active_score, final_flow, final_spot, final_perp, variation]).all() and variation > 0)
    confirmed = bool(valid and final_flow != 0 and np.sign(final_flow) == np.sign(final_spot) == np.sign(final_perp))
    return {"source_valid":valid,"direction_confirmed":confirmed,"lower_cross_quantilogram":lower,"upper_cross_quantilogram":upper,"same_minute_lower":cross_quantilogram(flow[:-1],perp_return[:-1])[0],"same_minute_upper":cross_quantilogram(flow[:-1],perp_return[:-1])[1],"active_score":active_score,"final_hour_spot_flow":final_flow,"final_spot_return":final_spot,"final_perp_return":final_perp,"full_variation":variation}


def build_panel(perp: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    perp, spot = _prepare(perp), _prepare(spot)
    decisions = pd.date_range(START, END, freq="8h", inclusive="left")
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        perp_window, spot_window = perp.reindex(expected), spot.reindex(expected)
        if _valid(perp_window) and _valid(spot_window, flow=True):
            metrics = block_metrics(spot_window, perp_window)
        else:
            metrics = {key:False if key in ("source_valid","direction_confirmed") else np.nan for key in ("source_valid","direction_confirmed","lower_cross_quantilogram","upper_cross_quantilogram","same_minute_lower","same_minute_upper","active_score","final_hour_spot_flow","final_spot_return","final_perp_return","full_variation")}
        rows.append({"decision_time": decision, "perp_source_rows": int(perp_window.notna().all(axis=1).sum()), "spot_source_rows": int(spot_window.notna().all(axis=1).sum()), **metrics})
    panel = pd.DataFrame(rows)
    panel["lower_score_rank"] = strict_prior_midrank(panel.lower_cross_quantilogram.where(panel.source_valid))
    panel["upper_score_rank"] = strict_prior_midrank(panel.upper_cross_quantilogram.where(panel.source_valid))
    panel["same_minute_lower_rank"] = strict_prior_midrank(panel.same_minute_lower.where(panel.source_valid))
    panel["same_minute_upper_rank"] = strict_prior_midrank(panel.same_minute_upper.where(panel.source_valid))
    panel["active_score_rank"] = np.where(panel.final_hour_spot_flow.gt(0),panel.upper_score_rank,panel.lower_score_rank)
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation.where(panel.source_valid))
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    engine = postgres_engine()
    params = {"start": (START - pd.Timedelta(hours=8)).to_pydatetime(), "end": END.to_pydatetime()}
    with engine.connect() as connection:
        perp = pd.read_sql_query(text(QUERY.format(table="bars_binance",flow_columns="")), connection, params=params)
        spot = pd.read_sql_query(text(QUERY.format(table="bars_binance_spot",flow_columns=",quote_asset_volume,taker_buy_quote")), connection, params=params)
    engine.dispose()
    panel = build_panel(perp, spot)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "hvsfcq_8_btc_source_v1",
        "queries": {"perpetual": QUERY.format(table="bars_binance",flow_columns=""), "spot": QUERY.format(table="bars_binance_spot",flow_columns=",quote_asset_volume,taker_buy_quote")},
        "tables": ["bars_binance", "bars_binance_spot"], "symbol": "BTCUSDT", "interval": "1m",
        "window": [START.isoformat(), END.isoformat()], "exact_candidate_outcomes_opened": False,
        "candidate_incidence_opened": False, "no_imputation": True,
        "output": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    frame["direction_confirmed"] = frame.direction_confirmed.astype(str).str.lower().eq("true")
    for column in ("lower_cross_quantilogram","upper_cross_quantilogram","same_minute_lower","same_minute_upper","active_score","lower_score_rank","upper_score_rank","same_minute_lower_rank","same_minute_upper_rank","active_score_rank","final_hour_spot_flow","final_spot_return","final_perp_return","full_variation","variation_rank"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    active_score = frame.active_score
    score_rank = frame.active_score_rank
    source_valid = frame.source_valid
    if control == "one_decision_stale_dependence":
        active_score, score_rank = active_score.shift(1), score_rank.shift(1)
    if control == "same_minute_cross_quantilogram":
        active_score = np.where(frame.final_hour_spot_flow.gt(0),frame.same_minute_upper,frame.same_minute_lower)
        score_rank = np.where(frame.final_hour_spot_flow.gt(0),frame.same_minute_upper_rank,frame.same_minute_lower_rank)
        active_score, score_rank = pd.Series(active_score,index=frame.index),pd.Series(score_rank,index=frame.index)
    tail = pd.Series(True, index=frame.index) if control == "no_cross_quantilogram_tail" else score_rank.ge(0.75)
    variation = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame.variation_rank.ge(0.65)
    eligible = source_valid & frame.direction_confirmed & np.isfinite(active_score) & pd.Series(active_score,index=frame.index).gt(0) & tail & variation & frame.source_valid & np.isfinite(frame.variation_rank)
    side = np.sign(frame.final_hour_spot_flow).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    active = eligible & frame.source_valid.shift(1, fill_value=False) & ~eligible.shift(1, fill_value=False)
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=8, minutes=5)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision, "entry_time": entry,
            "exit_time": exit_time, "side": int(side.at[index]),
            "lower_cross_quantilogram":float(frame.at[index,"lower_cross_quantilogram"]),"upper_cross_quantilogram":float(frame.at[index,"upper_cross_quantilogram"]),
            "active_score":float(frame.at[index,"active_score"]),"active_score_rank":float(frame.at[index,"active_score_rank"]),
            "final_hour_spot_flow":float(frame.at[index,"final_hour_spot_flow"]),
            "final_spot_return":float(frame.at[index,"final_spot_return"]),"final_perp_return":float(frame.at[index,"final_perp_return"]),
            "full_variation": float(frame.at[index, "full_variation"]),
            "variation_rank": float(frame.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSFCQ preregistration drift")
    source_manifest = materialize()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, values in support.items() for key, value in ((f"{name}_minimum_events", values["events"] >= MINIMUM[name]), (f"{name}_side_balance", values["minority_side_share"] >= 0.2), (f"{name}_month_concentration", values["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvsfcq_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "funding_values_opened":False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
