"""Materialize source-only HVSCTBA-8 clocks before novelty or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_spot_conditioned_transmission_beta_asymmetry_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "b961c8abd03bb01870ebae82c4882ff93b48049a8fddddb12742288d6f4410dc"
START = pd.Timestamp("2023-03-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_transmission_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_conditioned_transmission_beta_asymmetry_relay_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_asymmetry_tail", "no_variation_gate", "no_onset",
    "one_block_stale_asymmetry", "direction_flip", "same_clock_forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "up_spot_beta", "down_spot_beta",
    "transmission_asymmetry", "asymmetry_rank", "positive_spot_minutes",
    "negative_spot_minutes", "full_variation", "variation_rank",
)
QUERY = """
SELECT ts,open,high,low,close
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
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates("ts", keep=False).set_index("ts").sort_index()


def _valid(window: pd.DataFrame) -> bool:
    finite = np.isfinite(window[["open", "high", "low", "close"]]).all(axis=1)
    positive = window[["open", "high", "low", "close"]].gt(0).all(axis=1)
    coherent = (
        window.high.ge(window[["open", "close"]].max(axis=1))
        & window.low.le(window[["open", "close"]].min(axis=1))
        & window.high.ge(window.low)
    )
    return len(window) == 481 and bool((finite & positive & coherent).all())


def transmission_metrics(spot_close: np.ndarray, perp_close: np.ndarray) -> dict[str, float | int | bool]:
    spot_return = np.diff(np.log(np.asarray(spot_close, dtype=float)))
    perp_return = np.diff(np.log(np.asarray(perp_close, dtype=float)))
    up = spot_return > 0.0
    down = spot_return < 0.0
    up_count, down_count = int(up.sum()), int(down.sum())
    up_den = float(np.square(spot_return[up]).sum())
    down_den = float(np.square(spot_return[down]).sum())
    up_beta = float(np.dot(spot_return[up], perp_return[up]) / up_den) if up_den > 0 else float("nan")
    down_beta = float(np.dot(spot_return[down], perp_return[down]) / down_den) if down_den > 0 else float("nan")
    asymmetry = float(np.log(down_beta / up_beta)) if up_beta > 0 and down_beta > 0 else float("nan")
    variation = float(np.sqrt(np.square(perp_return).sum()))
    valid = bool(
        up_count >= 120 and down_count >= 120
        and np.isfinite([up_beta, down_beta, asymmetry, variation]).all()
        and up_beta > 0 and down_beta > 0 and asymmetry != 0.0 and variation > 0
    )
    return {
        "source_valid": valid, "up_spot_beta": up_beta, "down_spot_beta": down_beta,
        "transmission_asymmetry": asymmetry, "positive_spot_minutes": up_count,
        "negative_spot_minutes": down_count, "full_variation": variation,
    }


def build_panel(perp: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    perp, spot = _prepare(perp), _prepare(spot)
    decisions = pd.date_range(START, END, freq="8h", inclusive="left")
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=8, minutes=1), decision, freq="1min", inclusive="left")
        perp_window, spot_window = perp.reindex(expected), spot.reindex(expected)
        if _valid(perp_window) and _valid(spot_window):
            metrics = transmission_metrics(spot_window.close.to_numpy(), perp_window.close.to_numpy())
        else:
            metrics = {
                "source_valid": False, "up_spot_beta": np.nan, "down_spot_beta": np.nan,
                "transmission_asymmetry": np.nan, "positive_spot_minutes": 0,
                "negative_spot_minutes": 0, "full_variation": np.nan,
            }
        rows.append({"decision_time": decision, "perp_source_rows": int(perp_window.notna().all(axis=1).sum()), "spot_source_rows": int(spot_window.notna().all(axis=1).sum()), **metrics})
    panel = pd.DataFrame(rows)
    panel["asymmetry_rank"] = strict_prior_midrank(panel.transmission_asymmetry.abs().where(panel.source_valid))
    panel["variation_rank"] = strict_prior_midrank(panel.full_variation.where(panel.source_valid))
    return panel


def materialize() -> dict[str, Any]:
    from sqlalchemy import text

    engine = postgres_engine()
    params = {"start": (START - pd.Timedelta(hours=8, minutes=1)).to_pydatetime(), "end": END.to_pydatetime()}
    with engine.connect() as connection:
        perp = pd.read_sql_query(text(QUERY.format(table="bars_binance")), connection, params=params)
        spot = pd.read_sql_query(text(QUERY.format(table="bars_binance_spot")), connection, params=params)
    engine.dispose()
    panel = build_panel(perp, spot)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "hvsctba_8_btc_source_v1",
        "queries": {"perpetual": QUERY.format(table="bars_binance"), "spot": QUERY.format(table="bars_binance_spot")},
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
    for column in ("up_spot_beta", "down_spot_beta", "transmission_asymmetry", "asymmetry_rank", "positive_spot_minutes", "negative_spot_minutes", "full_variation", "variation_rank"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    asymmetry = frame.transmission_asymmetry
    asymmetry_rank = frame.asymmetry_rank
    source_valid = frame.source_valid
    if control == "one_block_stale_asymmetry":
        asymmetry, asymmetry_rank, source_valid = asymmetry.shift(1), asymmetry_rank.shift(1), source_valid.shift(1, fill_value=False)
    tail = pd.Series(True, index=frame.index) if control == "no_asymmetry_tail" else asymmetry_rank.ge(0.80)
    variation = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame.variation_rank.ge(0.65)
    eligible = source_valid & np.isfinite(asymmetry) & asymmetry.ne(0) & tail & variation & frame.source_valid & np.isfinite(frame.variation_rank)
    side = -np.sign(asymmetry).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index)
    if control == "no_onset":
        active = eligible
    else:
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
            "up_spot_beta": float(frame.at[index, "up_spot_beta"]),
            "down_spot_beta": float(frame.at[index, "down_spot_beta"]),
            "transmission_asymmetry": float(frame.at[index, "transmission_asymmetry"]),
            "asymmetry_rank": float(frame.at[index, "asymmetry_rank"]),
            "positive_spot_minutes": int(frame.at[index, "positive_spot_minutes"]),
            "negative_spot_minutes": int(frame.at[index, "negative_spot_minutes"]),
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
        raise RuntimeError("HVSCTBA preregistration drift")
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
        "protocol_version": "hvsctba_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
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
