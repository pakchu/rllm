"""Materialize source-only HVSPSTL-8 clocks before novelty or economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_spot_perpetual_sign_transition_leadership_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "083d31c6caa3286d4bc3f150cd694ea3a05d10c30073b8e724078c5e38798d7f"
START = pd.Timestamp("2023-03-01T01:00:00Z")
END = pd.Timestamp("2026-08-01T01:00:00Z")
SOURCE_DIR = Path("data/high_volatility_spot_perpetual_sign_transition_leadership_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "eight_hour_sign_transition_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_spot_perpetual_sign_transition_leadership_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_spot_perpetual_sign_transition_leadership_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_perpetual_sign_transition_leadership_relay_support_2026-08-13.json")
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
    "entry_time", "exit_time", "side", "spot_to_perp_fraction", "perp_to_spot_fraction",
    "leadership_advantage", "leadership_rank", "spot_transition_pairs",
    "perp_transition_pairs", "final_spot_return", "final_perp_return", "full_variation", "variation_rank",
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


def transition_metrics(spot_close: np.ndarray, perp_close: np.ndarray) -> dict[str, float | int | bool]:
    spot_return = np.diff(np.log(np.asarray(spot_close, dtype=float)))
    perp_return = np.diff(np.log(np.asarray(perp_close, dtype=float)))
    ss, ps = np.sign(spot_return), np.sign(perp_return)
    sp = (ss[:-1] != 0) & (ps[1:] != 0); pp = (ps[:-1] != 0) & (ss[1:] != 0)
    sp_n, pp_n = int(sp.sum()), int(pp.sum())
    sp_f = float(np.mean(ss[:-1][sp] == ps[1:][sp])) if sp_n else float("nan")
    pp_f = float(np.mean(ps[:-1][pp] == ss[1:][pp])) if pp_n else float("nan")
    advantage = sp_f - pp_f
    final_spot, final_perp = float(spot_return[-60:].sum()), float(perp_return[-60:].sum())
    variation = float(np.sqrt(np.square(perp_return).sum()))
    valid = bool(
        sp_n >= 360 and pp_n >= 360 and np.isfinite([sp_f, pp_f, advantage, final_spot, final_perp, variation]).all()
        and advantage > 0 and final_spot != 0 and np.sign(final_spot) == np.sign(final_perp) and variation > 0
    )
    return {
        "source_valid": valid, "spot_to_perp_fraction": sp_f, "perp_to_spot_fraction": pp_f,
        "leadership_advantage": advantage, "spot_transition_pairs": sp_n, "perp_transition_pairs": pp_n,
        "final_spot_return": final_spot, "final_perp_return": final_perp, "full_variation": variation,
    }


def build_panel(perp: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    perp, spot = _prepare(perp), _prepare(spot)
    decisions = pd.date_range(START, END, freq="8h", inclusive="left")
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        expected = pd.date_range(decision - pd.Timedelta(hours=8, minutes=1), decision, freq="1min", inclusive="left")
        perp_window, spot_window = perp.reindex(expected), spot.reindex(expected)
        if _valid(perp_window) and _valid(spot_window):
            metrics = transition_metrics(spot_window.close.to_numpy(), perp_window.close.to_numpy())
        else:
            metrics = {
                "source_valid": False, "spot_to_perp_fraction": np.nan, "perp_to_spot_fraction": np.nan,
                "leadership_advantage": np.nan, "spot_transition_pairs": 0, "perp_transition_pairs": 0,
                "final_spot_return":np.nan,"final_perp_return":np.nan,"full_variation": np.nan,
            }
        rows.append({"decision_time": decision, "perp_source_rows": int(perp_window.notna().all(axis=1).sum()), "spot_source_rows": int(spot_window.notna().all(axis=1).sum()), **metrics})
    panel = pd.DataFrame(rows)
    panel["leadership_rank"] = strict_prior_midrank(panel.leadership_advantage.where(panel.source_valid))
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
        "protocol_version": "hvspstl_8_btc_source_v1",
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
    for column in ("spot_to_perp_fraction","perp_to_spot_fraction","leadership_advantage","leadership_rank","spot_transition_pairs","perp_transition_pairs","final_spot_return","final_perp_return","full_variation","variation_rank"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    advantage = frame.leadership_advantage
    leadership_rank = frame.leadership_rank
    source_valid = frame.source_valid
    if control == "one_block_stale_leadership":
        advantage, leadership_rank, source_valid = advantage.shift(1), leadership_rank.shift(1), source_valid.shift(1, fill_value=False)
    tail = pd.Series(True, index=frame.index) if control == "no_leadership_tail" else leadership_rank.ge(0.75)
    variation = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame.variation_rank.ge(0.65)
    eligible = source_valid & np.isfinite(advantage) & advantage.gt(0) & tail & variation & frame.source_valid & np.isfinite(frame.variation_rank)
    side = np.sign(frame.final_spot_return).fillna(0).astype(int)
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
            "spot_to_perp_fraction":float(frame.at[index,"spot_to_perp_fraction"]),"perp_to_spot_fraction":float(frame.at[index,"perp_to_spot_fraction"]),
            "leadership_advantage":float(frame.at[index,"leadership_advantage"]),"leadership_rank":float(frame.at[index,"leadership_rank"]),
            "spot_transition_pairs":int(frame.at[index,"spot_transition_pairs"]),"perp_transition_pairs":int(frame.at[index,"perp_transition_pairs"]),
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
        raise RuntimeError("HVSPSTL preregistration drift")
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
        "protocol_version": "hvspstl_8_source_support_v1", "policy_id": prereg.POLICY_ID,
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
