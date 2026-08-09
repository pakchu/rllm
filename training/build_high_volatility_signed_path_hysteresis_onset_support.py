"""Open source-only OOS incidence for preregistered HVSPH-8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_signed_path_hysteresis_onset as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "b7221b2b261d5b1d50b8e9930f5b0f7352ee5cc3f6db9bcf1ebcb1776e6b4d3b"
SOURCE_DIR = Path("data/high_volatility_signed_path_hysteresis_onset_sources_2023_2026")
PANEL = SOURCE_DIR / "hourly_path_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_signed_path_hysteresis_onset_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_signed_path_hysteresis_onset_controls_2023_2026")
RESULT = Path("results/high_volatility_signed_path_hysteresis_onset_support_2026-08-10.json")
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_area_gate",
    "one_hour_stale_features",
    "direction_flip",
    "forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "normalized_area", "absolute_area_rank",
    "realized_variation", "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 720, minimum: int = 480
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def normalized_loop_area(close: np.ndarray) -> tuple[float, float]:
    close = np.asarray(close, dtype=float)
    if close.shape != (480,) or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("HVSPH close path invalid")
    returns = np.diff(np.log(close))
    variation = float(np.square(returns).sum())
    root_variation = math.sqrt(variation)
    if not math.isfinite(root_variation) or root_variation <= 0:
        raise ValueError("HVSPH path variation invalid")
    x = np.r_[0.0, np.cumsum(returns)]
    y = np.r_[0.0, np.cumsum(np.square(returns)) / variation]
    area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    normalized = area / root_variation
    if not math.isfinite(normalized) or normalized == 0:
        raise ValueError("HVSPH normalized loop area invalid")
    return normalized, variation


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def feature_panel(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START + pd.Timedelta(hours=8), END, freq="1h", inclusive="left"):
        expected = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        window = frame.reindex(expected)
        ohlc = window[["open", "high", "low", "close"]]
        valid = bool(
            len(window) == 480
            and np.isfinite(ohlc).all(axis=1).all()
            and ohlc.gt(0).all(axis=1).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        area = variation = float("nan")
        if valid:
            try:
                area, variation = normalized_loop_area(window.close.to_numpy(float))
            except ValueError:
                valid = False
        rows.append({
            "decision_time": decision,
            "source_valid": valid,
            "normalized_area": area,
            "realized_variation": variation,
        })
    panel = pd.DataFrame(rows)
    panel["absolute_area_rank"] = strict_prior_midrank(
        panel.normalized_area.abs().where(panel.source_valid)
    )
    panel["realized_variation_rank"] = strict_prior_midrank(
        panel.realized_variation.where(panel.source_valid)
    )
    return panel


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = panel.copy()
    area = frame.normalized_area
    area_rank = frame.absolute_area_rank
    variation_rank = frame.realized_variation_rank
    valid = frame.source_valid
    feature_available_time = frame.decision_time
    if control == "one_hour_stale_features":
        area = area.shift(1)
        area_rank = area_rank.shift(1)
        variation_rank = variation_rank.shift(1)
        valid = valid.shift(1, fill_value=False)
        feature_available_time = frame.decision_time - pd.Timedelta(hours=1)
    prior_valid = valid.shift(1, fill_value=False)
    reversal = area.notna() & area.shift(1).notna() & np.sign(area).ne(np.sign(area.shift(1)))
    volatility_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.80)
    area_gate = pd.Series(True, index=frame.index) if control == "no_area_gate" else area_rank.ge(0.80)
    eligible = valid & prior_valid & reversal & volatility_gate & area_gate
    side = pd.Series(np.where(area.gt(0), 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[eligible & frame.decision_time.ge(SPLITS["train"][0])]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_
        rows.append({
            "candidate": "HVSPH-8",
            "control": control,
            "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(feature_available_time.at[index]),
            "entry_time": entry,
            "exit_time": exit_,
            "side": int(side.at[index]),
            "normalized_area": float(area.at[index]),
            "absolute_area_rank": float(area_rank.at[index]),
            "realized_variation": float(frame.at[index, "realized_variation"]),
            "realized_variation_rank": float(variation_rank.at[index]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    from sqlalchemy import text

    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVSPH preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVSPH preregistration payload drift")
    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()}
        )
    database.dispose()
    panel = feature_panel(bars)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    source_core = {
        "protocol_version": "hvsph_8_source_materialization_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "no_imputation": True,
        "oos_postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
    }
    source = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    primary = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvsph_8_oos_source_support_v1",
        "policy_id": "HVSPH-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "oos_postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
