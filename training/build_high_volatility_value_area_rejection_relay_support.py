"""Source-only support evaluator for frozen HVVAR-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_value_area_rejection_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-05-25T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "ada048a779ca82c0b887c143b1f232bc5d68b008b7bf7fd9688683659fe169b3"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_variation_gate", "no_onset", "direction_flip", "value_area_68pct", "forced_long")
ROOT = Path("data/high_volatility_value_area_rejection_relay_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_value_area_rejection_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_value_area_rejection_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_value_area_rejection_relay_support_2026-08-13.json")
QUERY = """SELECT ts,open,high,low,close,volume,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "value_area_lower", "value_area_upper",
    "point_of_control_lower", "point_of_control_upper", "final_hour_high",
    "final_hour_low", "final_close", "rejection", "realized_variation", "variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def strict_prior_midrank(series: pd.Series, lookback: int = 180, minimum: int = 120) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(value) and len(prior) >= minimum:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index, dtype=float)


def contiguous_value_area(
    value_prices: np.ndarray,
    quote_weights: np.ndarray,
    low: float,
    high: float,
    share: float = 0.70,
    bins: int = 24,
) -> tuple[float, float, float, float]:
    """Return value-area and point-of-control edges under the frozen expansion rule."""
    prices = np.asarray(value_prices, dtype=float)
    weights = np.asarray(quote_weights, dtype=float)
    if (
        prices.ndim != 1 or weights.ndim != 1 or len(prices) != len(weights) or not len(prices)
        or not np.isfinite(prices).all() or not np.isfinite(weights).all()
        or np.any(prices <= 0) or np.any(weights <= 0)
        or not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= low
        or not 0 < share <= 1 or bins < 1
    ):
        return (math.nan,) * 4
    log_low, log_high = math.log(low), math.log(high)
    edges = np.linspace(log_low, log_high, bins + 1)
    locations = np.searchsorted(edges, np.log(prices), side="right") - 1
    locations = np.clip(locations, 0, bins - 1)
    volumes = np.bincount(locations, weights=weights, minlength=bins).astype(float)
    total = float(volumes.sum())
    if not math.isfinite(total) or total <= 0:
        return (math.nan,) * 4
    poc = int(np.flatnonzero(volumes == volumes.max())[0])
    left = right = poc
    enclosed = float(volumes[poc])
    while enclosed < share * total and (left > 0 or right < bins - 1):
        lower_volume = volumes[left - 1] if left > 0 else -1.0
        upper_volume = volumes[right + 1] if right < bins - 1 else -1.0
        if lower_volume >= upper_volume:
            left -= 1
            enclosed += float(volumes[left])
        else:
            right += 1
            enclosed += float(volumes[right])
    return tuple(map(float, np.exp([edges[left], edges[right + 1], edges[poc], edges[poc + 1]])))


def _prepare_five_minute_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "open", "high", "low", "close", "volume", "quote_asset_volume"}
    if set(bars.columns) != required:
        raise ValueError(f"HVVAR source schema must be exactly {sorted(required)}")
    market = bars.copy()
    market["ts"] = pd.to_datetime(market["ts"], utc=True, errors="coerce")
    for column in required - {"ts"}:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    if market["ts"].isna().any():
        raise ValueError("HVVAR source contains invalid timestamps")
    duplicated = market["ts"].duplicated(keep=False)
    market["row_valid"] = ~duplicated
    finite = np.isfinite(market[list(required - {"ts"})]).all(axis=1)
    positive = market[list(required - {"ts"})].gt(0).all(axis=1)
    coherent = (
        market["high"].ge(market[["open", "close"]].max(axis=1))
        & market["low"].le(market[["open", "close"]].min(axis=1))
        & market["high"].ge(market["low"])
    )
    market["row_valid"] &= finite & positive & coherent
    market = market.set_index("ts").sort_index()
    grouped = market.resample("5min", label="left", closed="left")
    result = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        quote_asset_volume=("quote_asset_volume", "sum"),
        rows=("close", "size"), valid_rows=("row_valid", "sum"),
    )
    result["source_valid"] = result["rows"].eq(5) & result["valid_rows"].eq(5)
    result["bar_value_price"] = result["quote_asset_volume"] / result["volume"]
    return result


def build_states(bars: pd.DataFrame, start: pd.Timestamp = START, end: pd.Timestamp = END) -> pd.DataFrame:
    market = _prepare_five_minute_bars(bars)
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(pd.Timestamp(start).ceil("4h"), end, freq="4h", inclusive="left"):
        expected = pd.date_range(decision - pd.Timedelta(hours=24), decision, freq="5min", inclusive="left")
        window = market.reindex(expected)
        source_valid = bool(len(window) == 288 and window["source_valid"].eq(True).all())
        lower = upper = poc_lower = poc_upper = math.nan
        lower68 = upper68 = math.nan
        final_high = final_low = final_close = variation = math.nan
        upper_rejection = lower_rejection = False
        upper_rejection68 = lower_rejection68 = False
        if source_valid:
            values = window["bar_value_price"].to_numpy(float)
            quote = window["quote_asset_volume"].to_numpy(float)
            range_low = float(window["low"].min())
            range_high = float(window["high"].max())
            lower, upper, poc_lower, poc_upper = contiguous_value_area(values, quote, range_low, range_high)
            lower68, upper68, _, _ = contiguous_value_area(values, quote, range_low, range_high, share=0.68)
            final = window.iloc[-12:]
            final_high = float(final["high"].max())
            final_low = float(final["low"].min())
            final_close = float(final["close"].iloc[-1])
            upper_rejection = final_high > upper and final_low >= lower and lower <= final_close <= upper
            lower_rejection = final_low < lower and final_high <= upper and lower <= final_close <= upper
            upper_rejection68 = final_high > upper68 and final_low >= lower68 and lower68 <= final_close <= upper68
            lower_rejection68 = final_low < lower68 and final_high <= upper68 and lower68 <= final_close <= upper68
            closes = window["close"].to_numpy(float)
            variation = float(np.sqrt(np.square(np.diff(np.log(closes))).sum()))
            source_valid = bool(
                all(math.isfinite(x) for x in (lower, upper, poc_lower, poc_upper, lower68, upper68, variation))
                and variation > 0 and (upper_rejection != lower_rejection or not (upper_rejection or lower_rejection))
            )
        rows.append({
            "decision_time": decision, "source_valid": source_valid,
            "value_area_lower": lower, "value_area_upper": upper,
            "value_area_68_lower": lower68, "value_area_68_upper": upper68,
            "point_of_control_lower": poc_lower, "point_of_control_upper": poc_upper,
            "final_hour_high": final_high, "final_hour_low": final_low, "final_close": final_close,
            "upper_rejection": upper_rejection, "lower_rejection": lower_rejection,
            "upper_rejection_68": upper_rejection68, "lower_rejection_68": lower_rejection68,
            "realized_variation": variation,
        })
    states = pd.DataFrame(rows)
    states["variation_rank"] = strict_prior_midrank(states["realized_variation"].where(states["source_valid"]))
    return states


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = engine()
    try:
        with database.connect() as connection:
            bars = pd.read_sql_query(
                text(QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()
    states = build_states(bars)
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvvar_source_v1", "query": QUERY,
        "window": [START.isoformat(), END.isoformat()], "source_table": "bars_binance",
        "symbol": "BTCUSDT", "interval": "1m", "outcomes_opened": False,
        "funding_opened": False, "gross9_rows_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(states),
                   "valid_rows": int(states["source_valid"].sum())},
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return states, manifest


def active(states: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.Series]:
    if control != "primary" and control not in CONTROLS:
        raise ValueError(f"unknown HVVAR control: {control}")
    if control == "value_area_68pct":
        upper = states["upper_rejection_68"].eq(True)
        lower = states["lower_rejection_68"].eq(True)
        rejection = pd.Series(np.where(upper, "upper", np.where(lower, "lower", "")), index=states.index)
    else:
        upper = states["upper_rejection"].eq(True)
        lower = states["lower_rejection"].eq(True)
        rejection = pd.Series(np.where(upper, "upper", np.where(lower, "lower", "")), index=states.index)
    raw = states["source_valid"].eq(True) & upper.ne(lower) & (upper | lower)
    variation = pd.Series(True, index=states.index) if control == "no_variation_gate" else states["variation_rank"].ge(0.65)
    state = raw & variation
    eligible = state if control == "no_onset" else state & ~state.shift(1, fill_value=False)
    side = pd.Series(np.where(upper, -1, np.where(lower, 1, 0)), index=states.index, dtype=int)
    if control == "direction_flip": side = -side
    if control == "forced_long": side = pd.Series(1, index=states.index, dtype=int)
    return eligible, side, rejection


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    eligible, side, rejection = active(states, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in states.index[eligible]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        reserved_until = exit_time
        if control == "value_area_68pct":
            value_lower, value_upper = states.at[index, "value_area_68_lower"], states.at[index, "value_area_68_upper"]
        else:
            value_lower, value_upper = states.at[index, "value_area_lower"], states.at[index, "value_area_upper"]
        rows.append({
            "candidate": "HVVAR-8", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "value_area_lower": float(value_lower), "value_area_upper": float(value_upper),
            "point_of_control_lower": float(states.at[index, "point_of_control_lower"]),
            "point_of_control_upper": float(states.at[index, "point_of_control_upper"]),
            "final_hour_high": float(states.at[index, "final_hour_high"]),
            "final_hour_low": float(states.at[index, "final_hour_low"]),
            "final_close": float(states.at[index, "final_close"]), "rejection": rejection.at[index],
            "realized_variation": float(states.at[index, "realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected["side"].eq(1).sum()), int(selected["side"].eq(-1).sum())
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(selected["entry_time"].dt.strftime("%Y-%m").value_counts().max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVVAR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvvar_8_source_support_v1", "policy_id": "HVVAR-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT),
                            "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST),
                            "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "funding_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                            "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value),
                            "promotion_authorized": False} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
