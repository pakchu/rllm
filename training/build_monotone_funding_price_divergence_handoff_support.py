"""Materialize outcome-blind source support for frozen MFDH-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_monotone_funding_price_divergence_handoff as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_monotone_funding_price_divergence_handoff_support.py")
PREREG_SHA = "5e5b1a98c63c8ea8680f61379d5943dace6ed4b5ec524f1e3471d6cf7447eab1"
SOURCE_DIR = Path("data/monotone_funding_price_divergence_handoff_sources_2022_2026")
FEATURES = SOURCE_DIR / "mfdh_preentry_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/monotone_funding_price_divergence_handoff_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/monotone_funding_price_divergence_handoff_controls_2023_2026")
RESULT = Path("results/monotone_funding_price_divergence_handoff_support_2026-08-09.json")

# The extra history supports the frozen 756-observation causal RV20 report and the
# 270-settlement return rank before the first train decision.
BAR_START = pd.Timestamp("2022-09-10T00:00:00Z")
FUNDING_START = pd.Timestamp("2022-10-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_return_rank",
    "two_settlement_acceleration",
    "funding_side_instead_of_price_side",
    "direction_flip",
)
RETURN_RANK_HISTORY = 270
RETURN_RANK_MINIMUM = 180
RETURN_RANK_CUTOFF = 0.60
RV20_THRESHOLD_HISTORY = 756

BAR_QUERY = (
    "SELECT ts,open,high,low,close FROM bars_binance "
    "WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
)
FUNDING_QUERY = (
    "SELECT funding_time,funding_rate,mark_price FROM funding_rates_binance "
    "WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end "
    "ORDER BY funding_time"
)
FEATURE_COLUMNS = (
    "settlement_time",
    "decision_time",
    "feature_available_time",
    "funding_rate_f2",
    "funding_rate_f1",
    "funding_rate_f0",
    "funding_event_valid",
    "three_settlement_path",
    "two_settlement_acceleration",
    "return_window_valid",
    "return_16h",
    "absolute_return_prior_midrank",
    "rv20",
    "rv20_prior_q90",
    "rv20_q90_active",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "settlement_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "funding_rate_f2",
    "funding_rate_f1",
    "funding_rate_f0",
    "return_16h",
    "absolute_return_prior_midrank",
    "rv20",
    "rv20_prior_q90",
    "rv20_q90_active",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    values: pd.Series,
    lookback: int = RETURN_RANK_HISTORY,
    minimum: int = RETURN_RANK_MINIMUM,
) -> pd.Series:
    """Rank each finite value against finite prior values only, excluding current."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            result.at[index] = (
                np.count_nonzero(array < current)
                + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return result


def strict_prior_quantile(
    values: pd.Series,
    quantile: float = 0.90,
    lookback: int = RV20_THRESHOLD_HISTORY,
) -> pd.Series:
    """Compute the frozen linear quantile from exactly ``lookback`` prior values."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) == lookback:
            result.at[index] = float(np.quantile(np.asarray(prior), quantile, method="linear"))
        if np.isfinite(current):
            history.append(float(current))
    return result


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only completed funding and pre-entry feature columns."""
    from sqlalchemy import text

    connection_engine = engine()
    try:
        bars = pd.read_sql_query(
            text(BAR_QUERY),
            connection_engine,
            params={"start": BAR_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()},
        )
        funding = pd.read_sql_query(
            text(FUNDING_QUERY),
            connection_engine,
            params={"start": FUNDING_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()},
        )
    finally:
        connection_engine.dispose()
    if bars.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("MFDH BTC bar schema drift")
    if funding.columns.tolist() != ["funding_time", "funding_rate", "mark_price"]:
        raise RuntimeError("MFDH funding schema drift")
    return bars, funding


def _normalise_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close"]
    if not set(required).issubset(bars.columns):
        raise ValueError("MFDH bars missing required columns")
    frame = bars[required].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("ts", kind="mergesort", na_position="last").reset_index(drop=True)


def _normalise_funding(funding: pd.DataFrame) -> pd.DataFrame:
    required = ["funding_time", "funding_rate", "mark_price"]
    if not set(required).issubset(funding.columns):
        raise ValueError("MFDH funding missing required columns")
    frame = funding[required].copy()
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values("funding_time", kind="mergesort", na_position="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    duplicate = frame["funding_time"].notna() & frame["funding_time"].duplicated(keep=False)
    timestamp = frame["funding_time"]
    on_schedule = (
        timestamp.notna()
        & timestamp.dt.second.eq(0)
        & timestamp.dt.microsecond.eq(0)
        & timestamp.dt.minute.eq(0)
        & timestamp.dt.hour.isin([0, 8, 16])
    )
    frame["funding_event_valid"] = (
        on_schedule
        & ~duplicate
        & np.isfinite(frame["funding_rate"])
        & np.isfinite(frame["mark_price"])
        & frame["mark_price"].gt(0)
    )
    return frame


def _valid_bar_rows(frame: pd.DataFrame) -> pd.Series:
    prices = frame[["open", "high", "low", "close"]]
    finite_positive = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    coherent = (
        frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame["low"])
    )
    return finite_positive & coherent


def _return_16h(bars: pd.DataFrame, settlement: pd.Timestamp) -> tuple[bool, float]:
    start = settlement - pd.Timedelta(hours=16)
    end = settlement - pd.Timedelta(minutes=1)
    window = bars.loc[start:end]
    expected = pd.date_range(start, settlement, freq="1min", inclusive="left")
    exact = (
        len(window) == 960
        and not window["ts"].duplicated().any()
        and window.index.equals(expected)
    )
    if not exact or not _valid_bar_rows(window).all():
        return False, float("nan")
    return True, float(np.log(window.iloc[-1]["close"] / window.iloc[0]["open"]))


def _daily_rv20(bars: pd.DataFrame) -> tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, float]]:
    usable = bars[bars["ts"].notna()].copy()
    usable["day"] = usable["ts"].dt.floor("D")
    closes: dict[pd.Timestamp, float] = {}
    for day, group in usable.groupby("day", sort=True):
        expected = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
        exact = (
            len(group) == 1440
            and not group["ts"].duplicated().any()
            and group["ts"].reset_index(drop=True).equals(pd.Series(expected, name="ts"))
        )
        if exact and _valid_bar_rows(group).all():
            closes[pd.Timestamp(day)] = float(group.iloc[-1]["close"])

    returns: dict[pd.Timestamp, float] = {}
    for day, close in closes.items():
        prior = day - pd.Timedelta(days=1)
        if prior in closes:
            returns[day] = float(np.log(close / closes[prior]))

    rv20: dict[pd.Timestamp, float] = {}
    for day in closes:
        days = pd.date_range(day - pd.Timedelta(days=20), day - pd.Timedelta(days=1), freq="D")
        if all(pd.Timestamp(item) in returns for item in days):
            values = np.asarray([returns[pd.Timestamp(item)] for item in days])
            rv20[day] = float(np.sqrt(365.0 * np.mean(np.square(values))))
    rv20_series = pd.Series(rv20, dtype=float).sort_index()
    threshold_series = (
        rv20_series.rolling(RV20_THRESHOLD_HISTORY, min_periods=RV20_THRESHOLD_HISTORY)
        .quantile(0.90, interpolation="linear")
        .shift(1)
    )
    return rv20, {pd.Timestamp(day): float(value) for day, value in threshold_series.dropna().items()}


def build_features(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    bars = _normalise_bars(bars)
    funding = _normalise_funding(funding)
    funding = funding[funding["funding_time"].notna()].copy().reset_index(drop=True)
    rv20_by_day, rv20_threshold_by_day = _daily_rv20(bars)
    indexed_bars = bars.set_index("ts", drop=False)
    rows: list[dict[str, Any]] = []
    for index, current in funding.iterrows():
        settlement = pd.Timestamp(current["funding_time"])
        window_valid, return_16h = _return_16h(indexed_bars, settlement)
        f0 = float(current["funding_rate"])
        f1 = float(funding.at[index - 1, "funding_rate"]) if index >= 1 else float("nan")
        f2 = float(funding.at[index - 2, "funding_rate"]) if index >= 2 else float("nan")
        event_valid = bool(current["funding_event_valid"])
        prior_one_valid = index >= 1 and bool(funding.at[index - 1, "funding_event_valid"])
        prior_two_valid = index >= 2 and bool(funding.at[index - 2, "funding_event_valid"])
        prior_one_consecutive = (
            index >= 1
            and settlement - pd.Timestamp(funding.at[index - 1, "funding_time"])
            == pd.Timedelta(hours=8)
        )
        prior_two_consecutive = (
            index >= 2
            and pd.Timestamp(funding.at[index - 1, "funding_time"])
            - pd.Timestamp(funding.at[index - 2, "funding_time"])
            == pd.Timedelta(hours=8)
        )
        two_path = (
            event_valid
            and prior_one_valid
            and prior_one_consecutive
            and f0 != 0
            and f1 != 0
            and np.sign(f0) == np.sign(f1)
            and abs(f1) < abs(f0)
        )
        three_path = (
            two_path
            and prior_two_valid
            and prior_two_consecutive
            and f2 != 0
            and np.sign(f2) == np.sign(f1)
            and abs(f2) < abs(f1)
        )
        rows.append(
            {
                "settlement_time": settlement,
                "decision_time": settlement,
                "feature_available_time": settlement,
                "funding_rate_f2": f2,
                "funding_rate_f1": f1,
                "funding_rate_f0": f0,
                "funding_event_valid": event_valid,
                "three_settlement_path": three_path,
                "two_settlement_acceleration": two_path,
                "return_window_valid": window_valid,
                "return_16h": return_16h,
                "rv20": rv20_by_day.get(settlement.floor("D"), float("nan")),
                "rv20_prior_q90": rv20_threshold_by_day.get(
                    settlement.floor("D"), float("nan")
                ),
            }
        )
    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rank_input = features["return_16h"].abs().where(
        features["funding_event_valid"] & features["return_window_valid"]
    )
    features["absolute_return_prior_midrank"] = strict_prior_midrank(rank_input)
    features["rv20_q90_active"] = (
        features["rv20_prior_q90"].notna()
        & features["rv20"].ge(features["rv20_prior_q90"])
    )
    return features.loc[:, FEATURE_COLUMNS]


def signal(features: pd.DataFrame, control: str = "primary") -> pd.Series:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    funding_sign = np.sign(features["funding_rate_f0"])
    return_sign = np.sign(features["return_16h"])
    path_column = (
        "two_settlement_acceleration"
        if control == "two_settlement_acceleration"
        else "three_settlement_path"
    )
    eligible = (
        features["funding_event_valid"]
        & features["return_window_valid"]
        & features[path_column]
        & features["return_16h"].ne(0)
        & return_sign.eq(-funding_sign)
    )
    if control != "no_return_rank":
        eligible &= features["absolute_return_prior_midrank"].ge(RETURN_RANK_CUTOFF)
    side = return_sign.astype("Int64").fillna(0).astype(int)
    if control == "funding_side_instead_of_price_side":
        side = funding_sign.astype("Int64").fillna(0).astype(int)
    elif control == "direction_flip":
        side = -side
    return side.where(eligible, 0)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    sides = signal(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        available = pd.Timestamp(features.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if available > decision or available >= entry:
            continue
        # Global half-open reservation: an entry exactly at the prior exit is allowed.
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:  # Includes the frozen split-crossing skip.
            continue
        next_allowed = exit_time
        row = {
            "candidate": "MFDH-8",
            "control": control,
            "split": split,
            "settlement_time": pd.Timestamp(features.at[index, "settlement_time"]),
            "decision_time": decision,
            "feature_available_time": available,
            "entry_time": entry,
            "exit_time": exit_time,
            "side": int(sides.at[index]),
        }
        for column in CLOCK_COLUMNS[9:]:
            value = features.at[index, column]
            row[column] = bool(value) if column == "rv20_q90_active" else float(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock["split"].eq(split)].copy()
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
            "rv20_q90_events_report_only": 0,
        }
    entries = pd.to_datetime(selected["entry_time"], utc=True)
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(selected),
        "rv20_q90_events_report_only": int(selected["rv20_q90_active"].sum()),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("MFDH preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    bars, funding = load_sources()
    features = build_features(bars, funding)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "mfdh_8_sources_v1",
        "queries": {"funding": FUNDING_QUERY, "bars": BAR_QUERY},
        "windows": {
            "funding": [FUNDING_START.isoformat(), SOURCE_END.isoformat()],
            "bars": [BAR_START.isoformat(), SOURCE_END.isoformat()],
        },
        "rows": {"funding": len(funding), "bars": len(bars), "features": len(features)},
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
        "feature_output": {"path": str(FEATURES), "sha256": sha256(FEATURES)},
        "completed_preentry_sources_opened": True,
        "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n")

    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "mfdh_8_source_support_v1",
        "policy_id": "MFDH-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST),
            "sha256": sha256(MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "rv20_q90": {"entry_filter": False, "report_only": True},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
