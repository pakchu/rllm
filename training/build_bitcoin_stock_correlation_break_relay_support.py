"""Build outcome-blind DCC-GARCH source support for BSCBR-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from preprocessing.live_db_features import load_env_file, postgres_url_from_env
from training import preregister_bitcoin_stock_correlation_break_relay as prereg
from training.build_cash_open_cross_asset_gap_features import parse_research_payload


ENV_FILE = "/home/pakchu/rllm/.env"
SOURCE_DIR = Path("data/bitcoin_stock_correlation_break_relay_sources_2020_2026")
BUILDER_PATH = Path("training/build_bitcoin_stock_correlation_break_relay_support.py")
SPY_RAW = SOURCE_DIR / "spy_yahoo_chart.json"
SPY_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY?"
    "period1=1575158400&period2=1785628800&interval=1d&events=div%2Csplits&"
    "includeAdjustedClose=true"
)
BTC_HOURLY = SOURCE_DIR / "btc_completed_hour.csv.gz"
SESSION_SCHEDULE = SOURCE_DIR / "spy_session_schedule.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "build_manifest.json"
CLOCK = Path("data/bitcoin_stock_correlation_break_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/bitcoin_stock_correlation_break_relay_controls_2023_2026")
RESULT = Path("results/bitcoin_stock_correlation_break_relay_support_2026-08-09.json")
PREREG_SHA256 = "9cc91a41ea86e5d2b07464a1af1a3d01f436be757506c474f216a26e479d34e4"
START = pd.Timestamp("2019-12-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
FIT_START = pd.Timestamp("2020-01-01")
FIT_END = pd.Timestamp("2023-01-01")
EXPECTED_FIRST_SESSION = pd.Timestamp("2019-12-02")
EXPECTED_LAST_SESSION = pd.Timestamp("2026-07-31")
CLOSED_DATES = frozenset(
    {
        "2019-12-25",
        "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25",
        "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",
        "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31",
        "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
        "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
        "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
        "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
        "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
        "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
        "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03",
    }
)
EARLY_CLOSES = frozenset(
    {
        "2020-11-27",
        "2020-12-24",
        "2021-11-26",
        "2022-11-25",
        "2023-07-03",
        "2023-11-24",
        "2024-07-03",
        "2024-11-29",
        "2024-12-24",
        "2025-07-03",
        "2025-11-28",
        "2025-12-24",
    }
)
BTC_QUERY = """
SELECT
  date_bin('1 hour', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS hour_start,
  (array_agg(open ORDER BY ts))[1] AS hour_open,
  (array_agg(close ORDER BY ts DESC))[1] AS hour_close,
  count(*) AS source_rows,
  count(DISTINCT ts) AS distinct_timestamps,
  min(ts) AS first_ts,
  max(ts) AS last_ts
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
""".strip()
STAGES = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "correlation_level_instead_of_change",
    "one_session_stale_change",
    "direction_flip",
    "rolling_60_session_pearson",
    "current_cross_product",
    "weekday_and_elapsed_gap",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "session_date",
    "cash_close_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "rho_post",
    "delta_rho",
    "btc_sigma_post",
    "btc_sigma_prior_midrank",
    "elapsed_gap_hours",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def query_btc_hourly(env_file: str = ENV_FILE) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    load_env_file(env_file)
    engine = create_engine(
        postgres_url_from_env(env_file),
        connect_args={"connect_timeout": 10},
    )
    try:
        with engine.connect() as connection:
            raw = pd.read_sql_query(
                text(BTC_QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()
    return normalize_btc_hourly(raw)


def normalize_btc_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for name in ("hour_start", "first_ts", "last_ts"):
        frame[name] = pd.to_datetime(frame[name], utc=True, format="mixed")
    for name in ("hour_open", "hour_close"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame["hour_start"].duplicated().any():
        raise RuntimeError("BSCBR duplicate BTC hour")
    grid = pd.DataFrame(
        {"hour_start": pd.date_range(START, END, freq="1h", inclusive="left")}
    )
    frame = grid.merge(frame, on="hour_start", how="left", validate="one_to_one")
    frame["source_rows"] = frame["source_rows"].fillna(0).astype(int)
    frame["distinct_timestamps"] = frame["distinct_timestamps"].fillna(0).astype(int)
    finite = np.isfinite(frame[["hour_open", "hour_close"]]).all(axis=1)
    positive = frame[["hour_open", "hour_close"]].gt(0.0).all(axis=1)
    frame["source_valid"] = (
        frame["source_rows"].eq(60)
        & frame["distinct_timestamps"].eq(60)
        & frame["first_ts"].eq(frame["hour_start"])
        & frame["last_ts"].eq(frame["hour_start"] + pd.Timedelta(minutes=59))
        & finite
        & positive
    )
    frame.loc[~frame["source_valid"], ["hour_open", "hour_close"]] = np.nan
    frame["decision_time"] = frame["hour_start"] + pd.Timedelta(hours=1)
    return frame


def load_spy() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, metadata = parse_research_payload(SPY_RAW.read_bytes(), "SPY")
    cutoff = END.tz_localize(None)
    frame = frame.loc[frame["date"] < cutoff].copy()
    if not frame["history_valid"].all():
        raise RuntimeError("BSCBR invalid SPY source row")
    if frame["date"].iloc[0] != EXPECTED_FIRST_SESSION or frame["date"].iloc[-1] != EXPECTED_LAST_SESSION:
        raise RuntimeError("BSCBR SPY source coverage drift")
    return frame, metadata


def expected_session_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    weekdays = pd.date_range(start, end, freq="B")
    return weekdays[~weekdays.strftime("%Y-%m-%d").isin(CLOSED_DATES)]


def cash_close_time(date: pd.Timestamp) -> tuple[pd.Timestamp, str, bool]:
    date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
    close_hour = 13 if date_text in EARLY_CLOSES else 16
    local = pd.Timestamp(date).tz_localize("America/New_York") + pd.Timedelta(hours=close_hour)
    return local.tz_convert("UTC"), f"{close_hour:02d}:00", close_hour == 13


def session_schedule(spy: pd.DataFrame) -> pd.DataFrame:
    observed = pd.DatetimeIndex(pd.to_datetime(spy["date"]))
    expected = expected_session_dates(observed.min(), observed.max())
    if not observed.equals(expected):
        missing = expected.difference(observed).strftime("%Y-%m-%d").tolist()
        extra = observed.difference(expected).strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"BSCBR SPY/NYSE calendar mismatch missing={missing[:5]} extra={extra[:5]}")
    rows = []
    for date in expected:
        close, local_text, early = cash_close_time(date)
        rows.append(
            {
                "session_date": date,
                "cash_close_time": close,
                "close_local_time": local_text,
                "early_close": early,
            }
        )
    result = pd.DataFrame(rows)
    if result["session_date"].duplicated().any() or not result["session_date"].is_monotonic_increasing:
        raise RuntimeError("BSCBR SPY session schedule invalid")
    return result


def paired_returns(
    spy: pd.DataFrame,
    schedule: pd.DataFrame,
    btc: pd.DataFrame,
) -> pd.DataFrame:
    market = spy[["date", "close", "cash_dividend", "history_valid"]].rename(
        columns={"date": "session_date", "close": "spy_close"}
    )
    frame = schedule.merge(market, on="session_date", validate="one_to_one")
    closes = btc[["decision_time", "hour_close", "source_valid"]].rename(
        columns={"decision_time": "cash_close_time", "hour_close": "btc_close"}
    )
    frame = frame.merge(closes, on="cash_close_time", how="left", validate="one_to_one")
    frame["spy_return"] = np.log(
        (frame["spy_close"] + frame["cash_dividend"])
        / frame["spy_close"].shift(1)
    )
    frame["btc_return"] = np.log(frame["btc_close"] / frame["btc_close"].shift(1))
    frame["elapsed_gap_hours"] = frame["cash_close_time"].diff().dt.total_seconds() / 3600.0
    frame["pair_valid"] = (
        frame["history_valid"].astype(bool)
        & frame["source_valid"].fillna(False).astype(bool)
        & frame["source_valid"].shift(1, fill_value=False).astype(bool)
        & np.isfinite(frame[["spy_return", "btc_return"]]).all(axis=1)
    )
    return frame


def _softmax_pair(u: float, v: float) -> tuple[float, float]:
    values = np.asarray([0.0, u, v], dtype=float)
    weights = np.exp(values - values.max())
    weights /= weights.sum()
    return 0.999 * float(weights[1]), 0.999 * float(weights[2])


def _univariate_parameters(theta: np.ndarray) -> tuple[float, float, float]:
    omega = float(np.exp(theta[0]))
    alpha, beta = _softmax_pair(float(theta[1]), float(theta[2]))
    return omega, alpha, beta


def garch_filter(
    residuals: np.ndarray,
    parameters: tuple[float, float, float],
    initial_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega, alpha, beta = parameters
    pre = np.empty(len(residuals), dtype=float)
    post = np.empty(len(residuals), dtype=float)
    variance = float(initial_variance)
    for index, residual in enumerate(residuals):
        pre[index] = variance
        variance = omega + alpha * residual * residual + beta * variance
        post[index] = variance
    return pre, post


def fit_garch(values: np.ndarray) -> dict[str, Any]:
    mean = float(np.mean(values))
    residuals = np.asarray(values, dtype=float) - mean
    initial_variance = float(np.mean(residuals * residuals))
    if not np.isfinite(initial_variance) or initial_variance <= 0.0:
        raise RuntimeError("BSCBR zero GARCH fit variance")
    initial_omega = 0.05 * initial_variance
    start = np.asarray([np.log(initial_omega), np.log(0.05 / 0.05), np.log(0.90 / 0.05)])

    def objective(theta: np.ndarray) -> float:
        pre, _ = garch_filter(residuals, _univariate_parameters(theta), initial_variance)
        if not np.isfinite(pre).all() or np.any(pre <= 0.0):
            return 1e100
        return float(np.sum(np.log(pre) + residuals * residuals / pre))

    fitted = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * 3,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"BSCBR GARCH optimizer failed: {fitted.message}")
    reject_optimizer_boundary(fitted.x, "GARCH")
    omega, alpha, beta = _univariate_parameters(fitted.x)
    return {
        "mean": mean,
        "initial_variance": initial_variance,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "objective": float(fitted.fun),
        "iterations": int(fitted.nit),
    }


def _normalize_q(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diag(matrix)
    if not np.isfinite(matrix).all() or np.any(diagonal <= 0.0):
        raise RuntimeError("BSCBR invalid DCC Q state")
    scale = np.sqrt(diagonal)
    correlation = matrix / np.outer(scale, scale)
    if np.linalg.det(correlation) <= 0.0:
        raise RuntimeError("BSCBR non-positive DCC correlation determinant")
    return correlation


def dcc_filter(
    standardized: np.ndarray,
    qbar: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    pre_rho = np.empty(len(standardized), dtype=float)
    post_rho = np.empty(len(standardized), dtype=float)
    q_pre = np.asarray(qbar, dtype=float).copy()
    for index, z in enumerate(standardized):
        pre_rho[index] = _normalize_q(q_pre)[0, 1]
        q_post = (1.0 - a - b) * qbar + a * np.outer(z, z) + b * q_pre
        post_rho[index] = _normalize_q(q_post)[0, 1]
        q_pre = q_post
    return pre_rho, post_rho


def fit_dcc(standardized: np.ndarray) -> dict[str, Any]:
    qbar = np.cov(standardized, rowvar=False, ddof=0)
    if qbar.shape != (2, 2) or np.linalg.det(qbar) <= 0.0:
        raise RuntimeError("BSCBR invalid DCC Qbar")
    start = np.asarray([np.log(0.02 / 0.03), np.log(0.95 / 0.03)])

    def objective(theta: np.ndarray) -> float:
        a, b = _softmax_pair(float(theta[0]), float(theta[1]))
        q_pre = qbar.copy()
        total = 0.0
        try:
            for z in standardized:
                correlation = _normalize_q(q_pre)
                total += np.log(np.linalg.det(correlation)) + z @ np.linalg.solve(correlation, z)
                q_pre = (1.0 - a - b) * qbar + a * np.outer(z, z) + b * q_pre
        except (RuntimeError, np.linalg.LinAlgError):
            return 1e100
        return float(total) if np.isfinite(total) else 1e100

    fitted = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * 2,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"BSCBR DCC optimizer failed: {fitted.message}")
    reject_optimizer_boundary(fitted.x, "DCC")
    a, b = _softmax_pair(float(fitted.x[0]), float(fitted.x[1]))
    return {
        "a": a,
        "b": b,
        "qbar": qbar.tolist(),
        "objective": float(fitted.fun),
        "iterations": int(fitted.nit),
    }


def reject_optimizer_boundary(parameters: np.ndarray, label: str) -> None:
    if np.any(np.isclose(np.abs(np.asarray(parameters, dtype=float)), 20.0, atol=1e-7, rtol=0.0)):
        raise RuntimeError(f"BSCBR {label} optimizer boundary drift")


def strict_prior_midrank(values: pd.Series, window: int = 90, minimum: int = 60) -> pd.Series:
    array = values.to_numpy(float)
    ranks = np.full(len(array), np.nan)
    for index, current in enumerate(array):
        prior = array[max(0, index - window) : index]
        prior = prior[np.isfinite(prior)]
        if len(prior) >= minimum and np.isfinite(current):
            ranks[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
    return pd.Series(ranks, index=values.index)


def fit_and_filter(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = frame.loc[frame["session_date"].ge(FIT_START)].copy()
    invalid = required.loc[~required["pair_valid"]]
    if not invalid.empty:
        dates = invalid["session_date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"BSCBR missing required paired session: {dates[:5]}")
    valid = required
    fit_mask = valid["session_date"].ge(FIT_START) & valid["session_date"].lt(FIT_END)
    fit = valid.loc[fit_mask]
    if len(fit) < 500:
        raise RuntimeError("BSCBR insufficient DCC fit rows")
    spy_fit = fit_garch(fit["spy_return"].to_numpy(float))
    btc_fit = fit_garch(fit["btc_return"].to_numpy(float))
    valid = valid.loc[valid["session_date"].ge(FIT_START)].copy().reset_index(drop=True)
    standardized = []
    sigma_post = None
    fits = (spy_fit, btc_fit)
    for column, fitted in zip(("spy_return", "btc_return"), fits, strict=True):
        residuals = valid[column].to_numpy(float) - fitted["mean"]
        pre, post = garch_filter(
            residuals,
            (fitted["omega"], fitted["alpha"], fitted["beta"]),
            fitted["initial_variance"],
        )
        standardized.append(residuals / np.sqrt(pre))
        if column == "btc_return":
            sigma_post = np.sqrt(post)
    standardized_array = np.column_stack(standardized)
    fit_count = int(fit_mask.sum())
    dcc_fit = fit_dcc(standardized_array[:fit_count])
    _, valid["rho_post"] = dcc_filter(
        standardized_array,
        np.asarray(dcc_fit["qbar"]),
        dcc_fit["a"],
        dcc_fit["b"],
    )
    valid["btc_sigma_post"] = sigma_post
    valid["spy_z"] = standardized_array[:, 0]
    valid["btc_z"] = standardized_array[:, 1]
    valid["delta_rho"] = valid["rho_post"].diff()
    valid["btc_sigma_prior_midrank"] = strict_prior_midrank(valid["btc_sigma_post"])
    valid["rolling_rho"] = valid["spy_return"].rolling(60, min_periods=60).corr(valid["btc_return"])
    valid["rolling_delta_rho"] = valid["rolling_rho"].diff()
    return valid, {"spy_garch": spy_fit, "btc_garch": btc_fit, "dcc": dcc_fit, "fit_rows": fit_count}


def _signal(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    delta = frame["delta_rho"].copy()
    side_source = delta.copy()
    active = delta.abs().ge(0.02)
    if control == "correlation_level_instead_of_change":
        side_source = frame["rho_post"]
        active = side_source.abs().ge(0.02)
    elif control == "one_session_stale_change":
        side_source = delta.shift(1)
        active = side_source.abs().ge(0.02)
    elif control == "rolling_60_session_pearson":
        side_source = frame["rolling_delta_rho"]
        active = side_source.abs().ge(0.02)
    elif control == "current_cross_product":
        side_source = frame["spy_z"] * frame["btc_z"]
    elif control == "weekday_and_elapsed_gap":
        active &= frame["elapsed_gap_hours"].le(24.0)
    if control != "no_volatility_gate":
        active &= frame["btc_sigma_prior_midrank"].ge(0.65)
    side = -np.sign(side_source).astype("Int64")
    if control == "direction_flip":
        side = -side
    active &= side_source.notna() & side.ne(0)
    return active, side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = _signal(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        close = pd.Timestamp(frame.at[index, "cash_close_time"])
        feature = close + pd.Timedelta(minutes=5)
        entry = feature + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (
                name
                for name, (start, end) in STAGES.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "BSCBR-24",
                "control": control,
                "split": split,
                "session_date": frame.at[index, "session_date"],
                "cash_close_time": close,
                "feature_available_time": feature,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides.at[index]),
                "rho_post": float(frame.at[index, "rho_post"]),
                "delta_rho": float(frame.at[index, "delta_rho"]),
                "btc_sigma_post": float(frame.at[index, "btc_sigma_post"]),
                "btc_sigma_prior_midrank": float(frame.at[index, "btc_sigma_prior_midrank"]),
                "elapsed_gap_hours": float(frame.at[index, "elapsed_gap_hours"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock.loc[clock["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = pd.to_datetime(selected["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": int(len(selected)),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run(env_file: str = ENV_FILE) -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("BSCBR preregistration hash drift")
    spy, spy_metadata = load_spy()
    schedule = session_schedule(spy)
    btc = query_btc_hourly(env_file)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(btc, BTC_HOURLY)
    write_gzip_csv(schedule, SESSION_SCHEDULE)
    paired = paired_returns(spy, schedule, btc)
    features, estimator = fit_and_filter(paired)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "bscbr_24_source_v1",
        "preregistration_sha256": PREREG_SHA256,
        "outcomes_opened": False,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "spy": {
            "path": str(SPY_RAW),
            "sha256": sha256(SPY_RAW),
            "provider": "Yahoo Finance chart cache (research only)",
            "url": SPY_URL,
            "metadata": spy_metadata,
        },
        "btc": {
            "path": str(BTC_HOURLY),
            "sha256": sha256(BTC_HOURLY),
            "query": BTC_QUERY,
            "table": "bars_binance",
            "rows": int(len(btc)),
            "valid_rows": int(btc["source_valid"].sum()),
        },
        "session_schedule": {
            "path": str(SESSION_SCHEDULE),
            "sha256": sha256(SESSION_SCHEDULE),
            "rows": int(len(schedule)),
            "closed_dates": sorted(CLOSED_DATES),
            "early_close_dates": sorted(EARLY_CLOSES),
            "references": [
                "https://www.nyse.com/markets/hours-calendars",
                "https://ir.theice.com/press/news-details/2019/NYSE-Group-Announces-2020-2021-and-2022-Holiday-and-Early-Closings-Calendar/default.aspx",
            ],
        },
        "estimator": estimator,
        "builder": {"path": str(BUILDER_PATH), "sha256": sha256(BUILDER_PATH)},
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: support_stats(primary, name) for name in STAGES}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "bscbr_24_source_support_v1",
        "policy_id": "BSCBR-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA256,
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": int(len(primary))},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": int(len(frame)),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=ENV_FILE)
    args = parser.parse_args()
    report = run(args.env_file)
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))


if __name__ == "__main__":
    main()
