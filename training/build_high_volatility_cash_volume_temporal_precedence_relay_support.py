"""Build source-only support for frozen HVCVTP-8."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_cash_volume_temporal_precedence_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "573eaefe87cf1c69501264243f887032281955c8e4cbe65f22f8950d196836af"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
GATES = REGISTRATION["source_support_gates"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,open,high,low,close,quote_asset_volume
FROM {table}
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
QUERIES = {
    "spot": QUERY.format(table="bars_binance_spot"),
    "perpetual": QUERY.format(table="bars_binance"),
}

ROOT = Path("data/high_volatility_cash_volume_temporal_precedence_relay_sources_2023_2026")
PANEL = ROOT / "block_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_cash_volume_temporal_precedence_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_cash_volume_temporal_precedence_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_cash_volume_temporal_precedence_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_cash_volume_temporal_precedence_relay_support_2026-08-13.json")
BUILDER = Path("training/build_high_volatility_cash_volume_temporal_precedence_relay_support.py")

PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid",
    "spot_minute_count", "perpetual_minute_count",
    "spot_weighted_median_minute", "perpetual_weighted_median_minute",
    "cash_precedence", "precedence_rank",
    "spot_weighted_mean_minute", "perpetual_weighted_mean_minute",
    "mean_arrival_precedence", "spot_return", "perpetual_return",
    "spot_final_two_hour_return", "perpetual_final_two_hour_return",
    "direction_side", "perpetual_variation", "variation_rank", "eligible", "onset",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", *PANEL_COLUMNS[5:19],
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    series: pd.Series,
    lookback: int = POLICY["prior_decisions"],
    minimum: int = POLICY["minimum_prior_decisions"],
) -> pd.Series:
    """Rank finite values against at most ``lookback`` finite prior values only."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(value) and len(prior) >= minimum:
            output[index] = float(
                (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
            )
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index, dtype=float)


def weighted_median_minute(weights: np.ndarray) -> float:
    """Return the lowest zero-based minute crossing half of positive total weight."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (480,) or not np.isfinite(values).all() or np.any(values < 0):
        return math.nan
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return math.nan
    return float(np.searchsorted(np.cumsum(values), total / 2.0, side="left"))


def weighted_mean_minute(weights: np.ndarray) -> float:
    """Return the quote-turnover-weighted zero-based mean minute."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (480,) or not np.isfinite(values).all() or np.any(values < 0):
        return math.nan
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return math.nan
    return float(np.dot(np.arange(480, dtype=float), values) / total)


def _engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only frozen completed signal columns; no execution or outcome source is opened."""
    from sqlalchemy import text

    database = _engine()
    params = {"start": START.to_pydatetime(), "end": END.to_pydatetime()}
    try:
        with database.connect() as connection:
            spot = pd.read_sql_query(text(QUERIES["spot"]), connection, params=params)
            perpetual = pd.read_sql_query(text(QUERIES["perpetual"]), connection, params=params)
    finally:
        database.dispose()
    return spot, perpetual


def prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    expected = ["ts", "open", "high", "low", "close", "quote_asset_volume"]
    if frame.columns.tolist() != expected:
        raise RuntimeError(f"HVCVTP {label} source schema drift")
    prepared = frame.copy()
    prepared["ts"] = pd.to_datetime(prepared["ts"], utc=True, errors="coerce")
    for column in expected[1:]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared["ts"].isna().any() or prepared["ts"].duplicated().any():
        raise RuntimeError(f"HVCVTP {label} invalid or duplicate timestamp")
    prices = prepared[["open", "high", "low", "close"]]
    prepared["row_valid"] = (
        np.isfinite(prepared[expected[1:]]).all(axis=1)
        & prices.gt(0).all(axis=1)
        & prepared["quote_asset_volume"].ge(0)
        & prepared["high"].ge(prepared[["open", "close"]].max(axis=1))
        & prepared["low"].le(prepared[["open", "close"]].min(axis=1))
        & prepared["high"].ge(prepared["low"])
    )
    return prepared.set_index("ts").sort_index()


def immediate_prior_onset(eligible: pd.Series, source_valid: pd.Series) -> pd.Series:
    """Require the immediately previous scheduled decision to be valid and ineligible."""
    valid = source_valid.eq(True)
    state = eligible.eq(True)
    return state & valid & valid.shift(1, fill_value=False) & ~state.shift(1, fill_value=False)


def _block_metrics(spot: pd.DataFrame, perpetual: pd.DataFrame) -> dict[str, float]:
    spot_weights = spot["quote_asset_volume"].to_numpy(float)
    perpetual_weights = perpetual["quote_asset_volume"].to_numpy(float)
    spot_median = weighted_median_minute(spot_weights)
    perpetual_median = weighted_median_minute(perpetual_weights)
    spot_mean = weighted_mean_minute(spot_weights)
    perpetual_mean = weighted_mean_minute(perpetual_weights)
    spot_return = float(math.log(spot["close"].iloc[-1] / spot["open"].iloc[0]))
    perpetual_return = float(
        math.log(perpetual["close"].iloc[-1] / perpetual["open"].iloc[0])
    )
    spot_final = float(math.log(spot["close"].iloc[-1] / spot["open"].iloc[-120]))
    perpetual_final = float(
        math.log(perpetual["close"].iloc[-1] / perpetual["open"].iloc[-120])
    )
    minute_returns = np.log(
        perpetual["close"].to_numpy(float) / perpetual["open"].to_numpy(float)
    )
    variation = float(np.sqrt(np.square(minute_returns).sum()))
    signs = np.sign([spot_return, perpetual_return, spot_final, perpetual_final])
    direction = float(signs[0]) if np.all(signs == signs[0]) and signs[0] != 0 else 0.0
    return {
        "spot_weighted_median_minute": spot_median,
        "perpetual_weighted_median_minute": perpetual_median,
        "cash_precedence": perpetual_median - spot_median,
        "spot_weighted_mean_minute": spot_mean,
        "perpetual_weighted_mean_minute": perpetual_mean,
        "mean_arrival_precedence": perpetual_mean - spot_mean,
        "spot_return": spot_return,
        "perpetual_return": perpetual_return,
        "spot_final_two_hour_return": spot_final,
        "perpetual_final_two_hour_return": perpetual_final,
        "direction_side": direction,
        "perpetual_variation": variation,
    }


def build_panel(spot_raw: pd.DataFrame, perpetual_raw: pd.DataFrame) -> pd.DataFrame:
    spot = prepare(spot_raw, "spot")
    perpetual = prepare(perpetual_raw, "perpetual")
    rows: list[dict[str, Any]] = []
    first_decision = START + pd.Timedelta(hours=4)
    for decision in pd.date_range(first_decision, END, freq="8h", inclusive="left"):
        minute_grid = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left"
        )
        spot_block = spot.reindex(minute_grid)
        perpetual_block = perpetual.reindex(minute_grid)
        spot_count = int(spot_block["row_valid"].eq(True).sum())
        perpetual_count = int(perpetual_block["row_valid"].eq(True).sum())
        source_valid = bool(
            len(spot_block) == 480
            and len(perpetual_block) == 480
            and spot_block["row_valid"].eq(True).all()
            and perpetual_block["row_valid"].eq(True).all()
        )
        metrics = {column: math.nan for column in PANEL_COLUMNS[5:19]}
        if source_valid:
            metrics = _block_metrics(spot_block, perpetual_block)
            source_valid = bool(
                np.isfinite(list(metrics.values())).all()
                and metrics["perpetual_variation"] > 0
            )
        rows.append(
            {
                "decision_time": decision,
                "feature_available_time": decision,
                "source_valid": source_valid,
                "spot_minute_count": spot_count,
                "perpetual_minute_count": perpetual_count,
                **metrics,
            }
        )
    panel = pd.DataFrame(rows)
    valid = panel["source_valid"].eq(True)
    positive_precedence = valid & panel["cash_precedence"].gt(0)
    panel["precedence_rank"] = strict_prior_midrank(
        panel["cash_precedence"].where(positive_precedence)
    )
    panel["variation_rank"] = strict_prior_midrank(
        panel["perpetual_variation"].where(valid)
    )
    panel["eligible"] = (
        valid
        & panel["cash_precedence"].gt(0)
        & panel["precedence_rank"].ge(POLICY["precedence_rank_min"])
        & panel["variation_rank"].ge(POLICY["variation_rank_min"])
        & panel["direction_side"].ne(0)
    )
    panel["onset"] = immediate_prior_onset(panel["eligible"], valid)
    return panel.loc[:, PANEL_COLUMNS]


def active(
    panel: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVCVTP control: {control}")
    used = panel.copy()
    if control == "one_decision_stale_precedence":
        columns = [
            "spot_weighted_median_minute", "perpetual_weighted_median_minute",
            "cash_precedence", "precedence_rank", "spot_weighted_mean_minute",
            "perpetual_weighted_mean_minute", "mean_arrival_precedence",
        ]
        used[columns] = panel[columns].shift(1)
        used["feature_available_time"] = panel["feature_available_time"].shift(1)
    valid = used["source_valid"].eq(True)
    precedence = (
        used["mean_arrival_precedence"]
        if control == "mean_arrival_precedence"
        else used["cash_precedence"]
    )
    precedence_gate = precedence.gt(0)
    if control != "no_precedence_tail":
        precedence_gate &= used["precedence_rank"].ge(POLICY["precedence_rank_min"])
    variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_variation_gate"
        else used["variation_rank"].ge(POLICY["variation_rank_min"])
    )
    eligible = valid & precedence_gate & variation_gate & used["direction_side"].ne(0)
    onset = immediate_prior_onset(eligible, valid)
    side = pd.to_numeric(used["direction_side"], errors="coerce").fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return onset & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, side, used = active(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[onset]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {
                "candidate": prereg.build()["policy_id"],
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{column: float(used.at[index, column]) for column in CLOCK_COLUMNS[8:]},
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = pd.to_datetime(selected["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def deterministic_gzip(frame: pd.DataFrame) -> bytes:
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as stream:
        stream.write(raw)
    return buffer.getvalue()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite immutable artifact {path}")
    path.write_bytes(payload)


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCVTP preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVCVTP preregistration payload drift")
    spot, perpetual = load_sources()
    panel = build_panel(spot, perpetual)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary["split"].eq(name)].copy() for name in SPLITS}

    immutable_write(PANEL, deterministic_gzip(panel))
    immutable_write(CLOCK, deterministic_gzip(primary))
    for name, frame in controls.items():
        immutable_write(CONTROL_DIR / f"{name}.csv.gz", deterministic_gzip(frame))
    for name, frame in splits.items():
        immutable_write(SPLIT_DIR / f"{name}.csv.gz", deterministic_gzip(frame))

    source_core = {
        "protocol_version": "hvcvtp_8_source_v1",
        "queries": QUERIES,
        "query_sha256": {
            name: hashlib.sha256(query.encode()).hexdigest() for name, query in QUERIES.items()
        },
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {"spot": len(spot), "perpetual": len(perpetual)},
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
        "panel": {
            "path": str(PANEL), "sha256": sha256(PANEL), "rows": len(panel),
            "valid_rows": int(panel["source_valid"].sum()),
        },
        "outcomes_opened": False,
        "execution_prices_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable_write(MANIFEST, canonical_json_bytes(source_manifest))

    support = {name: support_stats(primary, name) for name in SPLITS}
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
        "protocol_version": "hvcvtp_8_source_support_v1",
        "policy_id": registration["policy_id"],
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
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_artifacts": {
            name: {
                "path": str(SPLIT_DIR / f"{name}.csv.gz"),
                "sha256": sha256(SPLIT_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
            }
            for name, frame in splits.items()
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
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
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable_write(RESULT, canonical_json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "result": str(RESULT)}, ensure_ascii=False))
