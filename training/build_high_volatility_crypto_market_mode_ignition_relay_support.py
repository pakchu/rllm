"""Build outcome-blind source support for frozen HVCMMI-8."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_crypto_market_mode_ignition_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_crypto_market_mode_ignition_relay_support.py")
PREREG_SHA = "78c1a90c8aef0d67cb00f1951afb8fb1b49c427632b490b9cd37aa4dfc10e9e2"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
ALTS = tuple(prereg.ALTS)
SYMBOLS = ("BTCUSDT", *ALTS)
SPLITS = {name: tuple(map(pd.Timestamp, bounds)) for name, bounds in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,symbol,open,high,low,close
FROM bars_binance
WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path("data/high_volatility_crypto_market_mode_ignition_relay_sources_2023_2026")
BLOCK_PANEL = SOURCE_DIR / "block_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_crypto_market_mode_ignition_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_crypto_market_mode_ignition_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_crypto_market_mode_ignition_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_crypto_market_mode_ignition_relay_support_2026-08-10.json")

BLOCK_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "minute_count",
    "pc1_variance_share", "mode_rank", "mode_dominant", "mode_onset",
    "direction_score", "equal_weight_direction_score", "direction_side",
    "equal_weight_direction_side", "btc_variation", "btc_variation_rank",
    *(f"pc1_loading_{symbol}" for symbol in ALTS),
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "pc1_variance_share", "mode_rank",
    "mode_dominant", "mode_onset", "direction_score", "btc_variation",
    "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            result.at[index] = float((np.count_nonzero(array < current) + .5 * np.count_nonzero(array == current)) / len(array))
        if math.isfinite(current):
            history.append(float(current))
    return result


def orient_pc1(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[float, np.ndarray]:
    """Return frozen PC1 share and deterministically oriented loading."""
    values = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=float)
    if values.shape != (len(ALTS),) or vectors.shape != (len(ALTS), len(ALTS)):
        raise ValueError("HVCMMI eigensystem shape drift")
    if not np.isfinite(values).all() or not np.isfinite(vectors).all() or np.any(values < -1e-12):
        raise ValueError("HVCMMI invalid eigensystem")
    values = np.where(values < 0, 0.0, values)
    order = np.argsort(values)[::-1]
    values = values[order]
    loading = vectors[:, order[0]].copy()
    total = float(values.sum())
    if total <= 0:
        raise ValueError("HVCMMI nonpositive eigenvalue sum")
    loading_sum = float(loading.sum())
    if loading_sum < -1e-12:
        loading *= -1
    elif abs(loading_sum) <= 1e-12:
        first_maximum = int(np.flatnonzero(np.abs(loading) == np.abs(loading).max())[0])
        if loading[first_maximum] < 0:
            loading *= -1
    return float(values[0] / total), loading


def market_mode(alt_returns: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray(alt_returns, dtype=float)
    if values.shape != (POLICY["window_minutes"], len(ALTS)) or not np.isfinite(values).all():
        raise ValueError("HVCMMI invalid alt-return matrix")
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("HVCMMI nonpositive return scale")
    standardized = (values - means) / scales
    correlation = standardized.T @ standardized / len(values)
    correlation = (correlation + correlation.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    return orient_pc1(eigenvalues, eigenvectors)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> dict[str, pd.DataFrame]:
    """Read only the preregistered completed one-minute OHLC source fields."""
    from sqlalchemy import text
    engine = postgres_engine()
    frames: dict[str, pd.DataFrame] = {}
    try:
        with engine.connect() as connection:
            for symbol in SYMBOLS:
                frames[symbol] = pd.read_sql_query(text(QUERY), connection, params={
                    "symbol": symbol, "start": START.to_pydatetime(), "end": END.to_pydatetime(),
                })
    finally:
        engine.dispose()
    return frames


def prepare_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = ["ts", "symbol", "open", "high", "low", "close"]
    if frame.columns.tolist() != required:
        raise RuntimeError("HVCMMI source schema drift")
    data = frame.copy()
    data["ts"] = pd.to_datetime(data.ts, utc=True, errors="coerce")
    if data.ts.isna().any() or data.ts.duplicated().any() or not data.symbol.astype(str).eq(symbol).all():
        raise RuntimeError("HVCMMI invalid source key")
    for column in required[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    prices = data[required[2:]]
    data["row_valid"] = (
        np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & data.high.ge(prices[["open", "close"]].max(axis=1))
        & data.low.le(prices[["open", "close"]].min(axis=1)) & data.high.ge(data.low)
    )
    data["minute_return"] = np.log(data.close / data.open)
    return data.set_index("ts")[["row_valid", "minute_return"]].sort_index()


def _invalid_block(minute_count: int) -> dict[str, Any]:
    return {
        "source_valid": False, "minute_count": minute_count, "pc1_variance_share": np.nan,
        "direction_score": np.nan, "equal_weight_direction_score": np.nan,
        "direction_side": 0, "equal_weight_direction_side": 0, "btc_variation": np.nan,
        **{f"pc1_loading_{symbol}": np.nan for symbol in ALTS},
    }


def boundary_state(source: dict[str, pd.DataFrame], decision: pd.Timestamp) -> dict[str, Any]:
    minutes = pd.date_range(decision - pd.Timedelta(minutes=POLICY["window_minutes"]), decision, freq="1min", inclusive="left")
    blocks = {symbol: source[symbol].reindex(minutes) for symbol in SYMBOLS}
    minute_count = sum(int(block.row_valid.eq(True).sum()) for block in blocks.values())
    if len(minutes) != POLICY["window_minutes"] or not all(bool(block.row_valid.eq(True).all()) for block in blocks.values()):
        return _invalid_block(minute_count)
    alt_returns = np.column_stack([blocks[symbol].minute_return.to_numpy(float) for symbol in ALTS])
    btc_returns = blocks["BTCUSDT"].minute_return.to_numpy(float)
    try:
        share, loading = market_mode(alt_returns)
    except ValueError:
        return _invalid_block(minute_count)
    final_hour = alt_returns[-POLICY["direction_minutes"]:].sum(axis=0)
    score = float(loading @ final_hour)
    equal_score = float(final_hour.sum())
    variation = float(np.square(btc_returns).sum())
    if not np.isfinite([score, equal_score, variation]).all():
        return _invalid_block(minute_count)
    return {
        "source_valid": True, "minute_count": minute_count, "pc1_variance_share": share,
        "direction_score": score, "equal_weight_direction_score": equal_score,
        "direction_side": int(np.sign(score)), "equal_weight_direction_side": int(np.sign(equal_score)),
        "btc_variation": variation,
        **{f"pc1_loading_{symbol}": float(loading[index]) for index, symbol in enumerate(ALTS)},
    }


def source_valid_onset(dominant: pd.Series, source_valid: pd.Series) -> pd.Series:
    onset = pd.Series(False, index=dominant.index, dtype=bool)
    previous: int | None = None
    for index in dominant.index:
        if not bool(source_valid.at[index]):
            continue
        if bool(dominant.at[index]) and previous is not None:
            onset.at[index] = not bool(dominant.at[previous])
        previous = int(index)
    return onset


def build_block_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source = {symbol: prepare_symbol(raw[symbol], symbol) for symbol in SYMBOLS}
    rows = [{"decision_time": decision, "feature_available_time": decision, **boundary_state(source, decision)}
            for decision in pd.date_range(START + pd.Timedelta(hours=8), END, freq="8h", inclusive="left")]
    panel = pd.DataFrame(rows)
    valid = panel.source_valid.eq(True)
    panel["mode_rank"] = strict_prior_midrank(panel.pc1_variance_share.where(valid), POLICY["prior_blocks"], POLICY["minimum_prior_blocks"])
    panel["btc_variation_rank"] = strict_prior_midrank(panel.btc_variation.where(valid), POLICY["prior_blocks"], POLICY["minimum_prior_blocks"])
    panel["mode_dominant"] = valid & panel.mode_rank.ge(POLICY["mode_rank_min"])
    panel["mode_onset"] = source_valid_onset(panel.mode_dominant, valid)
    return panel.loc[:, BLOCK_COLUMNS]


def active_and_side(panel: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVCMMI control: {control}")
    used = panel.copy()
    if control == "one_block_stale_geometry":
        geometry = ["source_valid", "mode_rank", "mode_dominant", "mode_onset", "direction_score", "direction_side", "feature_available_time"]
        used.loc[:, geometry] = panel.loc[:, geometry].shift(1)
    valid = used.source_valid.eq(True)
    variation = panel.btc_variation_rank.ge(POLICY["btc_variation_rank_min"])
    event = valid & used.mode_onset.eq(True) & variation
    if control == "no_btc_variation_gate":
        event = valid & used.mode_onset.eq(True)
    elif control == "no_mode_onset":
        event = valid & used.mode_dominant.eq(True) & variation
    side = pd.to_numeric(used.direction_side, errors="coerce").fillna(0).astype(int)
    if control == "equal_weight_final_hour":
        side = pd.to_numeric(panel.equal_weight_direction_side, errors="coerce").fillna(0).astype(int)
    elif control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return event & side.ne(0), side, used


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = active_and_side(panel, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[active]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "pc1_variance_share": float(panel.at[index, "pc1_variance_share"]),
            "mode_rank": float(used.at[index, "mode_rank"]), "mode_dominant": bool(used.at[index, "mode_dominant"]),
            "mode_onset": bool(used.at[index, "mode_onset"]), "direction_score": float(used.at[index, "direction_score"]),
            "btc_variation": float(panel.at[index, "btc_variation"]), "btc_variation_rank": float(panel.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def deterministic_csv_gzip(frame: pd.DataFrame) -> bytes:
    content = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(content)
    return buffer.getvalue()


def deterministic_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable HVCMMI artifact: {path}")
        return
    path.write_bytes(content)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCMMI preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if registration != REGISTRATION or tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVCMMI committed preregistration drift")
    raw = load_source()
    panel = build_block_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    write_immutable(BLOCK_PANEL, deterministic_csv_gzip(panel))
    write_immutable(CLOCK, deterministic_csv_gzip(primary))
    for name, frame in splits.items():
        write_immutable(SPLIT_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))
    for name, frame in controls.items():
        write_immutable(CONTROL_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))
    rows = {symbol: len(frame) for symbol, frame in raw.items()}
    source_core = {
        "protocol_version": "hvcmmmi_8_source_v1", "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "query_execution": "one frozen-symbol parameter per query",
        "table": "bars_binance", "symbols": list(SYMBOLS), "interval": "1m",
        "columns": ["ts", "symbol", "open", "high", "low", "close"], "window": [START.isoformat(), END.isoformat()],
        "physical_rows_by_symbol": rows, "physical_rows": sum(rows.values()),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "block_panel": {"path": str(BLOCK_PANEL), "sha256": sha(BLOCK_PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False,
        "gross9_rows_opened": False, "no_imputation": True, "deterministic_immutable_artifacts": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    write_immutable(SOURCE_MANIFEST, deterministic_json(manifest))
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {check: passed for name, item in support.items() for check, passed in (
        (f"{name}_minimum_events", item["events"] >= GATES["minimum_events"][name]),
        (f"{name}_side_balance", item["minority_side_share"] >= GATES["minority_side_share_min"]),
        (f"{name}_month_concentration", item["max_month_share"] <= GATES["max_month_share"]),
    )}
    passed = all(checks.values())
    core = {
        "protocol_version": "hvcmmmi_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "ranking": {"lookback_valid_decisions": POLICY["prior_blocks"], "minimum_prior_valid_decisions": POLICY["minimum_prior_blocks"], "current_excluded": True, "ties": "midrank"},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in splits.items()},
        "reservation": {"scope": "global", "hours": POLICY["hold_hours"], "interval": "half_open", "equal_open_after_exit_allowed": True, "split_crossing_action": "skip"},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject", "deterministic_immutable_artifacts": True,
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    write_immutable(RESULT, deterministic_json(result))
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
