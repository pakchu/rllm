"""Materialize source-only HVFXTE-12 support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_fx_transfer_entropy_network_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "6053ab00cf402534e49a26ed7c553bc8e1f8d6ae32a7adf6d2c5e9d1d3ee29c1"
SOURCE_DIR = Path("data/high_volatility_fx_transfer_entropy_network_relay_sources_2023_2026")
SESSIONS = SOURCE_DIR / "network_sessions.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_fx_transfer_entropy_network_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_fx_transfer_entropy_network_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_fx_transfer_entropy_network_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_fx_transfer_entropy_network_relay_support_2026-08-13.json")
BTC_HOURLY = Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz")
BTC_HOURLY_SHA = "f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
BTC_MANIFEST = BTC_HOURLY.parent / "manifest.json"
BTC_MANIFEST_SHA = "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
SYMBOLS = ("EURUSD", "GBPUSD", "USDAUD", "USDCAD", "USDCHF", "USDJPY")
MULTIPLIER = {"EURUSD": -1.0, "GBPUSD": -1.0, "USDAUD": 1.0, "USDCAD": 1.0, "USDCHF": 1.0, "USDJPY": 1.0}
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_source_strength_tail", "no_variation_gate", "no_breadth_gate",
    "linear_lag_network", "one_session_stale_network", "direction_flip", "same_clock_forced_long",
)
QUERY = """SELECT ts,symbol,open,high,low,close FROM bars_polygon
WHERE symbol IN ('EURUSD','GBPUSD','USDAUD','USDCAD','USDCHF','USDJPY')
AND interval='1m' AND ts>=:start AND ts<:end
AND extract(isodow from ts) BETWEEN 1 AND 5 AND extract(hour from ts)>=13 AND extract(hour from ts)<21
ORDER BY ts,symbol"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "source_node", "source_strength", "source_strength_rank",
    "positive_outgoing_edges", "source_direction", "btc_realized_variation", "btc_variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            result.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return result


def transfer_entropy(source: pd.Series, target: pd.Series) -> tuple[float, int, int]:
    triples = pd.concat(
        [source.shift(1).rename("x"), target.rename("y"), target.shift(1).rename("z")],
        axis=1,
    ).dropna()
    if len(triples) < 360:
        raise ValueError("insufficient FX transfer-entropy transitions")
    values = triples.astype(int).to_numpy()
    if not np.isin(values, [-1, 1]).all():
        raise ValueError("invalid FX sign state")
    values = (values > 0).astype(int)
    counts = np.zeros((2, 2, 2), dtype=int)
    for x, y, z in values:
        counts[x, y, z] += 1
    minimum_cell = int(counts.sum(axis=1).min())
    if minimum_cell < 20:
        raise ValueError("insufficient FX transfer-entropy conditioning cell")
    total = float(len(values))
    information = 0.0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                n_xyz = counts[x, y, z]
                if not n_xyz:
                    continue
                n_z = counts[:, :, z].sum()
                n_xz = counts[x, :, z].sum()
                n_yz = counts[:, y, z].sum()
                information += (n_xyz / total) * math.log((n_xyz * n_z) / (n_xz * n_yz))
    if not math.isfinite(information):
        raise ValueError("nonfinite FX transfer entropy")
    return float(max(information, 0.0)), len(values), minimum_cell


def transfer_entropy_network(signs: pd.DataFrame) -> tuple[str, float, int, dict[str, float]]:
    edges = {symbol: {} for symbol in SYMBOLS}
    for left_index, left in enumerate(SYMBOLS):
        for right in SYMBOLS[left_index + 1:]:
            forward, _, _ = transfer_entropy(signs[left], signs[right])
            reverse, _, _ = transfer_entropy(signs[right], signs[left])
            edge = forward - reverse
            if not math.isfinite(edge):
                raise ValueError("nonfinite FX transfer-entropy edge")
            edges[left][right] = edge
            edges[right][left] = -edge
    scores = {symbol: float(sum(edges[symbol].values())) for symbol in SYMBOLS}
    maximum = max(scores.values())
    leaders = [symbol for symbol, score in scores.items() if score == maximum]
    if len(leaders) != 1 or maximum <= 0:
        raise ValueError("directed FX source is not unique and positive")
    source = leaders[0]
    breadth = sum(value > 0 for value in edges[source].values())
    return source, maximum, breadth, scores


def linear_lag_network(returns: pd.DataFrame) -> tuple[str, float, int]:
    edges = {symbol: {} for symbol in SYMBOLS}
    for left_index, left in enumerate(SYMBOLS):
        for right in SYMBOLS[left_index + 1:]:
            pair = pd.concat(
                [returns[left].shift(1).rename("left"), returns[right].rename("right")], axis=1
            ).dropna()
            reverse_pair = pd.concat(
                [returns[right].shift(1).rename("right"), returns[left].rename("left")], axis=1
            ).dropna()
            if len(pair) < 360 or len(reverse_pair) < 360:
                raise ValueError("insufficient linear FX lag transitions")
            forward = float(pair.corr().iloc[0, 1])
            reverse = float(reverse_pair.corr().iloc[0, 1])
            edge = forward - reverse
            if not math.isfinite(edge):
                raise ValueError("nonfinite linear FX lag edge")
            edges[left][right] = edge
            edges[right][left] = -edge
    scores = {symbol: float(sum(edges[symbol].values())) for symbol in SYMBOLS}
    maximum = max(scores.values())
    leaders = [symbol for symbol, score in scores.items() if score == maximum]
    if len(leaders) != 1 or maximum <= 0:
        raise ValueError("linear FX source is not unique and positive")
    source = leaders[0]
    return source, maximum, sum(value > 0 for value in edges[source].values())


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize_sessions() -> dict[str, Any]:
    from sqlalchemy import text
    engine = postgres_engine()
    with engine.connect() as connection:
        raw = pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    engine.dispose()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw["source_day"] = raw["ts"].dt.floor("D")
    for column in ("open", "high", "low", "close"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw.duplicated(["ts", "symbol"]).any():
        raise RuntimeError("duplicate HVFXTE source timestamp")
    rows: list[dict[str, Any]] = []
    for source_day, daily in raw.groupby("source_day", sort=True):
        closes: dict[str, pd.Series] = {}
        valid = True
        for symbol in SYMBOLS:
            frame = daily[daily["symbol"].eq(symbol)].sort_values("ts")
            valid_symbol = (
                len(frame) >= 420 and frame["ts"].nunique() >= 420
                and frame["ts"].min() <= source_day + pd.Timedelta(hours=13, minutes=5)
                and frame["ts"].max() >= source_day + pd.Timedelta(hours=20, minutes=55)
                and np.isfinite(frame[["open", "high", "low", "close"]]).all().all()
                and frame[["open", "high", "low", "close"]].gt(0).all().all()
                and frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()
                and frame["low"].le(frame[["open", "close"]].min(axis=1)).all()
            )
            valid &= bool(valid_symbol)
            if valid_symbol:
                closes[symbol] = frame.set_index("ts")["close"].astype(float)
        if valid:
            try:
                common = pd.concat(closes, axis=1, join="inner").sort_index()
                if (
                    len(common) < 420
                    or common.index[0] > source_day + pd.Timedelta(hours=13, minutes=5)
                    or common.index[-1] < source_day + pd.Timedelta(hours=20, minutes=55)
                ):
                    raise ValueError("insufficient common FX path")
                consecutive = common.index.to_series().diff().eq(pd.Timedelta(minutes=1)).to_numpy()
                returns = np.log(common / common.shift(1))
                for symbol in SYMBOLS:
                    returns[symbol] *= MULTIPLIER[symbol]
                returns.loc[~consecutive, :] = np.nan
                signs = np.sign(returns).replace(0, np.nan)
                source, strength, breadth, _ = transfer_entropy_network(signs)
                linear_source, linear_strength, linear_breadth = linear_lag_network(returns)
                direction = float(MULTIPLIER[source] * np.log(common[source].iloc[-1] / common[source].iloc[0]))
                linear_direction = float(
                    MULTIPLIER[linear_source] * np.log(common[linear_source].iloc[-1] / common[linear_source].iloc[0])
                )
                valid = all(
                    math.isfinite(value) and value != 0
                    for value in (direction, linear_direction, strength, linear_strength)
                )
            except ValueError:
                valid = False
        if not valid:
            source, strength, breadth, direction = "", math.nan, 0, math.nan
            linear_source, linear_strength, linear_breadth, linear_direction = "", math.nan, 0, math.nan
        rows.append({
            "source_day": source_day, "decision_time": source_day + pd.Timedelta(hours=21),
            "source_valid": valid, "source_node": source, "source_strength": strength,
            "positive_outgoing_edges": breadth, "source_direction": direction,
            "linear_source_node": linear_source, "linear_source_strength": linear_strength,
            "linear_positive_outgoing_edges": linear_breadth, "linear_source_direction": linear_direction,
        })
    sessions = pd.DataFrame(rows).sort_values("source_day").reset_index(drop=True)
    sessions["source_strength_rank"] = strict_prior_midrank(sessions["source_strength"].where(sessions["source_valid"]))
    sessions["linear_source_strength_rank"] = strict_prior_midrank(
        sessions["linear_source_strength"].where(sessions["source_valid"])
    )
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(sessions, SESSIONS)
    core = {
        "protocol_version": "hvfxte_12_transfer_entropy_fx_source_v1", "query": QUERY,
        "table": "bars_polygon", "symbols": list(SYMBOLS), "canonical_dollar_multipliers": MULTIPLIER,
        "interval": "1m", "window": [START.isoformat(), END.isoformat()], "outcomes_opened": False,
        "candidate_incidence_opened": True, "no_imputation": True,
        "output": {"path": str(SESSIONS), "sha256": sha256(SESSIONS), "rows": len(sessions), "valid_rows": int(sessions["source_valid"].sum())},
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def load_features() -> pd.DataFrame:
    if sha256(BTC_HOURLY) != BTC_HOURLY_SHA or sha256(BTC_MANIFEST) != BTC_MANIFEST_SHA:
        raise RuntimeError("HVFXTE BTC source drift")
    sessions = pd.read_csv(SESSIONS, compression="gzip")
    sessions["source_day"] = pd.to_datetime(sessions["source_day"], utc=True)
    sessions["decision_time"] = pd.to_datetime(sessions["decision_time"], utc=True)
    sessions["source_valid"] = sessions["source_valid"].astype(str).str.lower().eq("true")
    for column in (
        "source_strength", "source_strength_rank", "positive_outgoing_edges", "source_direction",
        "linear_source_strength", "linear_source_strength_rank", "linear_positive_outgoing_edges",
        "linear_source_direction",
    ):
        sessions[column] = pd.to_numeric(sessions[column], errors="coerce")
    btc = pd.read_csv(BTC_HOURLY, compression="gzip")
    btc["decision_time"] = pd.to_datetime(btc["decision_time"], utc=True, format="mixed")
    btc["open"] = pd.to_numeric(btc["open"], errors="coerce")
    btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
    btc["valid"] = btc["source_valid"].astype(str).str.lower().eq("true") & np.isfinite(btc[["open", "close"]]).all(axis=1) & btc[["open", "close"]].gt(0).all(axis=1)
    btc = btc.sort_values("decision_time").reset_index(drop=True)
    btc["hour_return"] = np.log(btc["close"] / btc["open"])
    consecutive = btc["decision_time"].diff().eq(pd.Timedelta(hours=1))
    btc["btc_realized_variation"] = np.sqrt(btc["hour_return"].pow(2).rolling(24, min_periods=24).sum())
    btc["btc_valid"] = btc["valid"].rolling(24, min_periods=24).sum().eq(24) & consecutive.rolling(23, min_periods=23).sum().eq(23)
    sessions = sessions.merge(btc[["decision_time", "btc_realized_variation", "btc_valid"]], on="decision_time", how="left", validate="one_to_one")
    joint_valid = sessions["source_valid"] & sessions["btc_valid"].fillna(False)
    sessions["btc_variation_rank"] = strict_prior_midrank(
        sessions["btc_realized_variation"].where(joint_valid)
    )
    sessions["signal_valid"] = (
        joint_valid
        & np.isfinite(
            sessions[
                [
                    "source_strength", "source_strength_rank", "source_direction",
                    "linear_source_strength", "linear_source_strength_rank", "linear_source_direction",
                    "btc_realized_variation", "btc_variation_rank",
                ]
            ]
        ).all(axis=1)
        & sessions["source_direction"].ne(0)
        & sessions["linear_source_direction"].ne(0)
    )
    return sessions


def onset_after_previous_source_valid(source_valid: pd.Series, eligible: pd.Series) -> pd.Series:
    result = pd.Series(False, index=eligible.index)
    previous_eligible = False
    for index in eligible.index:
        if not bool(source_valid.at[index]):
            continue
        current = bool(eligible.at[index])
        result.at[index] = current and not previous_eligible
        previous_eligible = current
    return result


def active_and_side(features: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    strength_rank = features["source_strength_rank"]
    breadth = features["positive_outgoing_edges"]
    direction = features["source_direction"]
    variation_rank = features["btc_variation_rank"]
    if control == "linear_lag_network":
        strength_rank = features["linear_source_strength_rank"]
        breadth = features["linear_positive_outgoing_edges"]
        direction = features["linear_source_direction"]
    if control == "one_session_stale_network":
        strength_rank, breadth, direction = strength_rank.shift(1), breadth.shift(1), direction.shift(1)
    strength_gate = pd.Series(True, index=features.index) if control == "no_source_strength_tail" else strength_rank.ge(0.75)
    breadth_gate = pd.Series(True, index=features.index) if control == "no_breadth_gate" else breadth.ge(4)
    variation_gate = pd.Series(True, index=features.index) if control == "no_variation_gate" else variation_rank.ge(0.65)
    eligible = features["signal_valid"] & np.isfinite(direction) & direction.ne(0) & strength_gate & breadth_gate & variation_gate
    active = onset_after_previous_source_valid(features["source_valid"], eligible)
    side = -np.sign(direction)
    if control == "direction_flip": side = -side
    elif control == "same_clock_forced_long": side = pd.Series(1.0, index=features.index)
    return active, side


def make_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = active_and_side(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time
        linear = control == "linear_lag_network"
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "source_day": features.at[index, "source_day"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]),
            "source_node": features.at[index, "linear_source_node" if linear else "source_node"],
            "source_strength": float(features.at[index, "linear_source_strength" if linear else "source_strength"]),
            "source_strength_rank": float(features.at[index, "linear_source_strength_rank" if linear else "source_strength_rank"]),
            "positive_outgoing_edges": int(features.at[index, "linear_positive_outgoing_edges" if linear else "positive_outgoing_edges"]),
            "source_direction": float(features.at[index, "linear_source_direction" if linear else "source_direction"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame = clock[clock["split"].eq(split)]
    if frame.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(frame["side"].eq(1).sum()), int(frame["side"].eq(-1).sum())
    months = frame["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {"events": len(frame), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(frame), "max_month_share": int(months.max()) / len(frame)}


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVFXTE preregistration hash drift")
    source_manifest = materialize_sessions()
    features = load_features()
    primary = make_clock(features)
    controls = {name: make_clock(features, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); SPLIT_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for split in SPLITS:
        _write_gzip_csv(primary[primary["split"].eq(split)].reset_index(drop=True), SPLIT_DIR / f"{split}.csv.gz")
    for name, frame in controls.items(): _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, metrics in support.items():
        checks[f"{name}_minimum_events"] = metrics["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = metrics["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = metrics["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); passed = all(checks.values())
    core = {
        "protocol_version": "hvfxte_12_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifests": {
            "transfer_entropy_fx": {"path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
            "completed_btc": {"path": str(BTC_MANIFEST), "sha256": sha256(BTC_MANIFEST)},
        },
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_clocks": {
            split: {"path": str(SPLIT_DIR / f"{split}.csv.gz"), "sha256": sha256(SPLIT_DIR / f"{split}.csv.gz"), "rows": int(primary["split"].eq(split).sum())}
            for split in SPLITS
        },
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
