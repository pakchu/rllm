"""Source-only support evaluator for frozen HVVWMMR-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_volume_weighted_median_migration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "14ec00d3055ce4d3ec2a78d4493b95d00d91c8d3190638542fb3a051956874c5"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_migration_rank_gate",
    "no_volatility_gate",
    "arithmetic_vwap_migration",
    "one_boundary_stale_migration",
    "direction_flip",
)
ROOT = Path("data/high_volatility_volume_weighted_median_migration_relay_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_volume_weighted_median_migration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_volume_weighted_median_migration_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_volume_weighted_median_migration_relay_support_2026-08-10.json")
QUERY = """SELECT ts,open,high,low,close,volume,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "first_half_weighted_median",
    "second_half_weighted_median", "value_migration", "absolute_migration_rank",
    "first_half_arithmetic_vwap", "second_half_arithmetic_vwap",
    "arithmetic_vwap_migration", "realized_variation", "variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    series: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    """Rank each finite value against finite prior values only; exclude current."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(value) and len(prior) >= minimum:
            output[index] = (
                np.sum(prior < value) + 0.5 * np.sum(prior == value)
            ) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index, dtype=float)


def quote_weighted_median(prices: np.ndarray, quote_weights: np.ndarray) -> float:
    """Return the frozen lower crossing of half quote weight after stable price sort."""
    prices = np.asarray(prices, dtype=float)
    quote_weights = np.asarray(quote_weights, dtype=float)
    if (
        prices.ndim != 1
        or quote_weights.ndim != 1
        or len(prices) != len(quote_weights)
        or not len(prices)
        or not np.isfinite(prices).all()
        or not np.isfinite(quote_weights).all()
        or np.any(prices <= 0)
        or np.any(quote_weights <= 0)
    ):
        return math.nan
    order = np.argsort(prices, kind="stable")
    sorted_prices = prices[order]
    sorted_weights = quote_weights[order]
    total = float(sorted_weights.sum())
    if not math.isfinite(total) or total <= 0:
        return math.nan
    crossing = int(np.searchsorted(np.cumsum(sorted_weights), total / 2.0, side="left"))
    return float(sorted_prices[crossing])


def _valid_rows(frame: pd.DataFrame) -> pd.Series:
    columns = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
    finite = pd.Series(np.isfinite(frame[columns]).all(axis=1), index=frame.index)
    positive = frame[columns].gt(0).all(axis=1)
    coherent = (
        frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame["low"])
    )
    return finite & positive & coherent


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "open", "high", "low", "close", "volume", "quote_asset_volume"}
    if set(bars.columns) != required:
        raise ValueError(f"HVVWMMR source schema must be exactly {sorted(required)}")
    market = bars.copy()
    market["ts"] = pd.to_datetime(market["ts"], utc=True, errors="coerce")
    for column in required - {"ts"}:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    if market["ts"].isna().any():
        raise ValueError("HVVWMMR source contains invalid timestamps")
    # Duplicates are retained as invalidity evidence: reindexing below must not hide them.
    if market["ts"].duplicated().any():
        duplicates = market.loc[market["ts"].duplicated(keep=False), "ts"].unique()
        market = market.loc[~market["ts"].isin(duplicates)]
    return market.set_index("ts").sort_index()


def build_states(
    bars: pd.DataFrame,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> pd.DataFrame:
    """Build frozen boundary features without opening execution prices or outcomes."""
    market = _prepare_bars(bars)
    rows: list[dict[str, Any]] = []
    first_decision = pd.Timestamp(start).ceil("8h")
    for decision in pd.date_range(first_decision, end, freq="8h", inclusive="left"):
        block_index = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left"
        )
        variation_index = pd.date_range(
            decision - pd.Timedelta(hours=24), decision, freq="1min", inclusive="left"
        )
        block = market.reindex(block_index)
        variation_window = market.reindex(variation_index)
        block_valid = bool(len(block) == 480 and _valid_rows(block).all())
        variation_valid = bool(
            len(variation_window) == 1440 and _valid_rows(variation_window).all()
        )

        first_median = second_median = migration = math.nan
        first_vwap = second_vwap = arithmetic_migration = math.nan
        if block_valid:
            prices = block["quote_asset_volume"].to_numpy(float) / block["volume"].to_numpy(float)
            quote = block["quote_asset_volume"].to_numpy(float)
            base = block["volume"].to_numpy(float)
            first_median = quote_weighted_median(prices[:240], quote[:240])
            second_median = quote_weighted_median(prices[240:], quote[240:])
            migration = float(math.log(second_median / first_median))
            quote_totals = (float(quote[:240].sum()), float(quote[240:].sum()))
            base_totals = (float(base[:240].sum()), float(base[240:].sum()))
            if all(math.isfinite(total) and total > 0 for total in quote_totals + base_totals):
                first_vwap = quote_totals[0] / base_totals[0]
                second_vwap = quote_totals[1] / base_totals[1]
                arithmetic_migration = float(math.log(second_vwap / first_vwap))
            block_valid = bool(math.isfinite(migration) and migration != 0)

        variation = math.nan
        if variation_valid:
            minute_returns = np.log(
                variation_window["close"].to_numpy(float)
                / variation_window["open"].to_numpy(float)
            )
            variation = float(np.sqrt(np.square(minute_returns).sum()))
            variation_valid = bool(math.isfinite(variation) and variation > 0)

        source_valid = block_valid and variation_valid
        rows.append({
            "decision_time": decision,
            "block_valid": block_valid,
            "variation_valid": variation_valid,
            "source_valid": source_valid,
            "first_half_weighted_median": first_median,
            "second_half_weighted_median": second_median,
            "value_migration": migration,
            "first_half_arithmetic_vwap": first_vwap,
            "second_half_arithmetic_vwap": second_vwap,
            "arithmetic_vwap_migration": arithmetic_migration,
            "realized_variation": variation,
        })
    states = pd.DataFrame(rows)
    valid = states["source_valid"]
    states["absolute_migration_rank"] = strict_prior_midrank(
        states["value_migration"].abs().where(valid)
    )
    states["variation_rank"] = strict_prior_midrank(
        states["realized_variation"].where(valid)
    )
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
        "protocol_version": "hvvwmmr_source_v1",
        "query": QUERY,
        "window": [START.isoformat(), END.isoformat()],
        "source_table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "outcomes_opened": False,
        "funding_opened": False,
        "gross9_rows_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {
            "path": str(PANEL), "sha256": sha256(PANEL), "rows": len(states),
            "valid_rows": int(states["source_valid"].sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return states, manifest


def control_features(states: pd.DataFrame, control: str) -> pd.DataFrame:
    """Select diagnostic geometry while preserving the frozen primary rank population."""
    if control != "primary" and control not in CONTROLS:
        raise ValueError(f"unknown HVVWMMR control: {control}")
    frame = states.copy()
    if control == "one_boundary_stale_migration":
        migration_columns = [
            "block_valid", "first_half_weighted_median", "second_half_weighted_median",
            "value_migration", "absolute_migration_rank", "first_half_arithmetic_vwap",
            "second_half_arithmetic_vwap", "arithmetic_vwap_migration",
        ]
        frame[migration_columns] = frame[migration_columns].shift(1)
    frame["control_source_valid"] = frame["block_valid"].eq(True) & frame["variation_valid"].eq(True)
    frame["control_migration"] = (
        frame["arithmetic_vwap_migration"]
        if control == "arithmetic_vwap_migration"
        else frame["value_migration"]
    )
    # Arithmetic VWAP is diagnostic geometry and direction only. Its eligibility tail
    # remains the preregistered primary weighted-median absolute-migration rank.
    return frame


def active(states: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    frame = control_features(states, control)
    migration_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_migration_rank_gate"
        else frame["absolute_migration_rank"].ge(0.75)
    )
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else frame["variation_rank"].ge(0.65)
    )
    migration = frame["control_migration"]
    eligible = (
        frame["control_source_valid"]
        & migration.notna()
        & migration.ne(0)
        & migration_gate
        & variation_gate
    )
    side = pd.Series(np.sign(migration), index=frame.index, dtype=float)
    if control == "direction_flip":
        side = -side
    return eligible, side


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = control_features(states, control)
    eligible, side = active(states, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[eligible]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": "HVVWMMR-8", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "first_half_weighted_median": float(frame.at[index, "first_half_weighted_median"]),
            "second_half_weighted_median": float(frame.at[index, "second_half_weighted_median"]),
            "value_migration": float(frame.at[index, "value_migration"]),
            "absolute_migration_rank": float(frame.at[index, "absolute_migration_rank"]),
            "first_half_arithmetic_vwap": float(frame.at[index, "first_half_arithmetic_vwap"]),
            "second_half_arithmetic_vwap": float(frame.at[index, "second_half_arithmetic_vwap"]),
            "arithmetic_vwap_migration": float(frame.at[index, "arithmetic_vwap_migration"]),
            "realized_variation": float(frame.at[index, "realized_variation"]),
            "variation_rank": float(frame.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(
            selected["entry_time"].dt.strftime("%Y-%m").value_counts().max()
        ) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVVWMMR preregistration drift")
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
        "protocol_version": "hvvwmmr_8_source_support_v1",
        "policy_id": "HVVWMMR-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST), "sha256": sha256(MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(value), "promotion_authorized": False,
            }
            for name, value in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
