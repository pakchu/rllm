"""Source-only support evaluator for frozen HVHCEM-12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_high_volatility_haar_coarse_energy_migration_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "524df960b81c1791ea32b97521cbd17db0e3c8c008bc7f58469f309d68ac4900"
DECISION_START = pd.Timestamp("2023-02-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
QUERY_START = DECISION_START - pd.Timedelta(minutes=640)
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_migration_gate",
    "no_volatility_gate",
    "coarse_share_level",
    "one_boundary_stale_migration",
    "direction_flip",
)
ROOT = Path("data/high_volatility_haar_coarse_energy_migration_continuation_sources_2023_2026")
SNAPSHOT = ROOT / "states.csv.gz"
SOURCE_MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_haar_coarse_energy_migration_continuation_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_haar_coarse_energy_migration_continuation_controls_2023_2026")
RESULT = Path("results/high_volatility_haar_coarse_energy_migration_continuation_support_2026-08-10.json")
QUERY = """SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "completed_return",
    "realized_variation",
    "coarse_energy_share",
    "coarse_energy_migration",
    "variation_rank",
    "migration_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def haar_pyramid(values: Iterable[float]) -> tuple[list[float], float]:
    """Return level 1..7 detail energies and the final approximation coefficient."""
    approximation = [float(value) for value in values]
    if len(approximation) != 128 or not all(math.isfinite(value) for value in approximation):
        raise ValueError("Haar input must contain exactly 128 finite values")
    energies: list[float] = []
    scale = math.sqrt(2.0)
    while len(approximation) > 1:
        next_approximation: list[float] = []
        energy = 0.0
        for position in range(0, len(approximation), 2):
            left, right = approximation[position], approximation[position + 1]
            next_approximation.append((left + right) / scale)
            detail = (left - right) / scale
            energy += detail * detail
        energies.append(energy)
        approximation = next_approximation
    return energies, approximation[0]


def haar_detail_energies(values: Iterable[float]) -> list[float]:
    return haar_pyramid(values)[0]


def strict_prior_midrank(
    values: pd.Series | Iterable[float], lookback: int = 270, minimum: int = 252
) -> pd.Series:
    """Causal midrank over finite prior observations; the current value is excluded."""
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = float(
                (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
            )
        if math.isfinite(current):
            history.append(float(current))
    return output


def _coherent(window: pd.DataFrame) -> bool:
    if len(window) != 640 or window.isna().any().any():
        return False
    values = window[["open", "high", "low", "close"]]
    return bool(
        np.isfinite(values.to_numpy(float)).all()
        and values.gt(0).all().all()
        and window["high"].ge(window[["open", "close"]].max(axis=1)).all()
        and window["low"].le(window[["open", "close"]].min(axis=1)).all()
        and window["high"].ge(window["low"]).all()
    )


def score_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Build causal path states from exact completed one-minute OHLC bars."""
    frame = bars.copy()
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame = frame.set_index("ts")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.duplicated().any():
        raise RuntimeError("duplicate HVHCEM source timestamps")
    frame = frame.sort_index()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    previous_valid_share: float | None = None
    decisions = pd.date_range(DECISION_START, END, freq="12h", inclusive="left")
    for decision in decisions:
        expected = pd.date_range(
            decision - pd.Timedelta(minutes=640), decision, freq="1min", inclusive="left"
        )
        window = frame.reindex(expected)
        valid = _coherent(window)
        energies = [math.nan] * 7
        completed_return = variation = share = migration = math.nan
        if valid:
            opens = window["open"].to_numpy(float)[::5]
            closes = window["close"].to_numpy(float)[4::5]
            returns = np.log(closes / opens)
            valid = len(returns) == 128 and np.isfinite(returns).all()
            if valid:
                energies = haar_detail_energies(returns)
                total_energy = float(sum(energies))
                valid = total_energy > 0 and math.isfinite(total_energy)
            if valid:
                completed_return = float(returns.sum())
                variation = float(math.sqrt(float(np.square(returns).sum())))
                share = float(sum(energies[4:7]) / total_energy)
                migration = (
                    math.nan if previous_valid_share is None else share - previous_valid_share
                )
                previous_valid_share = share
            else:
                energies = [math.nan] * 7
        rows.append(
            {
                "decision_time": decision,
                "feature_available_time": decision,
                "source_valid": bool(valid),
                "valid_minute_count": 640 if valid else int(window.notna().all(axis=1).sum()),
                "completed_return": completed_return,
                "realized_variation": variation,
                **{f"detail_energy_level_{level}": energies[level - 1] for level in range(1, 8)},
                "coarse_energy_share": share,
                "coarse_energy_migration": migration,
            }
        )
    states = pd.DataFrame(rows)
    valid = states["source_valid"].astype(bool)
    states["variation_rank"] = strict_prior_midrank(states["realized_variation"].where(valid))
    states["migration_rank"] = strict_prior_midrank(
        states["coarse_energy_migration"].where(valid)
    )
    states["coarse_share_rank"] = strict_prior_midrank(states["coarse_energy_share"].where(valid))
    return states


def conditions(states: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control != "primary" and control not in CONTROLS:
        raise ValueError(f"unknown HVHCEM control: {control}")
    migration = states["coarse_energy_migration"]
    migration_rank = states["migration_rank"]
    if control == "one_boundary_stale_migration":
        migration = migration.shift(1)
        migration_rank = migration_rank.shift(1)
    if control == "no_migration_gate":
        migration_gate = pd.Series(True, index=states.index)
    elif control == "coarse_share_level":
        migration_gate = states["coarse_share_rank"].ge(0.75)
    else:
        migration_gate = migration.gt(0) & migration_rank.ge(0.75)
    variation_gate = (
        pd.Series(True, index=states.index)
        if control == "no_volatility_gate"
        else states["variation_rank"].ge(0.65)
    )
    active = (
        states["source_valid"].fillna(False).astype(bool)
        & np.isfinite(states["completed_return"])
        & states["completed_return"].ne(0)
        & migration_gate
        & variation_gate
    )
    side = np.sign(states["completed_return"])
    if control == "direction_flip":
        side = -side
    return active, side


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(states, control)
    rows: list[dict[str, Any]] = []
    next_available: pd.Timestamp | None = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        if decision.minute != 0 or decision.second != 0 or decision.hour not in (0, 12):
            continue
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_available is not None and entry < next_available:
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
        next_available = exit_time
        rows.append(
            {
                "candidate": "HVHCEM-12",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "completed_return": float(states.at[index, "completed_return"]),
                "realized_variation": float(states.at[index, "realized_variation"]),
                "coarse_energy_share": float(states.at[index, "coarse_energy_share"]),
                "coarse_energy_migration": float(states.at[index, "coarse_energy_migration"]),
                "variation_rank": float(states.at[index, "variation_rank"]),
                "migration_rank": float(states.at[index, "migration_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
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


def engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    env_file = "/home/pakchu/rllm/.env"
    load_env_file(env_file)
    return create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = engine()
    try:
        with database.connect() as connection:
            bars = pd.read_sql_query(
                text(QUERY),
                connection,
                params={"start": QUERY_START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()
    states = score_bars(bars)
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, SNAPSHOT)
    source_core = {
        "protocol_version": "hvhcem_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "columns": ["ts", "open", "high", "low", "close"],
        "query_window": [QUERY_START.isoformat(), END.isoformat()],
        "decision_window": [DECISION_START.isoformat(), END.isoformat()],
        "outcomes_opened": False,
        "execution_prices_opened": False,
        "funding_oi_premium_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
        "snapshot": {
            "path": str(SNAPSHOT),
            "sha256": sha256(SNAPSHOT),
            "rows": len(states),
            "valid_rows": int(states["source_valid"].sum()),
        },
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return states, manifest


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVHCEM preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    states, source_manifest = materialize()
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM_EVENTS[name]),
            (f"{name}_side_balance", item["minority_side_share"] >= 0.20),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvhcem_12_source_support_v1",
        "policy_id": "HVHCEM-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "ranking": {
            "lookback_boundaries": 270,
            "minimum_prior_boundaries": 252,
            "current_excluded": True,
        },
        "completed_preentry_sources_opened": True,
        "outcomes_opened": False,
        "execution_prices_opened": False,
        "funding_oi_premium_opened": False,
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


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
