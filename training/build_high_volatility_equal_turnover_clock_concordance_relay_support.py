"""Source-only support gate for frozen HVETCC-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_equal_turnover_clock_concordance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
SOURCE_START = pd.Timestamp("2022-12-31T18:00:00Z")
DECISION_START = pd.Timestamp("2023-01-01T02:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "64a9e6099c1493ca8168928cde06f390cb9bed3cc609e0874568f6a1494e6bf2"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "three_of_four_concordance",
    "equal_physical_time",
    "one_decision_stale_geometry",
    "direction_flip",
    "same_clock_forced_long",
)
ROOT = Path("data/high_volatility_equal_turnover_clock_concordance_relay_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_equal_turnover_clock_concordance_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_equal_turnover_clock_concordance_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_equal_turnover_clock_concordance_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_equal_turnover_clock_concordance_relay_support_2026-08-13.json")
QUERY = """SELECT ts,open,high,low,close,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-270:], dtype=float)
        if math.isfinite(value) and len(prior) >= 180:
            output[index] = (
                np.sum(prior < value) + 0.5 * np.sum(prior == value)
            ) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def turnover_segments(
    quote_turnover: np.ndarray, minute_returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    turnover = np.asarray(quote_turnover, dtype=float)
    returns = np.asarray(minute_returns, dtype=float)
    if (
        len(turnover) != 480
        or len(returns) != 480
        or not np.isfinite(turnover).all()
        or not np.isfinite(returns).all()
        or np.any(turnover <= 0)
    ):
        return np.full(4, np.nan), np.zeros(4, dtype=int)
    total = float(turnover.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full(4, np.nan), np.zeros(4, dtype=int)
    segment_returns = np.zeros(4, dtype=float)
    counts = np.zeros(4, dtype=int)
    cumulative = 0.0
    for value, minute_return in zip(turnover, returns, strict=True):
        segment = min(3, int(math.floor(4 * cumulative / total)))
        segment_returns[segment] += minute_return
        counts[segment] += 1
        cumulative += value
    return segment_returns, counts


def common_side(segment_returns: np.ndarray, minimum_agree: int = 4) -> int:
    values = np.asarray(segment_returns, dtype=float)
    if len(values) != 4 or not np.isfinite(values).all() or np.any(values == 0):
        return 0
    positives = int(np.sum(values > 0))
    negatives = int(np.sum(values < 0))
    if positives >= minimum_agree:
        return 1
    if negatives >= minimum_agree:
        return -1
    return 0


def onset_after_previous_source_valid(
    source_valid: pd.Series, eligible: pd.Series
) -> pd.Series:
    output = pd.Series(False, index=eligible.index)
    previous_eligible = False
    for index in eligible.index:
        if not bool(source_valid.at[index]):
            continue
        current = bool(eligible.at[index])
        output.at[index] = current and not previous_eligible
        previous_eligible = current
    return output


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = engine()
    try:
        with database.connect() as connection:
            frame = pd.read_sql_query(
                text(QUERY),
                connection,
                params={"start": SOURCE_START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    value_columns = ("open", "high", "low", "close", "quote_asset_volume")
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.duplicated().any():
        raise RuntimeError("duplicate HVETCC source timestamps")
    frame = frame.set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(DECISION_START, END, freq="8h", inclusive="left"):
        index = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left"
        )
        window = frame.reindex(index)
        valid = bool(
            len(window) == 480
            and np.isfinite(window).all().all()
            and window[["open", "high", "low", "close"]].gt(0).all().all()
            and window.quote_asset_volume.gt(0).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        if valid:
            minute_returns = np.log(window.close.to_numpy(float) / window.open.to_numpy(float))
            segment_returns, counts = turnover_segments(
                window.quote_asset_volume.to_numpy(float), minute_returns
            )
            physical_returns = minute_returns.reshape(4, 120).sum(axis=1)
            variation = float(np.sqrt(np.square(minute_returns).sum()))
            turnover_side = common_side(segment_returns)
            physical_side = common_side(physical_returns)
            valid = bool(
                np.all(counts > 0)
                and np.isfinite(segment_returns).all()
                and np.isfinite(physical_returns).all()
                and math.isfinite(variation)
                and variation > 0
            )
        if not valid:
            segment_returns = np.full(4, np.nan)
            counts = np.zeros(4, dtype=int)
            physical_returns = np.full(4, np.nan)
            variation = math.nan
            turnover_side = physical_side = 0
        row: dict[str, Any] = {
            "decision_time": decision,
            "source_valid": valid,
            "realized_variation": variation,
            "turnover_side": turnover_side,
            "physical_side": physical_side,
        }
        for segment in range(4):
            row[f"turnover_segment_{segment}_return"] = float(segment_returns[segment])
            row[f"turnover_segment_{segment}_minutes"] = int(counts[segment])
            row[f"physical_segment_{segment}_return"] = float(physical_returns[segment])
        rows.append(row)
    states = pd.DataFrame(rows)
    states["variation_rank"] = prior_rank(states.realized_variation.where(states.source_valid))
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvetcc_source_v1",
        "query": QUERY,
        "source_window": [SOURCE_START.isoformat(), END.isoformat()],
        "decision_start": DECISION_START.isoformat(),
        "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {
            "path": str(PANEL),
            "sha256": sha256(PANEL),
            "rows": len(states),
            "valid_rows": int(states.source_valid.sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return states, manifest


def active(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    turnover_columns = [f"turnover_segment_{segment}_return" for segment in range(4)]
    physical_columns = [f"physical_segment_{segment}_return" for segment in range(4)]
    if control == "equal_physical_time":
        geometry = states[physical_columns]
        minimum_agree = 4
    else:
        geometry = states[turnover_columns]
        if control == "one_decision_stale_geometry":
            geometry = geometry.shift(1)
        minimum_agree = 3 if control == "three_of_four_concordance" else 4
    sides = geometry.apply(
        lambda row: common_side(row.to_numpy(float), minimum_agree), axis=1
    )
    variation_gate = (
        pd.Series(True, index=states.index)
        if control == "no_variation_gate"
        else states.variation_rank.ge(0.65)
    )
    eligible = states.source_valid & sides.ne(0) & variation_gate
    return onset_after_previous_source_valid(states.source_valid, eligible), sides


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, sides = active(states, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in states.index[onset]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_ <= end
            ),
            None,
        )
        if split is None:
            continue
        side = int(sides.at[index])
        if control == "direction_flip":
            side = -side
        elif control == "same_clock_forced_long":
            side = 1
        reserved_until = exit_
        row: dict[str, Any] = {
            "candidate": prereg.POLICY_ID,
            "control": control,
            "split": split,
            "decision_time": decision,
            "feature_available_time": decision,
            "entry_time": entry,
            "exit_time": exit_,
            "side": side,
            "realized_variation": float(states.at[index, "realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
        }
        for segment in range(4):
            row[f"turnover_segment_{segment}_return"] = float(
                states.at[index, f"turnover_segment_{segment}_return"]
            )
            row[f"turnover_segment_{segment}_minutes"] = int(
                states.at[index, f"turnover_segment_{segment}_minutes"]
            )
        rows.append(row)
    columns = [
        "candidate",
        "control",
        "split",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
        "side",
        "realized_variation",
        "variation_rank",
        *[f"turnover_segment_{segment}_return" for segment in range(4)],
        *[f"turnover_segment_{segment}_minutes" for segment in range(4)],
    ]
    return pd.DataFrame(rows, columns=columns)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock.split.eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(
            selected.entry_time.dt.strftime("%Y-%m").value_counts().max()
        ) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVETCC preregistration drift")
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for split in SPLITS:
        _write_gzip_csv(primary[primary.split.eq(split)].reset_index(drop=True), SPLIT_DIR / f"{split}.csv.gz")
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvetcc_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
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
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_clocks": {
            split: {
                "path": str(SPLIT_DIR / f"{split}.csv.gz"),
                "sha256": sha256(SPLIT_DIR / f"{split}.csv.gz"),
                "rows": int(primary.split.eq(split).sum()),
            }
            for split in SPLITS
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(value),
                "promotion_authorized": False,
            }
            for name, value in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
