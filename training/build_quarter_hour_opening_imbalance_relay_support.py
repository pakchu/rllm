"""Materialize source-only support clocks for frozen QHOIR-8."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_quarter_hour_opening_imbalance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER_PATH = Path("training/build_quarter_hour_opening_imbalance_relay_support.py")
PREREG_SHA256 = "a714b8cc65bef74b603df6428476f38f9aa6e48b5c4c60854fc12d56f73319cb"
SOURCE_START = pd.Timestamp("2023-04-01T00:00:00Z")
STAGE_END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:01:00Z")
SOURCE_DIR = Path("data/quarter_hour_opening_imbalance_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "quarter_hour_opening_imbalance_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/quarter_hour_opening_imbalance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/quarter_hour_opening_imbalance_relay_controls_2023_2026")
RESULT = Path("results/quarter_hour_opening_imbalance_relay_support_2026-08-09.json")

SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), STAGE_END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "shifted_phase_plus_2m",
    "five_minute_phase_only",
    "no_volatility_gate",
    "one_quarter_stale_imbalance",
    "direction_flip",
    "exclude_funding_boundaries",
)
PANEL_COLUMNS = (
    "decision_time",
    "is_quarter_hour",
    "source_valid",
    "opening_imbalance",
    "shifted_phase_plus_2m_valid",
    "shifted_phase_plus_2m_imbalance",
    "prior_quarter_valid",
    "prior_quarter_imbalance",
    "realized_variation",
    "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "opening_imbalance",
    "realized_variation",
    "variation_rank",
)
QUERY = """
SELECT ts,open,close,volume,taker_buy_base
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 1)

    def add(self, index: int, amount: int) -> None:
        index += 1
        while index < len(self.tree):
            self.tree[index] += amount
            index += index & -index

    def prefix(self, stop: int) -> int:
        """Return the count in compressed positions ``[0, stop)``."""
        total = 0
        while stop:
            total += self.tree[stop]
            stop -= stop & -stop
        return total


def strict_prior_midrank(
    values: pd.Series,
    lookback: int = 8640,
    minimum: int = 5760,
    update_mask: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """Causal rolling midrank; the current value is ranked before it enters history.

    ``update_mask`` permits five-minute diagnostic rows to be ranked against, but not
    added to, the preregistered history of quarter-hour observations.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    updates = (
        np.ones(len(numeric), dtype=bool)
        if update_mask is None
        else np.asarray(update_mask, dtype=bool)
    )
    if len(updates) != len(numeric):
        raise ValueError("update_mask length must match values")
    finite_values = numeric[np.isfinite(numeric)].tolist()
    coordinates = sorted(set(finite_values))
    positions = {value: index for index, value in enumerate(coordinates)}
    counts = _Fenwick(len(coordinates))
    history: deque[float] = deque()
    output = pd.Series(np.nan, index=numeric.index, dtype=float)

    for offset, (index, current) in enumerate(numeric.items()):
        if math.isfinite(current) and len(history) >= minimum:
            position = bisect.bisect_left(coordinates, current)
            below = counts.prefix(position)
            equal = counts.prefix(position + 1) - below
            output.at[index] = (below + 0.5 * equal) / len(history)
        if updates[offset] and math.isfinite(current):
            if len(history) == lookback:
                counts.add(positions[history.popleft()], -1)
            history.append(current)
            counts.add(positions[current], 1)
    return output


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def _validated_bars(
    bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    required = ["ts", "open", "close", "volume", "taker_buy_base"]
    if list(bars.columns) != required:
        raise RuntimeError(f"QHOIR source schema must be exactly {required}")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    if frame.ts.duplicated().any():
        raise RuntimeError("QHOIR source has duplicate minutes")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.reset_index(drop=True).equals(
        pd.Series(expected, name="ts")
    ):
        raise RuntimeError("QHOIR source is not the exact requested one-minute grid")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[required[1:]]).all(axis=1)
    price_valid = frame[["open", "close"]].gt(0).all(axis=1)
    flow_valid = (
        frame.volume.ge(0)
        & frame.taker_buy_base.ge(0)
        & frame.taker_buy_base.le(frame.volume)
    )
    if not bool((finite & price_valid & flow_valid).all()):
        raise RuntimeError("QHOIR source contains invalid price, volume, or taker flow")
    return frame.set_index("ts")


def build_source_panel(
    bars: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    rank_lookback: int = 8640,
    rank_minimum: int = 5760,
) -> pd.DataFrame:
    """Validate exact bars and derive only preregistered pre-entry features."""
    if bars.empty:
        raise RuntimeError("QHOIR source is empty")
    parsed = pd.to_datetime(bars["ts"], utc=True, errors="raise")
    source_start = pd.Timestamp(start) if start is not None else parsed.min()
    source_end = pd.Timestamp(end) if end is not None else parsed.max() + pd.Timedelta(minutes=1)
    if source_start.tzinfo is None:
        source_start = source_start.tz_localize("UTC")
    if source_end.tzinfo is None:
        source_end = source_end.tz_localize("UTC")
    source = _validated_bars(bars, source_start, source_end)

    squared_bar_return = np.log(source.close / source.open).pow(2)
    variation = np.sqrt(squared_bar_return.shift(1).rolling(1440, min_periods=1440).sum())
    decisions = pd.date_range(
        source_start + pd.Timedelta(days=1), source_end, freq="5min", inclusive="left"
    )
    panel = pd.DataFrame({"decision_time": decisions})
    panel["is_quarter_hour"] = panel.decision_time.dt.minute.isin((0, 15, 30, 45))

    opening = source.reindex(decisions)
    opening_volume = opening.volume.to_numpy(dtype=float)
    panel["source_valid"] = opening_volume > 0
    panel["opening_imbalance"] = np.where(
        panel.source_valid,
        (2 * opening.taker_buy_base.to_numpy(dtype=float) - opening_volume) / opening_volume,
        np.nan,
    )
    shifted = source.reindex(decisions + pd.Timedelta(minutes=2))
    shifted_volume = shifted.volume.to_numpy(dtype=float)
    panel["shifted_phase_plus_2m_valid"] = shifted_volume > 0
    panel["shifted_phase_plus_2m_imbalance"] = np.where(
        panel.shifted_phase_plus_2m_valid,
        (2 * shifted.taker_buy_base.to_numpy(dtype=float) - shifted_volume) / shifted_volume,
        np.nan,
    )
    panel["realized_variation"] = variation.reindex(decisions).to_numpy(dtype=float)

    quarter_imbalance = panel.loc[panel.is_quarter_hour].set_index("decision_time")[
        "opening_imbalance"
    ]
    prior_times = panel.decision_time - pd.Timedelta(minutes=15)
    panel["prior_quarter_imbalance"] = quarter_imbalance.reindex(prior_times).to_numpy()
    panel["prior_quarter_valid"] = np.isfinite(panel.prior_quarter_imbalance)
    panel["variation_rank"] = strict_prior_midrank(
        panel.realized_variation,
        lookback=rank_lookback,
        minimum=rank_minimum,
        update_mask=panel.is_quarter_hour,
    )
    return panel.loc[:, PANEL_COLUMNS]


def materialize() -> dict[str, Any]:
    """Read PostgreSQL exactly once and write the source panel after validation."""
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            bars = pd.read_sql_query(
                text(QUERY),
                connection,
                params={
                    "start": SOURCE_START.to_pydatetime(),
                    "end": SOURCE_END.to_pydatetime(),
                },
            )
    finally:
        database.dispose()
    panel = build_source_panel(bars, start=SOURCE_START, end=SOURCE_END)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    core = {
        "protocol_version": "qhoir_8_btc_source_v1",
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "columns": ["ts", "open", "close", "volume", "taker_buy_base"],
        "window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()],
        "exact_minute_grid": True,
        "no_imputation": True,
        "candidate_incidence_opened": False,
        "candidate_outcomes_opened": False,
        "builder": {"path": str(BUILDER_PATH), "sha256": sha(BUILDER_PATH)},
        "output": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel)},
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def features() -> pd.DataFrame:
    frame = pd.read_csv(PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True, errors="raise")
    for column in ("is_quarter_hour", "source_valid", "shifted_phase_plus_2m_valid", "prior_quarter_valid"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    for column in (
        "opening_imbalance",
        "shifted_phase_plus_2m_imbalance",
        "prior_quarter_imbalance",
        "realized_variation",
        "variation_rank",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown QHOIR control: {control}")
    quarter = frame.is_quarter_hour.astype(bool)
    phase = ~quarter if control == "five_minute_phase_only" else quarter
    imbalance = frame.opening_imbalance
    source_valid = frame.source_valid.astype(bool)
    if control == "shifted_phase_plus_2m":
        imbalance = frame.shifted_phase_plus_2m_imbalance
        source_valid = frame.shifted_phase_plus_2m_valid.astype(bool)
    elif control == "one_quarter_stale_imbalance":
        imbalance = frame.prior_quarter_imbalance
        source_valid = frame.prior_quarter_valid.astype(bool)
    volatility = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else frame.variation_rank.ge(0.65)
    )
    funding_boundary = (
        frame.decision_time.dt.minute.eq(0)
        & frame.decision_time.dt.hour.isin((0, 8, 16))
    )
    if control != "exclude_funding_boundaries":
        funding_boundary = pd.Series(False, index=frame.index)
    active = (
        phase
        & source_valid
        & np.isfinite(imbalance)
        & imbalance.ne(0)
        & np.isfinite(frame.realized_variation)
        & np.isfinite(frame.variation_rank)
        & volatility
        & ~funding_boundary
    )
    side = pd.Series(
        np.where(imbalance.gt(0), 1, np.where(imbalance.lt(0), -1, 0)),
        index=frame.index,
        dtype=int,
    )
    if control == "direction_flip":
        side = -side
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    """Build a globally reserved, half-open eight-hour source-only clock."""
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        feature_delay = 3 if control == "shifted_phase_plus_2m" else 1
        feature_available = decision + pd.Timedelta(minutes=feature_delay)
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if feature_available > entry:
            raise RuntimeError("QHOIR feature is not available before entry")
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
        if split is None:
            continue
        next_allowed = exit_time
        used_imbalance = (
            frame.at[index, "shifted_phase_plus_2m_imbalance"]
            if control == "shifted_phase_plus_2m"
            else frame.at[index, "prior_quarter_imbalance"]
            if control == "one_quarter_stale_imbalance"
            else frame.at[index, "opening_imbalance"]
        )
        rows.append(
            {
                "candidate": "QHOIR-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": feature_available,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "opening_imbalance": float(used_imbalance),
                "realized_variation": float(frame.at[index, "realized_variation"]),
                "variation_rank": float(frame.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
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


def support_checks(support: dict[str, dict[str, float | int]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for split, values in support.items():
        checks[f"{split}_minimum_events"] = values["events"] >= MINIMUM[split]
        checks[f"{split}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{split}_month_concentration"] = values["max_month_share"] <= 0.45
    return checks


def run() -> dict[str, Any]:
    # Source validation/materialization is deliberately first. Any source failure raises
    # before incidence, Gross9, execution prices, returns, or PnL can be opened.
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("QHOIR preregistration hash drift")
    source_manifest = materialize()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")

    support = {name: stats(primary, name) for name in SPLITS}
    checks = support_checks(support)
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "qhoir_8_source_support_v1",
        "policy_id": "QHOIR-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(candidate),
                "promotion_authorized": False,
            }
            for name, candidate in controls.items()
        },
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
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
