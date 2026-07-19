"""Freeze outcome-blind clocks for PSR-30/6 Premium Snapback Recenter."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_binance_aggtrade_microstructure import _write_gzip_csv  # noqa: E402


CANDIDATE = "PSR-30/6"
SOURCE = Path(
    "data/binance_um_premium_path_btc_2020_2026/"
    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
)
SOURCE_MANIFEST = Path("results/binance_um_premium_path_btc_2020_2026_manifest.json")
EXPECTED_SOURCE_SHA256 = (
    "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8"
)
DEFAULT_CLOCKS = Path("data/premium_snapback_recenter_clocks_2020_2026.csv.gz")
DEFAULT_CONTROLS_DIR = Path("data/premium_snapback_recenter_controls_2020_2026")
DEFAULT_RESULT = Path("results/premium_snapback_recenter_support_2026-07-19.json")

EXTERNAL_COMPARATORS = {
    "CLBR-24": (
        Path("data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz"),
        "df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0",
        "2023-06-25",
        "2024-10-15",
    ),
    "ICLA-60": (
        Path("data/inverse_collateral_liquidation_absorption_clocks_2023_2024.csv.gz"),
        "a55c23a7a0c296b98bb7a8958f713548c4313c0c682f1693c8f8be80b70dd053",
        "2023-06-25",
        "2024-10-15",
    ),
    "EBLR-60/30": (
        Path("data/eth_btc_liquidation_relay_clocks_2023_2024.csv.gz"),
        "b4b35a0e9ae0cf26bf08df67b5c2fc832393c638c97f5b91a86894ee693b430e",
        "2023-06-25",
        "2024-10-15",
    ),
}

PATH_MINUTES = 30
DECISION_MINUTES = 5
PATH_DECISIONS = PATH_MINUTES // DECISION_MINUTES
REFERENCE_DAYS = 30
REFERENCE_MINUTES = REFERENCE_DAYS * 24 * 60
REFERENCE_DECISIONS = REFERENCE_DAYS * 24 * (60 // DECISION_MINUTES)
MIN_REFERENCE_SHARE = 0.95
MIN_REFERENCE_MINUTES = int(REFERENCE_MINUTES * MIN_REFERENCE_SHARE)
MIN_REFERENCE_DECISIONS = int(REFERENCE_DECISIONS * MIN_REFERENCE_SHARE)
RANGE_QUANTILE = 0.90
EFFICIENCY_QUANTILE = 0.35
TURNS_QUANTILE = 0.70
EXCURSION_QUANTILE = 0.85
TERMINAL_DEVIATION_QUANTILE = 0.40
ENTRY_DELAY_MINUTES = 10
HOLD_MINUTES = 30
PSI_HOLD_MINUTES = 96 * 5
MAX_EXACT_JACCARD = 0.10
MAX_NEAR_PRIMARY_SHARE = 0.20
NEAR_MINUTES = 30
RANDOM_SEED = 20260719
SCHEMA_VERSION = 1

SPLITS = {
    "train": ("2020-02-01", "2023-01-01"),
    "test": ("2023-01-01", "2024-01-01"),
    "eval": ("2024-01-01", "2026-07-01"),
}
SUPPORT_MIN_TOTAL = {"train": 120, "test": 30, "eval": 80}
SUPPORT_MIN_PER_SIDE = {"train": 30, "test": 8, "eval": 20}
SUPPORT_MAX_MONTH_SHARE = {"train": 0.15, "test": 0.25, "eval": 0.15}
SUPPORT_SUBPERIODS = {
    "train": {
        "2020_partial": ("2020-02-01", "2021-01-01", 20),
        "2021": ("2021-01-01", "2022-01-01", 30),
        "2022": ("2022-01-01", "2023-01-01", 30),
    },
    "test": {
        "2023H1": ("2023-01-01", "2023-07-01", 10),
        "2023H2": ("2023-07-01", "2024-01-01", 10),
    },
    "eval": {
        "2024": ("2024-01-01", "2025-01-01", 25),
        "2025": ("2025-01-01", "2026-01-01", 25),
        "2026H1": ("2026-01-01", "2026-07-01", 12),
    },
}

SOURCE_COLUMNS = (
    "date",
    "source_close_time",
    "feature_available_time",
    "source_valid",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
)
PATH_FEATURES = ("path_range", "efficiency", "turns", "max_excursion", "terminal_deviation")
THRESHOLD_COLUMNS = {
    "path_range": ("prior_q90_path_range", RANGE_QUANTILE),
    "efficiency": ("prior_q35_efficiency", EFFICIENCY_QUANTILE),
    "turns": ("prior_q70_turns", TURNS_QUANTILE),
    "max_excursion": ("prior_q85_max_excursion", EXCURSION_QUANTILE),
    "terminal_deviation": (
        "prior_q40_terminal_deviation",
        TERMINAL_DEVIATION_QUANTILE,
    ),
}
CLOCK_COLUMNS = (
    "candidate",
    "split",
    "path_start_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "planned_exit_time",
    "direction",
    "prior_center",
    "path_range",
    "efficiency",
    "turns",
    "up_excursion",
    "down_excursion",
    "max_excursion",
    "terminal_deviation",
)


@dataclass(frozen=True)
class Config:
    source_path: str = str(SOURCE)
    source_manifest_path: str = str(SOURCE_MANIFEST)
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256
    expected_source_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256
    output_clock_path: str = str(DEFAULT_CLOCKS)
    controls_dir: str = str(DEFAULT_CONTROLS_DIR)
    output_result_path: str = str(DEFAULT_RESULT)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def load_source(cfg: Config) -> pd.DataFrame:
    """Load only frozen premium-index fields; never open a BTC outcome file."""

    manifest_path = Path(cfg.source_manifest_path)
    if sha256_file(manifest_path) != cfg.expected_source_manifest_sha256:
        raise ValueError("premium source manifest sha256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("source_only") is not True or protocol.get("outcomes_opened") is not False:
        raise ValueError("premium source manifest is not outcome blind")
    if protocol.get("btc_execution_prices_retained") is not False:
        raise ValueError("premium source manifest retained BTC execution prices")
    if manifest.get("file", {}).get("sha256") != cfg.expected_source_sha256:
        raise ValueError("premium source manifest disagrees with pinned data hash")

    source_path = Path(cfg.source_path)
    if sha256_file(source_path) != cfg.expected_source_sha256:
        raise ValueError("premium source sha256 mismatch")
    frame = pd.read_csv(
        source_path,
        compression="gzip",
        usecols=cast(Any, list(SOURCE_COLUMNS)),
        parse_dates=["date", "source_close_time", "feature_available_time"],
    ).loc[:, list(SOURCE_COLUMNS)]
    expected = pd.Series(
        pd.date_range("2020-01-01", "2026-07-01", freq="1min", inclusive="left"),
        name="date",
    )
    if not cast(pd.Series, frame["date"]).equals(expected):
        raise ValueError("premium source grid mismatch")
    if frame["date"].duplicated().any():
        raise ValueError("premium source contains duplicate timestamps")
    close_delay = cast(pd.Series, frame["source_close_time"] - frame["date"])
    available_delay = cast(pd.Series, frame["feature_available_time"] - frame["date"])
    if not bool(close_delay.eq(pd.Timedelta(seconds=59, milliseconds=999)).all()):
        raise ValueError("premium source close-time contract mismatch")
    if not bool(available_delay.eq(pd.Timedelta(minutes=1, seconds=1)).all()):
        raise ValueError("premium source availability contract mismatch")
    valid = cast(pd.Series, frame["source_valid"]).astype(bool)
    premium = list(SOURCE_COLUMNS[4:])
    if not bool(frame.loc[valid, premium].notna().all().all()):
        raise ValueError("valid premium source row has missing OHLC")
    if not bool(frame.loc[~valid, premium].isna().all().all()):
        raise ValueError("invalid premium source row carries OHLC")
    return frame


def _completed_path_features(source: pd.DataFrame) -> pd.DataFrame:
    valid = cast(pd.Series, source["source_valid"]).astype(bool)
    opened = _numeric(source, "premium_open") * 10_000.0
    high = _numeric(source, "premium_high") * 10_000.0
    low = _numeric(source, "premium_low") * 10_000.0
    close = _numeric(source, "premium_close") * 10_000.0

    prior_valid = cast(
        pd.Series,
        valid.astype(int)
        .shift(PATH_MINUTES)
        .rolling(REFERENCE_MINUTES, min_periods=MIN_REFERENCE_MINUTES)
        .sum(),
    )
    center = close.where(valid).shift(PATH_MINUTES).rolling(
        REFERENCE_MINUTES, min_periods=MIN_REFERENCE_MINUTES
    ).median()
    full_reference_elapsed = pd.Series(
        np.arange(len(source)) >= PATH_MINUTES + REFERENCE_MINUTES - 1,
        index=source.index,
    )
    reference_complete = full_reference_elapsed & (prior_valid >= MIN_REFERENCE_MINUTES)
    path_count = cast(
        pd.Series,
        valid.astype(int)
        .rolling(PATH_MINUTES, min_periods=PATH_MINUTES)
        .sum(),
    )
    path_valid = path_count == PATH_MINUTES
    path_range = (high - low).rolling(PATH_MINUTES, min_periods=PATH_MINUTES).sum()
    first_open = opened.shift(PATH_MINUTES - 1)
    efficiency = (close - first_open).abs().div(path_range.where(path_range.gt(0.0)))
    delta = close.diff()
    turns = delta.mul(delta.shift(1)).lt(0.0).astype(float).rolling(
        PATH_MINUTES - 2, min_periods=PATH_MINUTES - 2
    ).sum()
    up_excursion = high.rolling(PATH_MINUTES, min_periods=PATH_MINUTES).max() - center
    down_excursion = center - low.rolling(PATH_MINUTES, min_periods=PATH_MINUTES).min()
    max_excursion = pd.concat([up_excursion, down_excursion], axis=1).max(axis=1)
    terminal_deviation = (close - center).abs()

    output = pd.DataFrame(
        {
            "date": source["date"],
            "feature_available_time": source["feature_available_time"],
            "path_valid": path_valid,
            "reference_complete": reference_complete,
            "prior_center": center,
            "path_range": path_range,
            "efficiency": efficiency,
            "turns": turns,
            "up_excursion": up_excursion,
            "down_excursion": down_excursion,
            "max_excursion": max_excursion,
            "terminal_deviation": terminal_deviation,
            "terminal_signed_deviation": close - center,
        }
    )
    decision_rows = np.arange(len(output)) % DECISION_MINUTES == DECISION_MINUTES - 1
    output = output.loc[decision_rows].reset_index(drop=True)
    output["decision_time"] = output["date"] + pd.Timedelta(minutes=1)
    output["path_start_time"] = output["decision_time"] - pd.Timedelta(minutes=PATH_MINUTES)
    return output


def _add_prior_thresholds(state: pd.DataFrame) -> pd.DataFrame:
    output = state.copy()
    usable = cast(pd.Series, output["path_valid"]).astype(bool) & cast(
        pd.Series, output["reference_complete"]
    ).astype(bool)
    finite = np.isfinite(output.loc[:, list(PATH_FEATURES)].to_numpy(float)).all(axis=1)
    usable &= finite
    prior_usable = usable.astype(int).shift(PATH_DECISIONS).rolling(
        REFERENCE_DECISIONS, min_periods=MIN_REFERENCE_DECISIONS
    ).sum()
    full_reference_elapsed = pd.Series(
        np.arange(len(output)) >= PATH_DECISIONS + REFERENCE_DECISIONS - 1,
        index=output.index,
    )
    output["feature_reference_complete"] = full_reference_elapsed & prior_usable.ge(
        MIN_REFERENCE_DECISIONS
    )
    for feature, (threshold, quantile) in THRESHOLD_COLUMNS.items():
        sample = _numeric(output, feature).where(usable)
        output[threshold] = sample.shift(PATH_DECISIONS).rolling(
            REFERENCE_DECISIONS, min_periods=MIN_REFERENCE_DECISIONS
        ).quantile(quantile)
    return output


def _psi_comparators(source: pd.DataFrame) -> pd.DataFrame:
    valid = cast(pd.Series, source["source_valid"]).astype(bool)
    high = _numeric(source, "premium_high")
    low = _numeric(source, "premium_low")
    close = _numeric(source, "premium_close")
    five_count = cast(pd.Series, valid.astype(int).rolling(5, min_periods=5).sum())
    five_valid = five_count == 5
    five_high = high.rolling(5, min_periods=5).max()
    five_low = low.rolling(5, min_periods=5).min()
    five_close = close
    rows = np.arange(len(source)) % 5 == 4
    frame = pd.DataFrame(
        {
            "decision_time": cast(pd.Series, source["date"]) + pd.Timedelta(minutes=1),
            "valid": five_valid,
            "high": five_high,
            "low": five_low,
            "close": five_close,
        }
    ).loc[rows].reset_index(drop=True)
    candle_range = cast(pd.Series, frame["high"] - frame["low"]).where(frame["valid"])
    close_location = (
        2.0 * (frame["close"] - frame["low"]) / candle_range.replace(0.0, np.nan) - 1.0
    ).clip(-1.0, 1.0)
    output = frame.loc[:, ["decision_time"]].copy()
    for window in (2016, 8640):
        prior = candle_range.shift(1)
        minimum = max(288, window // 2)
        mean = prior.rolling(window, min_periods=minimum).mean()
        std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
        zscore = (candle_range - mean) / std
        previous = zscore.shift(1)
        active = (
            cast(pd.Series, frame["valid"]).astype(bool)
            & previous.lt(3.0)
            & zscore.ge(3.0)
            & close_location.abs().ge(0.75)
        )
        output[f"psi_{window}_active"] = active
        output[f"psi_{window}_direction"] = np.sign(close_location).fillna(0.0).astype(int)
    return output


def derive_state(source: pd.DataFrame) -> pd.DataFrame:
    """Derive the single frozen PSR rule from premium data only."""

    state = _add_prior_thresholds(_completed_path_features(source))
    thresholds_finite = np.isfinite(
        state.loc[:, [item[0] for item in THRESHOLD_COLUMNS.values()]].to_numpy(float)
    ).all(axis=1)
    base = (
        cast(pd.Series, state["path_valid"]).astype(bool)
        & cast(pd.Series, state["reference_complete"]).astype(bool)
        & cast(pd.Series, state["feature_reference_complete"]).astype(bool)
        & thresholds_finite
    )
    high_energy = state["path_range"].ge(state["prior_q90_path_range"])
    inefficient = state["efficiency"].le(state["prior_q35_efficiency"])
    alternating = state["turns"].ge(state["prior_q70_turns"])
    extreme = state["max_excursion"].ge(state["prior_q85_max_excursion"])
    recentered = state["terminal_deviation"].le(
        state["prior_q40_terminal_deviation"]
    )
    upper_only = state["up_excursion"].ge(state["prior_q85_max_excursion"]) & state[
        "down_excursion"
    ].lt(state["prior_q85_max_excursion"])
    lower_only = state["down_excursion"].ge(state["prior_q85_max_excursion"]) & state[
        "up_excursion"
    ].lt(state["prior_q85_max_excursion"])
    primary = base & high_energy & inefficient & alternating & extreme & recentered & (
        upper_only | lower_only
    )
    direction = np.select([lower_only, upper_only], [1, -1], default=0).astype(int)
    direction[~primary.to_numpy(bool)] = 0
    state["is_candidate"] = primary
    state["direction"] = direction

    state["simple_level_active"] = base & extreme & (upper_only | lower_only)
    state["simple_level_direction"] = np.select(
        [lower_only, upper_only], [1, -1], default=0
    ).astype(int)
    state["no_recenter_active"] = (
        base
        & high_energy
        & inefficient
        & alternating
        & extreme
        & ~recentered
        & (upper_only | lower_only)
    )
    state["no_recenter_direction"] = state["simple_level_direction"]

    psi = _psi_comparators(source)
    if not cast(pd.Series, psi["decision_time"]).equals(state["decision_time"]):
        raise ValueError("PSR and PSI decision grids differ")
    for column in psi.columns.difference(["decision_time"]):
        state[column] = psi[column].to_numpy()
    return state


def build_clocks(
    state: pd.DataFrame,
    *,
    active_column: str = "is_candidate",
    direction_column: str = "direction",
    candidate: str = CANDIDATE,
    entry_shift_minutes: int = 0,
    enforce_causal_availability: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    decision = cast(pd.Series, pd.to_datetime(state["decision_time"]))
    for split, (start_text, end_text) in SPLITS.items():
        start = cast(pd.Timestamp, pd.Timestamp(start_text))
        end = cast(pd.Timestamp, pd.Timestamp(end_text))
        active = (
            cast(pd.Series, state[active_column]).astype(bool)
            & decision.ge(start)
            & decision.lt(end)
        )
        next_allowed: pd.Timestamp = start
        for index in np.flatnonzero(active.to_numpy(bool)):
            row = state.iloc[int(index)]
            entry = cast(
                pd.Timestamp,
                cast(pd.Timestamp, pd.Timestamp(cast(Any, row["decision_time"])))
                + pd.Timedelta(minutes=ENTRY_DELAY_MINUTES + entry_shift_minutes),
            )
            exit_time = cast(
                pd.Timestamp, entry + pd.Timedelta(minutes=HOLD_MINUTES)
            )
            if entry < start or exit_time > end or entry < next_allowed:
                continue
            feature_available = cast(
                pd.Timestamp, pd.Timestamp(cast(Any, row["feature_available_time"]))
            )
            if enforce_causal_availability and feature_available >= entry:
                raise ValueError("PSR feature is not available before entry")
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "path_start_time": row["path_start_time"],
                    "decision_time": row["decision_time"],
                    "feature_available_time": feature_available,
                    "entry_time": entry,
                    "planned_exit_time": exit_time,
                    "direction": int(row[direction_column]),
                    **{column: float(row[column]) for column in CLOCK_COLUMNS[8:]},
                }
            )
            next_allowed = exit_time
    return pd.DataFrame(rows, columns=cast(Any, list(CLOCK_COLUMNS)))


def _derived_primary_control(
    primary: pd.DataFrame,
    *,
    candidate: str,
    direction_multiplier: int = 1,
    shift_minutes: int = 0,
) -> pd.DataFrame:
    output = primary.copy()
    output["candidate"] = candidate
    output["direction"] = output["direction"].astype(int) * direction_multiplier
    for column in ("entry_time", "planned_exit_time"):
        output[column] = pd.to_datetime(output[column]) + pd.Timedelta(minutes=shift_minutes)
    keep = np.ones(len(output), dtype=bool)
    for split, (start_text, end_text) in SPLITS.items():
        in_split = output["split"].eq(split).to_numpy(bool)
        entry = pd.to_datetime(output["entry_time"])
        exit_time = pd.to_datetime(output["planned_exit_time"])
        inside = entry.ge(start_text).to_numpy(bool) & exit_time.le(end_text).to_numpy(bool)
        keep[in_split] &= inside[in_split]
    return output.loc[keep, list(CLOCK_COLUMNS)].reset_index(drop=True)


def _random_clocks(primary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    work = primary.copy()
    work["_month"] = pd.to_datetime(work["entry_time"]).dt.to_period("M")
    for key, group in work.groupby(["split", "_month"], sort=True):
        split, month_value = cast(tuple[Any, Any], key)
        month = pd.Period(month_value, freq="M")
        split_start = cast(pd.Timestamp, pd.Timestamp(SPLITS[str(split)][0]))
        split_end = cast(pd.Timestamp, pd.Timestamp(SPLITS[str(split)][1]))
        month = cast(pd.Period, month)
        month_start = max(split_start, month.start_time)
        month_end = min(split_end, month.end_time + pd.Timedelta(nanoseconds=1))
        slots = pd.date_range(
            month_start,
            month_end - pd.Timedelta(minutes=HOLD_MINUTES),
            freq="5min",
            inclusive="left",
        )
        order = rng.permutation(len(slots))
        selected: list[pd.Timestamp] = []
        for position in order:
            entry = cast(pd.Timestamp, pd.Timestamp(cast(Any, slots[int(position)])))
            if all(abs(entry - other) >= pd.Timedelta(minutes=HOLD_MINUTES) for other in selected):
                selected.append(entry)
                if len(selected) == len(group):
                    break
        if len(selected) != len(group):
            raise ValueError(f"could not allocate random clocks for {split} {month}")
        sides = group["direction"].astype(int).to_numpy().copy()
        rng.shuffle(sides)
        for entry, side in zip(sorted(selected), sides, strict=True):
            rows.append(
                {
                    "candidate": "PSR-random",
                    "split": split,
                    "path_start_time": pd.NaT,
                    "decision_time": entry - pd.Timedelta(minutes=ENTRY_DELAY_MINUTES),
                    "feature_available_time": pd.NaT,
                    "entry_time": entry,
                    "planned_exit_time": entry + pd.Timedelta(minutes=HOLD_MINUTES),
                    "direction": int(side),
                    **{column: np.nan for column in CLOCK_COLUMNS[8:]},
                }
            )
    return (
        pd.DataFrame(rows, columns=cast(Any, list(CLOCK_COLUMNS)))
        .sort_values("entry_time")
        .reset_index(drop=True)
    )


def _frozen_psi_clocks(state: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Reconstruct the selected prior PSI entry/hold contract source-only."""

    active_column = f"psi_{window}_active"
    direction_column = f"psi_{window}_direction"
    events: list[dict[str, Any]] = []
    next_allowed = pd.Timestamp.min
    for index in np.flatnonzero(
        cast(pd.Series, state[active_column]).astype(bool).to_numpy(bool)
    ):
        row = state.iloc[int(index)]
        entry = cast(
            pd.Timestamp, pd.Timestamp(cast(Any, row["decision_time"]))
        )
        exit_time = cast(
            pd.Timestamp, entry + pd.Timedelta(minutes=PSI_HOLD_MINUTES)
        )
        if entry < next_allowed:
            continue
        split_name: str | None = None
        for split, (start_text, end_text) in SPLITS.items():
            if entry >= pd.Timestamp(start_text) and exit_time <= pd.Timestamp(end_text):
                split_name = split
                break
        events.append(
            {
                "candidate": f"PSI-{window}-frozen-comparator",
                "split": split_name,
                "path_start_time": entry - pd.Timedelta(minutes=5),
                "decision_time": entry,
                "feature_available_time": row["feature_available_time"],
                "entry_time": entry,
                "planned_exit_time": exit_time,
                "direction": int(row[direction_column]),
                **{column: np.nan for column in CLOCK_COLUMNS[8:]},
            }
        )
        next_allowed = exit_time
    frame = pd.DataFrame(events, columns=cast(Any, list(CLOCK_COLUMNS)))
    return frame.loc[frame["split"].notna()].reset_index(drop=True)


def build_controls(state: pd.DataFrame, primary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "direction_flip": _derived_primary_control(
            primary, candidate="PSR-direction-flip", direction_multiplier=-1
        ),
        "simple_level": build_clocks(
            state,
            active_column="simple_level_active",
            direction_column="simple_level_direction",
            candidate="PSR-simple-level",
        ),
        "no_recenter": build_clocks(
            state,
            active_column="no_recenter_active",
            direction_column="no_recenter_direction",
            candidate="PSR-no-recenter",
        ),
        "psi_2016": _frozen_psi_clocks(state, window=2016),
        "psi_8640": _frozen_psi_clocks(state, window=8640),
        "extra_latency": _derived_primary_control(
            primary, candidate="PSR-extra-latency", shift_minutes=5
        ),
        "future_premium_placebo": _derived_primary_control(
            primary, candidate="PSR-future-premium-placebo", shift_minutes=-40
        ),
        "random": _random_clocks(primary),
    }


def _support_stats(clocks: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clocks.loc[clocks["split"].eq(split)].copy()
    entry = pd.to_datetime(subset["entry_time"])
    month_counts = entry.dt.strftime("%Y-%m").value_counts().sort_index()
    subperiod_counts = {
        name: int(entry.ge(start).mul(entry.lt(end)).sum())
        for name, (start, end, _minimum) in SUPPORT_SUBPERIODS[split].items()
    }
    total = len(subset)
    return {
        "total": total,
        "long": int(subset["direction"].eq(1).sum()),
        "short": int(subset["direction"].eq(-1).sum()),
        "long_share": float(subset["direction"].eq(1).mean()) if total else 0.0,
        "short_share": float(subset["direction"].eq(-1).mean()) if total else 0.0,
        "max_month_share": float(month_counts.max() / total) if total else 1.0,
        "month_counts": {str(key): int(value) for key, value in month_counts.items()},
        "subperiod_counts": subperiod_counts,
    }


def _support_passes(stats: dict[str, dict[str, Any]]) -> bool:
    for split, row in stats.items():
        if row["total"] < SUPPORT_MIN_TOTAL[split]:
            return False
        if min(row["long"], row["short"]) < SUPPORT_MIN_PER_SIDE[split]:
            return False
        if min(row["long_share"], row["short_share"]) < 0.25:
            return False
        if row["max_month_share"] > SUPPORT_MAX_MONTH_SHARE[split]:
            return False
        for name, (_start, _end, minimum) in SUPPORT_SUBPERIODS[split].items():
            if row["subperiod_counts"][name] < minimum:
                return False
    return True


def _clock_times(path: Path, expected_hash: str) -> pd.DatetimeIndex:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"comparator clock hash mismatch: {path}")
    frame = pd.read_csv(
        path,
        compression="gzip",
        usecols=cast(Any, ["entry_time"]),
        parse_dates=["entry_time"],
    )
    entry = cast(pd.Series, frame["entry_time"]).drop_duplicates().sort_values()
    return pd.DatetimeIndex(entry)


def _overlap(
    primary: pd.DatetimeIndex,
    other: pd.DatetimeIndex,
    *,
    coverage_start: str | None = None,
    coverage_end: str | None = None,
) -> dict[str, Any]:
    primary = primary.dropna().sort_values().drop_duplicates()
    other = other.dropna().sort_values().drop_duplicates()
    primary_full, other_full = len(primary), len(other)
    if not len(primary) or not len(other):
        return {
            "primary_full": primary_full,
            "other_full": other_full,
            "shared_start": None,
            "shared_end": None,
            "primary": 0,
            "other": 0,
            "exact_intersection": 0,
            "exact_jaccard": 0.0,
            "within_30m_primary": 0,
            "within_30m_primary_share": 0.0,
        }
    if (coverage_start is None) != (coverage_end is None):
        raise ValueError("overlap coverage requires both start and end")
    if coverage_start is not None and coverage_end is not None:
        shared_start = cast(pd.Timestamp, pd.Timestamp(coverage_start))
        shared_end = cast(pd.Timestamp, pd.Timestamp(coverage_end))
        primary = primary[(primary >= shared_start) & (primary < shared_end)]
        other = other[(other >= shared_start) & (other < shared_end)]
    else:
        primary_start = cast(pd.Timestamp, primary.min())
        other_start = cast(pd.Timestamp, other.min())
        primary_end = cast(pd.Timestamp, primary.max())
        other_end = cast(pd.Timestamp, other.max())
        shared_start = min(primary_start, other_start)
        shared_end = max(primary_end, other_end)
    primary_ns = set(primary.view("int64").tolist())
    other_ns = set(other.view("int64").tolist())
    intersection = len(primary_ns & other_ns)
    union = len(primary_ns | other_ns)
    tolerance_ns = int(pd.Timedelta(minutes=NEAR_MINUTES).value)
    ordered = np.sort(np.fromiter(other_ns, dtype=np.int64))
    near = 0
    for value in primary_ns:
        position = int(np.searchsorted(ordered, value))
        candidates = ordered[max(0, position - 1) : min(len(ordered), position + 1)]
        if len(candidates) and int(np.min(np.abs(candidates - value))) <= tolerance_ns:
            near += 1
    return {
        "primary_full": primary_full,
        "other_full": other_full,
        "shared_start": str(shared_start),
        "shared_end": str(shared_end),
        "primary": len(primary_ns),
        "other": len(other_ns),
        "exact_intersection": intersection,
        "exact_jaccard": float(intersection / union) if union else 0.0,
        "within_30m_primary": near,
        "within_30m_primary_share": float(near / len(primary_ns)) if primary_ns else 0.0,
    }


def build_result(
    cfg: Config,
    primary: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    support = {split: _support_stats(primary, split) for split in SPLITS}
    primary_times = pd.DatetimeIndex(pd.to_datetime(primary["entry_time"]))
    novelty: dict[str, Any] = {}
    for name in ("psi_2016", "psi_8640"):
        novelty[name] = _overlap(
            primary_times,
            pd.DatetimeIndex(pd.to_datetime(controls[name]["entry_time"])),
        )
    for name, (path, expected_hash, coverage_start, coverage_end) in EXTERNAL_COMPARATORS.items():
        novelty[name] = _overlap(
            primary_times,
            _clock_times(path, expected_hash),
            coverage_start=coverage_start,
            coverage_end=coverage_end,
        )
    novelty_passes = all(
        row["exact_jaccard"] <= MAX_EXACT_JACCARD
        and row["within_30m_primary_share"] <= MAX_NEAR_PRIMARY_SHARE
        for row in novelty.values()
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "candidate": CANDIDATE,
        "protocol": {
            "source_only": True,
            "outcomes_opened": False,
            "btc_execution_prices_opened": False,
            "funding_opened": False,
            "candidate_count": 1,
            "threshold_grid": False,
            "direction_search": False,
            "hold_search": False,
            "entry": "decision + 10m after one empty completed 5m latency bucket",
            "hold": "30m fixed; no stop, take-profit, regime gate or dynamic exit",
        },
        "config": asdict(cfg),
        "frozen_constants": {
            "path_minutes": PATH_MINUTES,
            "decision_minutes": DECISION_MINUTES,
            "reference_days": REFERENCE_DAYS,
            "minimum_reference_share": MIN_REFERENCE_SHARE,
            "quantiles": {
                "path_range": RANGE_QUANTILE,
                "efficiency": EFFICIENCY_QUANTILE,
                "turns": TURNS_QUANTILE,
                "max_excursion": EXCURSION_QUANTILE,
                "terminal_deviation": TERMINAL_DEVIATION_QUANTILE,
            },
            "entry_delay_minutes": ENTRY_DELAY_MINUTES,
            "hold_minutes": HOLD_MINUTES,
            "splits": SPLITS,
        },
        "source": {
            "path": cfg.source_path,
            "sha256": cfg.expected_source_sha256,
            "manifest": cfg.source_manifest_path,
            "manifest_sha256": cfg.expected_source_manifest_sha256,
        },
        "primary_clock": {
            "path": cfg.output_clock_path,
            "sha256": sha256_file(cfg.output_clock_path),
            "rows": int(len(primary)),
        },
        "controls": {
            name: {
                "path": str(Path(cfg.controls_dir) / f"{name}.csv.gz"),
                "sha256": sha256_file(Path(cfg.controls_dir) / f"{name}.csv.gz"),
                "rows": int(len(frame)),
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_passes": _support_passes(support),
        "novelty": novelty,
        "novelty_passes": novelty_passes,
        "may_open_train": bool(_support_passes(support) and novelty_passes),
        "implementation_sha256": sha256_file(__file__),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def run(cfg: Config) -> dict[str, Any]:
    source = load_source(cfg)
    state = derive_state(source)
    primary = build_clocks(state)
    controls = build_controls(state, primary)
    output = Path(cfg.output_clock_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, output)
    controls_dir = Path(cfg.controls_dir)
    controls_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in controls.items():
        _write_gzip_csv(frame, controls_dir / f"{name}.csv.gz")
    result = build_result(cfg, primary, controls)
    result_path = Path(cfg.output_result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", default=Config.source_path)
    parser.add_argument("--source-manifest-path", default=Config.source_manifest_path)
    parser.add_argument("--output-clock-path", default=Config.output_clock_path)
    parser.add_argument("--controls-dir", default=Config.controls_dir)
    parser.add_argument("--output-result-path", default=Config.output_result_path)
    result = run(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "support": result["support"],
                "novelty": result["novelty"],
                "may_open_train": result["may_open_train"],
                "manifest_hash": result["manifest_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
