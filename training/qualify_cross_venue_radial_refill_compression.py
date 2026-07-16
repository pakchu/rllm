"""Outcome-blind support and clock-overlap qualification for CRRC-72.

This module is deliberately unable to calculate a return.  It reads only the
frozen calendar-2023 Binance book-shell and credibility panels, recreates the
preregistered incidence grid, and compares the selected causal event clock
with earlier order-book clocks before any execution price is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_cross_collateral_liquidity_credibility_fracture as pdf
from training import preregister_cross_collateral_liquidity_hysteresis as cclh
from training import preregister_cross_venue_radial_refill_compression as prereg
from training import preregister_radial_liquidity_wavefront_cascade as rlwc
from training.preregister_cross_collateral_liquidity_void_refill import (
    lagged_robust_zscore,
)


SELECTION_START = pd.Timestamp("2023-01-01 00:00:00")
SELECTION_END = pd.Timestamp("2024-01-01 00:00:00")
EXPECTED_ROWS = 105_120
EXPECTED_JOINT_COMPLETE_ROWS = 101_649
EXPECTED_CCLH_EVENT_HASH = (
    "e90079d95b111f95ce64459c42d17e4286636a1a2854ed948e8ada497a13dfa7"
)
EXPECTED_PDF_EVENT_HASH = (
    "ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2"
)
EXPECTED_NEAR_SCHEDULE_HASHES = {
    "q1": "1d683925ab040552c4ce79fe4262a9889b5f08c4df9fe27dd636d2078f37f38e",
    "q2": "6e50405079db515648a7f9edbb2561e0b223498267664dccf7f5857466c2fafd",
    "q3": "04877f5ff37b1864ea26f0728f67abf76e14fbaad2f3c6e0941f4089556add01",
    "q4": "0988e43567f94cb470605f8fe7dd45f15017436b360be6d16a2eb7c0bae2f426",
}
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "funding_rate",
        "return",
        "pnl",
        "equity",
    }
)
QUARTERS: tuple[tuple[str, str, str], ...] = (
    ("q1", "2023-01-01", "2023-04-01"),
    ("q2", "2023-04-01", "2023-07-01"),
    ("q3", "2023-07-01", "2023-10-01"),
    ("q4", "2023-10-01", "2024-01-01"),
)


@dataclass(frozen=True)
class Config:
    shells_csv: str = (
        "data/binance_cross_collateral_book_shells_btc_2023/"
        "BTC_cross_collateral_book_shells_5m_2023.csv.gz"
    )
    credibility_csv: str = (
        "data/binance_cross_collateral_book_credibility_btc_2023/"
        "BTC_cross_collateral_book_credibility_5m_2023.csv.gz"
    )
    preregistration_result: str = str(prereg.DEFAULT_OUTPUT)
    near_manifest: str = (
        "results/cross_collateral_near_pressure_pre2024_manifest_2026-07-16.json"
    )
    output: str = (
        "results/cross_venue_radial_refill_compression_support_2026-07-17.json"
    )
    event_clock_output: str = (
        "results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json"
    )
    rolling_window_rows: int = 8_640
    rolling_minimum_rows: int = 6_912
    hold_bars: int = 72
    entry_delay_bars: int = 2
    selected_q_add: float = 0.85
    selected_q_withdraw: float = 0.75
    selected_q_net: float = 0.55
    selected_q_flicker: float = 0.85
    minimum_events: int = 150
    minimum_half_events: int = 50
    minimum_quarter_events: int = 25
    minimum_side_share: float = 0.35
    maximum_month_share: float = 0.20
    maximum_quarter_share: float = 0.40
    exact_entry_jaccard_maximum: float = 0.05
    tolerant_bars: int = 12
    tolerant_candidate_match_share_maximum: float = 0.35
    position_time_jaccard_maximum: float = 0.25


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_existing(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / candidate
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(path)


def _parse_complete(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    parsed = values.astype("string").str.lower().map(
        {"true": True, "false": False}
    )
    if parsed.isna().any():
        raise ValueError("source_complete contains an unknown value")
    return parsed.astype(bool)


def _validate_grid(frame: pd.DataFrame, label: str) -> None:
    dates = pd.to_datetime(frame["date"])
    if len(dates) != EXPECTED_ROWS:
        raise ValueError(f"{label} does not contain every 2023 5m row")
    if dates.iloc[0] != SELECTION_START or dates.iloc[-1] != pd.Timestamp(
        "2023-12-31 23:55:00"
    ):
        raise ValueError(f"{label} calendar bounds differ")
    if not dates.diff().dropna().eq(pd.Timedelta("5min")).all():
        raise ValueError(f"{label} is not a complete 5m grid")
    if dates.max() >= SELECTION_END:
        raise RuntimeError(f"{label} opened a post-2023 row")
    forbidden = FORBIDDEN_OUTCOME_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise RuntimeError(f"{label} contains forbidden outcomes: {sorted(forbidden)}")


def _crrc_required_columns() -> tuple[list[str], list[str]]:
    shells = ["date", "source_complete"]
    credibility = ["date", "source_complete"]
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            shells.extend(
                [
                    f"{venue}_shell_flow_add_{side}1",
                    f"{venue}_shell_flow_add_{side}2",
                    f"{venue}_shell_flow_withdraw_{side}4",
                    f"{venue}_shell_flow_withdraw_{side}5",
                ]
            )
            credibility.extend(
                [
                    f"{venue}_log_net_{side}1",
                    f"{venue}_log_net_{side}2",
                    f"{venue}_log_mad_{side}1",
                    f"{venue}_log_mad_{side}2",
                    f"{venue}_log_step_{side}1",
                    f"{venue}_log_step_{side}2",
                ]
            )
    return shells, credibility


def load_sources(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    shell_path = resolve_existing(cfg.shells_csv)
    credibility_path = resolve_existing(cfg.credibility_csv)
    if sha256(shell_path) != prereg.SHELL_SOURCE_SHA256:
        raise RuntimeError("frozen shell panel hash mismatch")
    if sha256(credibility_path) != prereg.CREDIBILITY_SOURCE_SHA256:
        raise RuntimeError("frozen credibility panel hash mismatch")

    shells = pd.read_csv(shell_path, compression="infer", parse_dates=["date"])
    credibility = pd.read_csv(
        credibility_path, compression="infer", parse_dates=["date"]
    )
    for frame, label in ((shells, "shells"), (credibility, "credibility")):
        _validate_grid(frame, label)
        frame["source_complete"] = _parse_complete(frame["source_complete"])
        frame["quarantined"] = False
    if not np.array_equal(shells["date"].to_numpy(), credibility["date"].to_numpy()):
        raise RuntimeError("shell and credibility grids differ")

    shell_required, credibility_required = _crrc_required_columns()
    for frame, required, label in (
        (shells, shell_required, "shells"),
        (credibility, credibility_required, "credibility"),
    ):
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise RuntimeError(f"{label} misses CRRC columns: {missing}")

    joint_complete = shells["source_complete"] & credibility["source_complete"]
    if int(joint_complete.sum()) != EXPECTED_JOINT_COMPLETE_ROWS:
        raise RuntimeError("joint-complete row count drifted")
    numeric = pd.concat(
        [
            shells.loc[joint_complete, shell_required[2:]],
            credibility.loc[joint_complete, credibility_required[2:]],
        ],
        axis=1,
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("joint-complete CRRC input contains a non-finite value")

    return shells, credibility, {
        "shell_path": str(shell_path),
        "shell_sha256": sha256(shell_path),
        "credibility_path": str(credibility_path),
        "credibility_sha256": sha256(credibility_path),
        "rows": int(len(shells)),
        "joint_complete_rows": int(joint_complete.sum()),
        "post_2023_rows_loaded": False,
        "outcome_columns_loaded": False,
    }


def _validate_frozen_config(cfg: Config) -> None:
    expected = {
        "rolling_window_rows": 8_640,
        "rolling_minimum_rows": 6_912,
        "hold_bars": 72,
        "entry_delay_bars": 2,
        "selected_q_add": 0.85,
        "selected_q_withdraw": 0.75,
        "selected_q_net": 0.55,
        "selected_q_flicker": 0.85,
        "minimum_events": 150,
        "minimum_half_events": 50,
        "minimum_quarter_events": 25,
        "minimum_side_share": 0.35,
        "maximum_month_share": 0.20,
        "maximum_quarter_share": 0.40,
        "exact_entry_jaccard_maximum": 0.05,
        "tolerant_bars": 12,
        "tolerant_candidate_match_share_maximum": 0.35,
        "position_time_jaccard_maximum": 0.25,
    }
    changed = {
        key: {"expected": value, "observed": getattr(cfg, key)}
        for key, value in expected.items()
        if getattr(cfg, key) != value
    }
    if changed:
        raise ValueError(f"CRRC-72 support config is frozen: {changed}")


def raw_features(
    shells: pd.DataFrame,
    credibility: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], pd.Series], pd.Series]:
    joint_complete = (
        _parse_complete(shells["source_complete"])
        & _parse_complete(credibility["source_complete"])
    )
    output: dict[tuple[str, str, str], pd.Series] = {}
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            values = {
                "add": (
                    shells[f"{venue}_shell_flow_add_{side}1"].astype(float)
                    + 0.5
                    * shells[f"{venue}_shell_flow_add_{side}2"].astype(float)
                ),
                "withdraw": (
                    shells[f"{venue}_shell_flow_withdraw_{side}4"].astype(float)
                    + shells[f"{venue}_shell_flow_withdraw_{side}5"].astype(float)
                ),
                "net": 0.5
                * (
                    credibility[f"{venue}_log_net_{side}1"].astype(float)
                    + credibility[f"{venue}_log_net_{side}2"].astype(float)
                ),
                "flicker": 0.25
                * (
                    credibility[f"{venue}_log_mad_{side}1"].astype(float)
                    + credibility[f"{venue}_log_mad_{side}2"].astype(float)
                    + credibility[f"{venue}_log_step_{side}1"].astype(float)
                    + credibility[f"{venue}_log_step_{side}2"].astype(float)
                ),
            }
            for metric, value in values.items():
                output[(venue, side, metric)] = value.where(
                    joint_complete & np.isfinite(value)
                )
    return output, joint_complete


def lagged_quantile(
    values: pd.Series,
    quantile: float,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if not 1 <= minimum <= window:
        raise ValueError("invalid rolling quantile support")
    return (
        values.astype(float)
        .rolling(window, min_periods=minimum)
        .quantile(quantile, interpolation="linear")
        .shift(1)
    )


def classify(
    raw: dict[tuple[str, str, str], pd.Series],
    joint_complete: pd.Series,
    thresholds: dict[tuple[str, str, str], pd.Series],
) -> pd.DataFrame:
    passes: dict[tuple[str, str], pd.Series] = {}
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            observed = joint_complete.astype(bool).copy()
            for metric in ("add", "withdraw", "net", "flicker"):
                value = raw[(venue, side, metric)]
                threshold = thresholds[(venue, side, metric)]
                observed &= (
                    value.notna()
                    & threshold.notna()
                    & np.isfinite(value)
                    & np.isfinite(threshold)
                    & threshold.ne(0.0)
                )
            passes[(venue, side)] = (
                observed
                & raw[(venue, side, "add")].ge(thresholds[(venue, side, "add")])
                & raw[(venue, side, "withdraw")].ge(
                    thresholds[(venue, side, "withdraw")]
                )
                & raw[(venue, side, "net")].ge(thresholds[(venue, side, "net")])
                & raw[(venue, side, "flicker")].le(
                    thresholds[(venue, side, "flicker")]
                )
            )
    bid_both = passes[("um", "m")] & passes[("cm", "m")]
    ask_both = passes[("um", "p")] & passes[("cm", "p")]
    side = pd.Series(0, index=joint_complete.index, dtype=np.int8)
    side.loc[bid_both & ~ask_both] = 1
    side.loc[ask_both & ~bid_both] = -1
    return pd.DataFrame(
        {
            "candidate": side.ne(0),
            "side": side,
            "bid_both": bid_both,
            "ask_both": ask_both,
            "conflict": bid_both & ask_both,
        }
    )


def thresholds_for_cell(
    raw: dict[tuple[str, str, str], pd.Series],
    cell: dict[str, float | int],
    cfg: Config,
    *,
    cache: dict[tuple[str, str, str, float], pd.Series] | None = None,
) -> dict[tuple[str, str, str], pd.Series]:
    cache = {} if cache is None else cache
    quantiles = {
        "add": float(cell["q_add"]),
        "withdraw": float(cell["q_withdraw"]),
        "net": float(cell["q_net"]),
        "flicker": float(cell["q_flicker"]),
    }
    output: dict[tuple[str, str, str], pd.Series] = {}
    for (venue, side, metric), values in raw.items():
        key = (venue, side, metric, quantiles[metric])
        if key not in cache:
            cache[key] = lagged_quantile(
                values,
                quantiles[metric],
                window=cfg.rolling_window_rows,
                minimum=cfg.rolling_minimum_rows,
            )
        output[(venue, side, metric)] = cache[key]
    return output


def build_signal(
    dates: pd.Series,
    raw: dict[tuple[str, str, str], pd.Series],
    joint_complete: pd.Series,
    cell: dict[str, float | int],
    cfg: Config,
    *,
    cache: dict[tuple[str, str, str, float], pd.Series] | None = None,
) -> pd.DataFrame:
    state = classify(
        raw,
        joint_complete,
        thresholds_for_cell(raw, cell, cfg, cache=cache),
    )
    branch = pd.Series("none", index=dates.index, dtype="string")
    branch.loc[state["side"].gt(0)] = "bid_refill_compression"
    branch.loc[state["side"].lt(0)] = "ask_refill_compression"
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            **state.to_dict("series"),
            "branch": branch,
            "hold_bars": np.where(state["side"].ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def quarter_schedule(signal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dates = pd.to_datetime(signal["date"])
    side = signal["side"].to_numpy(np.int8)
    rows: list[dict[str, Any]] = []
    for quarter, start, end in QUARTERS:
        period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
            bool
        )
        next_entry = 0
        for signal_position in np.flatnonzero(side & period):
            signal_position = int(signal_position)
            entry_position = signal_position + int(cfg.entry_delay_bars)
            exit_position = entry_position + int(cfg.hold_bars)
            if entry_position < next_entry or exit_position >= len(signal):
                continue
            if not period[entry_position] or not period[exit_position]:
                continue
            rows.append(
                {
                    "quarter": quarter,
                    "signal_position": signal_position,
                    "entry_position": entry_position,
                    "exit_position": exit_position,
                    "signal_date": str(dates.iloc[signal_position]),
                    "entry_date": str(dates.iloc[entry_position]),
                    "exit_date": str(dates.iloc[exit_position]),
                    "side": int(side[signal_position]),
                    "branch": str(signal["branch"].iloc[signal_position]),
                    "hold_bars": int(cfg.hold_bars),
                }
            )
            # Closing an old trade and opening a new one at the same scheduled
            # open is allowed; the old close is ordered first.
            next_entry = exit_position
    return pd.DataFrame(rows)


def event_clock_hash(schedule: pd.DataFrame) -> str:
    records = [
        {
            "signal_position": int(row.signal_position),
            "entry_position": int(row.entry_position),
            "exit_position": int(row.exit_position),
            "side": int(row.side),
            "hold_bars": int(row.hold_bars),
        }
        for row in schedule.itertuples(index=False)
    ]
    return canonical_hash(records)


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    if schedule.empty:
        by_quarter = {name: 0 for name, _, _ in QUARTERS}
        by_month: dict[str, int] = {}
        total = longs = shorts = 0
    else:
        entry_dates = pd.to_datetime(schedule["entry_date"])
        by_quarter = {
            name: int(schedule["quarter"].eq(name).sum()) for name, _, _ in QUARTERS
        }
        by_month = {
            str(key): int(value)
            for key, value in entry_dates.dt.to_period("M").value_counts().sort_index().items()
        }
        total = int(len(schedule))
        longs = int(schedule["side"].gt(0).sum())
        shorts = int(schedule["side"].lt(0).sum())
    h1 = by_quarter["q1"] + by_quarter["q2"]
    h2 = by_quarter["q3"] + by_quarter["q4"]
    long_share = longs / total if total else 0.0
    short_share = shorts / total if total else 0.0
    maximum_month_share = max(by_month.values(), default=total) / total if total else 1.0
    maximum_quarter_share = max(by_quarter.values(), default=total) / total if total else 1.0
    passes = (
        total >= cfg.minimum_events
        and h1 >= cfg.minimum_half_events
        and h2 >= cfg.minimum_half_events
        and all(value >= cfg.minimum_quarter_events for value in by_quarter.values())
        and min(long_share, short_share) >= cfg.minimum_side_share
        and maximum_month_share <= cfg.maximum_month_share
        and maximum_quarter_share <= cfg.maximum_quarter_share
    )
    return {
        "nonoverlap_total": total,
        "by_quarter": by_quarter,
        "by_month": by_month,
        "h1": h1,
        "h2": h2,
        "longs": longs,
        "shorts": shorts,
        "long_share": float(long_share),
        "short_share": float(short_share),
        "maximum_observed_month_share": float(maximum_month_share),
        "maximum_observed_quarter_share": float(maximum_quarter_share),
        "passes_support": bool(passes),
    }


def _one_to_one_matches(
    first: Iterable[int],
    second: Iterable[int],
    *,
    tolerance: int,
) -> tuple[int, list[tuple[int, int]]]:
    if tolerance < 0:
        raise ValueError("overlap tolerance cannot be negative")
    left = sorted(int(value) for value in first)
    right = sorted(int(value) for value in second)
    left_cursor = right_cursor = 0
    pairs: list[tuple[int, int]] = []
    while left_cursor < len(left) and right_cursor < len(right):
        a = left[left_cursor]
        b = right[right_cursor]
        if abs(a - b) <= tolerance:
            pairs.append((a, b))
            left_cursor += 1
            right_cursor += 1
        elif a < b:
            left_cursor += 1
        else:
            right_cursor += 1
    return len(pairs), pairs


def _position_mask(schedule: pd.DataFrame, rows: int = EXPECTED_ROWS) -> np.ndarray:
    mask = np.zeros(rows, dtype=bool)
    for row in schedule.itertuples(index=False):
        start = max(0, int(row.entry_position))
        end = min(rows, int(row.exit_position))
        mask[start:end] = True
    return mask


def overlap_summary(
    current: pd.DataFrame,
    prior: pd.DataFrame,
    cfg: Config,
    *,
    evidentiary: bool = True,
) -> dict[str, Any]:
    current_entries = current["entry_position"].astype(int).tolist()
    prior_entries = prior["entry_position"].astype(int).tolist()
    exact_matches, exact_pairs = _one_to_one_matches(
        current_entries, prior_entries, tolerance=0
    )
    exact_union = len(current_entries) + len(prior_entries) - exact_matches
    tolerant_matches, tolerant_pairs = _one_to_one_matches(
        current_entries, prior_entries, tolerance=cfg.tolerant_bars
    )
    current_mask = _position_mask(current)
    prior_mask = _position_mask(prior)
    position_intersection = int(np.logical_and(current_mask, prior_mask).sum())
    position_union = int(np.logical_or(current_mask, prior_mask).sum())
    exact_jaccard = exact_matches / exact_union if exact_union else 1.0
    candidate_match_share = (
        tolerant_matches / len(current_entries) if current_entries else 1.0
    )
    position_jaccard = (
        position_intersection / position_union if position_union else 1.0
    )
    side_by_current = dict(
        zip(current["entry_position"].astype(int), current["side"].astype(int))
    )
    side_by_prior = dict(
        zip(prior["entry_position"].astype(int), prior["side"].astype(int))
    )
    exact_side_agreement = (
        sum(side_by_current[a] == side_by_prior[b] for a, b in exact_pairs)
        / exact_matches
        if exact_matches
        else None
    )
    passes = (
        exact_jaccard <= cfg.exact_entry_jaccard_maximum
        and candidate_match_share <= cfg.tolerant_candidate_match_share_maximum
        and position_jaccard <= cfg.position_time_jaccard_maximum
    )
    return {
        "current_events": int(len(current_entries)),
        "prior_events": int(len(prior_entries)),
        "exact_entry_matches": int(exact_matches),
        "exact_entry_jaccard": float(exact_jaccard),
        "exact_entry_jaccard_maximum": cfg.exact_entry_jaccard_maximum,
        "tolerance_bars": cfg.tolerant_bars,
        "tolerant_matches": int(tolerant_matches),
        "tolerant_candidate_match_share": float(candidate_match_share),
        "tolerant_candidate_match_share_maximum": (
            cfg.tolerant_candidate_match_share_maximum
        ),
        "position_intersection_bars": position_intersection,
        "position_union_bars": position_union,
        "position_time_jaccard": float(position_jaccard),
        "position_time_jaccard_maximum": cfg.position_time_jaccard_maximum,
        "exact_matched_side_agreement": exact_side_agreement,
        "evidentiary": evidentiary,
        "passes_frozen_overlap_gates": bool(passes),
        "tolerant_pairs_sha256": canonical_hash(tolerant_pairs),
    }


def _near_pressure_schedule(
    shells: pd.DataFrame,
    credibility: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    complete = shells["source_complete"] & credibility["source_complete"]
    venue_scores: list[pd.Series] = []
    for venue in ("um", "cm"):
        bid = (
            shells[f"{venue}_shell_flow_net_m1"].astype(float)
            + 0.5 * shells[f"{venue}_shell_flow_net_m2"].astype(float)
        )
        ask = (
            shells[f"{venue}_shell_flow_net_p1"].astype(float)
            + 0.5 * shells[f"{venue}_shell_flow_net_p2"].astype(float)
        )
        venue_scores.append(
            lagged_robust_zscore(
                (bid - ask).where(complete), window=8_640, minimum=2_016
            )
        )
    score = ((venue_scores[0] + venue_scores[1]) / np.sqrt(2.0)).where(complete)
    threshold = 4.434387570833191
    strong = score.abs().ge(threshold) & score.notna()
    side = np.sign(score).fillna(0.0).astype(np.int8)
    onset = strong & (
        ~strong.shift(1, fill_value=False) | side.ne(side.shift(1, fill_value=0))
    )
    dates = pd.to_datetime(shells["date"])
    rows: list[dict[str, Any]] = []
    for quarter, start, end in QUARTERS:
        period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
            bool
        )
        next_allowed = 0
        quarter_records: list[list[Any]] = []
        for signal_position in np.flatnonzero(onset.to_numpy(bool) & period):
            signal_position = int(signal_position)
            if signal_position < next_allowed:
                continue
            entry_position = signal_position + 1
            exit_position = entry_position + 288
            if exit_position >= len(shells) or not period[exit_position]:
                continue
            rows.append(
                {
                    "quarter": quarter,
                    "signal_position": signal_position,
                    "entry_position": entry_position,
                    "exit_position": exit_position,
                    "signal_date": str(dates.iloc[signal_position]),
                    "entry_date": str(dates.iloc[entry_position]),
                    "exit_date": str(dates.iloc[exit_position]),
                    "side": int(side.iloc[signal_position]),
                    "branch": "near_plain",
                    "hold_bars": 288,
                }
            )
            quarter_records.append(
                [
                    signal_position,
                    entry_position,
                    exit_position,
                    int(side.iloc[signal_position]),
                    str(dates.iloc[entry_position]),
                ]
            )
            next_allowed = exit_position + 1
        digest = hashlib.sha256(
            json.dumps(
                quarter_records, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        if digest != EXPECTED_NEAR_SCHEDULE_HASHES[quarter]:
            raise RuntimeError(f"near-pressure {quarter} clock did not replay")
    return pd.DataFrame(rows)


def replay_prior_clocks(
    shells: pd.DataFrame,
    credibility: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    cclh_signal = cclh.build_signal(credibility, cclh.Config())
    cclh_schedule = pdf._quarterly_schedule(cclh_signal, credibility)
    if pdf._event_clock_sha256(cclh_schedule) != EXPECTED_CCLH_EVENT_HASH:
        raise RuntimeError("CCLH event clock did not replay")

    pdf_signal = pdf.build_signal(credibility, pdf.Config())
    pdf_schedule = pdf._quarterly_schedule(pdf_signal, credibility)
    if rlwc._pdf_event_clock_sha256(pdf_schedule) != EXPECTED_PDF_EVENT_HASH:
        raise RuntimeError("PDF-10 event clock did not replay")

    rlwc_signal, _ = rlwc.build_signal(shells, rlwc.Config())
    rlwc_schedule = pdf._quarterly_schedule(rlwc_signal, shells)
    if int(rlwc_signal["candidate"].sum()) != 0 or not rlwc_schedule.empty:
        raise RuntimeError("frozen RLWC support rejection did not replay")

    near_schedule = _near_pressure_schedule(shells, credibility, Config())
    return {
        "pdf10": pdf_schedule,
        "cclh": cclh_schedule,
        "rlwc144": rlwc_schedule,
        "near_pressure": near_schedule,
    }


def _validate_preregistration(cfg: Config) -> dict[str, Any]:
    path = resolve_existing(cfg.preregistration_result)
    payload = json.loads(path.read_text())
    expected_protocol = prereg.protocol()
    if payload.get("protocol") != expected_protocol:
        raise RuntimeError("CRRC preregistration protocol drifted")
    if payload.get("protocol_hash") != prereg.canonical_hash(expected_protocol):
        raise RuntimeError("CRRC preregistration hash drifted")
    if payload["protocol"]["evidence_boundary"][
        "post_entry_price_return_or_equity_opened"
    ] is not False:
        raise RuntimeError("CRRC preregistration opened an outcome")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "protocol_hash": payload["protocol_hash"],
    }


def _validate_near_manifest(cfg: Config) -> dict[str, Any]:
    path = resolve_existing(cfg.near_manifest)
    payload = json.loads(path.read_text())
    if payload.get("future_outcomes_opened") is not False:
        raise RuntimeError("near-pressure selection manifest opened future outcomes")
    if payload.get("selected_spec") != {
        "feature": "near_plain",
        "hold_bars": 288,
        "quantile": 0.985,
        "threshold": 4.434387570833191,
    }:
        raise RuntimeError("near-pressure selected spec drifted")
    for quarter, expected in EXPECTED_NEAR_SCHEDULE_HASHES.items():
        if payload["selected_schedule_hashes"].get(quarter) != expected:
            raise RuntimeError(f"near-pressure {quarter} manifest hash drifted")
    return {"path": str(path), "sha256": sha256(path)}


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_frozen_config(cfg)
    preregistration = _validate_preregistration(cfg)
    near_manifest = _validate_near_manifest(cfg)
    shells, credibility, source = load_sources(cfg)
    raw, joint_complete = raw_features(shells, credibility)

    cache: dict[tuple[str, str, str, float], pd.Series] = {}
    trials: list[dict[str, Any]] = []
    selected_signal: pd.DataFrame | None = None
    selected_schedule: pd.DataFrame | None = None
    for cell in prereg.SUPPORT_GRID:
        signal = build_signal(
            shells["date"], raw, joint_complete, cell, cfg, cache=cache
        )
        schedule = quarter_schedule(signal, cfg)
        support = support_summary(schedule, cfg)
        if support["nonoverlap_total"] != int(cell["events"]):
            raise RuntimeError(f"support incidence drifted for cell {cell}")
        trials.append({"cell": dict(cell), "support": support})
        if support["passes_support"]:
            key = (
                float(cell["q_add"]),
                float(cell["q_withdraw"]),
                float(cell["q_net"]),
                -float(cell["q_flicker"]),
            )
            selected_key = (
                float(cfg.selected_q_add),
                float(cfg.selected_q_withdraw),
                float(cfg.selected_q_net),
                -float(cfg.selected_q_flicker),
            )
            if key == selected_key:
                selected_signal = signal
                selected_schedule = schedule

    passing = [row for row in trials if row["support"]["passes_support"]]
    if not passing:
        raise RuntimeError("no support-grid cell passed")
    chosen = max(
        passing,
        key=lambda row: (
            float(row["cell"]["q_add"]),
            float(row["cell"]["q_withdraw"]),
            float(row["cell"]["q_net"]),
            -float(row["cell"]["q_flicker"]),
        ),
    )
    if chosen["cell"] != dict(prereg.SUPPORT_GRID[-1]):
        raise RuntimeError("deterministic support selection did not replay")
    if selected_signal is None or selected_schedule is None:
        raise RuntimeError("selected CRRC clock was not built")

    prior_clocks = replay_prior_clocks(shells, credibility)
    independence = {
        name: overlap_summary(
            selected_schedule,
            prior,
            cfg,
            evidentiary=name != "rlwc144",
        )
        for name, prior in prior_clocks.items()
    }
    passes_independence = all(
        item["passes_frozen_overlap_gates"]
        for item in independence.values()
        if item["evidentiary"]
    )
    selected_support = support_summary(selected_schedule, cfg)
    all_gates_pass = selected_support["passes_support"] and passes_independence
    clock_hash = event_clock_hash(selected_schedule)
    event_records = selected_schedule.to_dict("records")
    event_clock = {
        "protocol": "CRRC-72 canonical outcome-blind event-clock freeze",
        "outcomes_opened": False,
        "price_funding_return_or_equity_loaded": False,
        "selection_end_exclusive": str(SELECTION_END),
        "canonical_fields": [
            "signal_position",
            "entry_position",
            "exit_position",
            "side",
            "hold_bars",
        ],
        "event_count": int(len(selected_schedule)),
        "side_counts": {
            "long": int(selected_schedule["side"].gt(0).sum()),
            "short": int(selected_schedule["side"].lt(0).sum()),
        },
        "quarter_counts": selected_support["by_quarter"],
        "event_clock_sha256": clock_hash,
        "events": event_records,
    }
    support_payload = {
        "protocol": {
            "name": prereg.protocol()["name"],
            "support_only": True,
            "outcomes_opened_for_crrc72": False,
            "price_funding_return_or_equity_loaded": False,
            "support_rejected": not all_gates_pass,
            "selection_end_exclusive": str(SELECTION_END),
            "later_windows_sealed": ["test2024", "eval2025", "holdout2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration": preregistration,
            "near_pressure_manifest": near_manifest,
            "qualifier_source": str(Path(__file__).relative_to(Path.cwd())),
            "qualifier_source_sha256": sha256(__file__),
        },
        "source": source,
        "support_selection": {
            "outcomes_used": False,
            "trials": trials,
            "selected_cell": chosen["cell"],
            "selection_rule": prereg.protocol()["support_selection"]["selection_rule"],
        },
        "raw_candidates": {
            "long": int(selected_signal["side"].gt(0).sum()),
            "short": int(selected_signal["side"].lt(0).sum()),
            "conflicts_flattened": int(selected_signal["conflict"].sum()),
        },
        "event_clock_sha256": clock_hash,
        "support": selected_support,
        "prior_clock_counts": {
            name: int(len(clock)) for name, clock in prior_clocks.items()
        },
        "outcome_blind_independence": independence,
        "passes_outcome_blind_independence": bool(passes_independence),
        "all_support_gates_pass": bool(all_gates_pass),
        "next_action": (
            "freeze physical 2023 execution sources and strict evaluator"
            if all_gates_pass
            else "retire before opening any return"
        ),
    }
    return support_payload, event_clock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    args = parser.parse_args()
    cfg = Config(output=args.output, event_clock_output=args.event_clock_output)
    support, clock = run_support(cfg)
    output = Path(cfg.output)
    event_output = Path(cfg.event_clock_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(support, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    event_output.write_text(
        json.dumps(clock, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_crrc72": False,
                "support_rejected": support["protocol"]["support_rejected"],
                "support": support["support"],
                "independence": support["outcome_blind_independence"],
                "output": str(output),
                "event_clock_output": str(event_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
