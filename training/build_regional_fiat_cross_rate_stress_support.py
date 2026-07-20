"""Build the outcome-blind RFXS2-576 source-support decision.

This module may open only the frozen four-close Spot panel, its manifest, the
frozen mechanism documents, and two source-only comparator clocks.  It has no
execution-OHLC or funding input and computes no trading return.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SOURCE_COMMIT = "e576d22c6f2d567d4b40358f755bef4b27c188d4"
STATIC_INPUT_SHA256 = {
    "data/binance_regional_fiat_cross_rate_btc_2020-11_2023/"
    "BTC_regional_fiat_cross_rate_1d_2020-11-01_2023-12-31.csv.gz": (
        "5dbc697c8299ac892295a01302e9f2d883a6e252c8d3d85a8f60f3a369b533d3"
    ),
    "data/binance_regional_fiat_cross_rate_btc_2020-11_2023/"
    "build_manifest.json": (
        "627fdd8298312ea61c2bfaa14d93d623e61d562d64abea0a3769d79c3a68673c"
    ),
    "docs/regional-fiat-cross-rate-stress-mechanism-decision-2026-07-20.md": (
        "c3f7bcfd12c4412be0ad8696b2fa339c709fa94f1a5e61a22cf33c45e4d3ae89"
    ),
    "docs/regional-fiat-cross-rate-stress-rfxs576-source-rejection-2026-07-20.md": (
        "20c016be3b8d1cebfdd4e22fa98d1d29950b75304b1cf67d6cd752a5887ae4c8"
    ),
    "docs/regional-fiat-cross-rate-stress-v2-mechanism-decision-2026-07-20.md": (
        "b9d0bd27f4c2b3b61a23f69bc308d8a6f4ce6292153fd485ea2431f08068e20c"
    ),
    "results/fiat_quote_participation_rotation_clocks_2026-07-17.csv": (
        "54a70cce565d4f1727d095707471235f01345b94179a6c37df9f4c37d1a458a2"
    ),
    "data/stablecoin_denominator_dislocation_clocks_2023.csv.gz": (
        "eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69"
    ),
}
OPENABLE_INPUTS = tuple(STATIC_INPUT_SHA256)
SOURCE_PANEL = Path(OPENABLE_INPUTS[0])
SOURCE_MANIFEST = Path(OPENABLE_INPUTS[1])
FQPR_CLOCKS = Path(OPENABLE_INPUTS[5])
SDDR_CLOCKS = Path(OPENABLE_INPUTS[6])
EVALUATOR_SOURCE = Path("training/build_regional_fiat_cross_rate_stress_support.py")
DEFAULT_OUTPUT = Path(
    "results/regional_fiat_cross_rate_stress_v2_support_2026-07-20.json"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "results/regional_fiat_cross_rate_stress_v2_clocks_2026-07-20.csv"
)

REGION_Z_COLUMNS = {"EUR": "z_eur", "TRY": "z_try", "BRL": "z_brl"}
CANDIDATE_COLUMNS = (
    "clock_name",
    "candidate_id",
    "source_day",
    "decision_time",
    "entry_time",
    "exit_time",
    "state",
    "side",
    "common_z",
    "z_eur",
    "z_try",
    "z_brl",
    "contributors",
)


@dataclass(frozen=True)
class SupportConfig:
    candidate: str = "RFXS2-576"
    baseline_days: int = 180
    threshold: float = 1.0
    bar_minutes: int = 5
    hold_bars: int = 576
    train_start: str = "2021-01-01"
    train_end: str = "2023-01-01"
    selection_start: str = "2023-01-01"
    selection_end: str = "2024-01-01"
    fqpr_start: str = "2021-01-01"
    fqpr_end: str = "2024-01-01"
    sddr_start: str = "2023-09-01"
    sddr_end: str = "2024-01-01"
    spearman_limit: float = 0.50
    fqpr_jaccard_limit: float = 0.20
    fqpr_exposure_correlation_limit: float = 0.40
    sddr_jaccard_limit: float = 0.10
    sddr_exposure_correlation_limit: float = 0.40


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_robust_z(values: pd.Series, window: int = 180) -> pd.Series:
    """Robust z-score against exactly ``window`` strictly prior finite values."""
    if window < 1:
        raise ValueError("robust-z window must be positive")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(numeric), np.nan, dtype=float)
    for index in range(window, len(numeric)):
        current = numeric[index]
        prior = numeric[index - window : index]
        if not np.isfinite(current) or not np.isfinite(prior).all():
            continue
        median = float(np.median(prior))
        mad = float(np.median(np.abs(prior - median)))
        if not np.isfinite(mad) or mad <= 0.0:
            continue
        result[index] = (current - median) / (1.4826 * mad)
    return pd.Series(result, index=values.index, name=values.name, dtype=float)


def _threshold_state(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    state = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric)
    state[finite] = 0.0
    state[finite & (numeric >= threshold)] = 1.0
    state[finite & (numeric <= -threshold)] = -1.0
    return pd.Series(state, index=values.index, dtype=float)


def derive_features(
    panel: pd.DataFrame, cfg: SupportConfig = SupportConfig()
) -> pd.DataFrame:
    required = (
        "date",
        "source_available_not_before",
        "BTCUSDT_close",
        "BTCEUR_close",
        "BTCTRY_close",
        "BTCBRL_close",
        "source_complete",
    )
    if tuple(panel.columns) != required:
        raise ValueError(f"unexpected RFXS2 panel columns: {panel.columns.tolist()}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame["source_available_not_before"] = pd.to_datetime(
        frame["source_available_not_before"], utc=True, errors="raise"
    )
    if not frame["date"].is_monotonic_increasing or not frame["date"].is_unique:
        raise ValueError("RFXS2 source dates are not strictly increasing")
    expected_dates = pd.date_range(
        "2020-11-01", "2024-01-01", freq="1D", inclusive="left", tz="UTC"
    )
    if not pd.DatetimeIndex(frame["date"]).equals(expected_dates):
        missing = expected_dates.difference(pd.DatetimeIndex(frame["date"]))
        extra = pd.DatetimeIndex(frame["date"]).difference(expected_dates)
        raise ValueError(
            "RFXS2 source does not match the exact frozen daily horizon; "
            f"missing={missing[:10].tolist()}, extra={extra[:10].tolist()}"
        )
    expected_availability = frame["date"] + pd.Timedelta(days=1)
    if not frame["source_available_not_before"].equals(expected_availability):
        raise ValueError("RFXS2 source availability boundary changed")
    if not frame["source_complete"].astype(bool).all():
        raise ValueError("RFXS2 source panel contains incomplete days")
    close_columns = [column for column in required if column.endswith("_close")]
    closes = frame[close_columns].to_numpy(float)
    if not np.isfinite(closes).all() or not (closes > 0.0).all():
        raise ValueError("RFXS2 source panel contains invalid closes")

    btc_return = np.log(frame["BTCUSDT_close"].astype(float)).diff()
    frame["btc_return"] = btc_return
    for region, symbol in (("eur", "BTCEUR"), ("try", "BTCTRY"), ("brl", "BTCBRL")):
        regional_return = np.log(frame[f"{symbol}_close"].astype(float)).diff()
        residual = regional_return - btc_return
        frame[f"x_{region}"] = residual
        frame[f"z_{region}"] = strict_robust_z(
            residual, window=cfg.baseline_days
        )
    z_columns = list(REGION_Z_COLUMNS.values())
    all_region_z = frame[z_columns].notna().all(axis=1)
    frame["common_z"] = frame[z_columns].median(axis=1, skipna=False).where(
        all_region_z
    )
    frame["btc_return_z"] = strict_robust_z(
        btc_return, window=cfg.baseline_days
    )
    frame["primary_state"] = _threshold_state(frame["common_z"], cfg.threshold)
    for region, column in REGION_Z_COLUMNS.items():
        frame[f"{region.lower()}_state"] = _threshold_state(
            frame[column], cfg.threshold
        )
    frame["btc_return_state"] = _threshold_state(
        frame["btc_return_z"], cfg.threshold
    )
    sign_state = np.full(len(frame), np.nan, dtype=float)
    z_values = frame[z_columns].to_numpy(float)
    finite_rows = np.isfinite(z_values).all(axis=1)
    sign_state[finite_rows] = 0.0
    sign_state[finite_rows & (z_values > 0.0).all(axis=1)] = 1.0
    sign_state[finite_rows & (z_values < 0.0).all(axis=1)] = -1.0
    frame["three_book_sign_state"] = sign_state
    frame["stale_primary_state"] = frame["primary_state"].shift(1)
    frame["z_EUR"] = frame["z_eur"]
    frame["z_TRY"] = frame["z_try"]
    frame["z_BRL"] = frame["z_brl"]
    frame["state"] = frame["primary_state"]
    return frame


def _contributors(row: pd.Series, *, state: int, threshold: float) -> str:
    contributors: list[str] = []
    for region, column in REGION_Z_COLUMNS.items():
        value = float(row.get(column, row.get(f"z_{region}", np.nan)))
        if (state == 1 and value >= threshold) or (
            state == -1 and value <= -threshold
        ):
            contributors.append(region)
    return "+".join(contributors)


def build_candidates(
    features: pd.DataFrame,
    *,
    clock_name: str = "primary",
    state_column: str | None = None,
    cfg: SupportConfig = SupportConfig(),
) -> pd.DataFrame:
    if state_column is None:
        state_column = "state" if "state" in features else "primary_state"
    if state_column not in features:
        raise ValueError(f"missing RFXS2 state column: {state_column}")
    rows: list[dict[str, Any]] = []
    previous_valid: int | None = None
    for row in features.itertuples(index=False):
        value = getattr(row, state_column)
        if not np.isfinite(value):
            continue
        state = int(value)
        is_event = previous_valid is not None and state != 0 and state != previous_valid
        previous_valid = state
        if not is_event:
            continue
        source_day = _utc(getattr(row, "date"))
        decision = source_day + pd.Timedelta(days=1)
        entry = decision + pd.Timedelta(minutes=cfg.bar_minutes)
        exit_time = entry + pd.Timedelta(
            minutes=cfg.bar_minutes * cfg.hold_bars
        )
        row_series = pd.Series(row._asdict())
        contributors = _contributors(
            row_series, state=state, threshold=cfg.threshold
        )
        rows.append(
            {
                "clock_name": clock_name,
                "candidate_id": f"{clock_name}|{source_day.isoformat()}",
                "source_day": source_day,
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "state": state,
                "side": -state,
                "common_z": float(getattr(row, "common_z", np.nan)),
                "z_eur": float(
                    getattr(row, "z_eur", getattr(row, "z_EUR", np.nan))
                ),
                "z_try": float(
                    getattr(row, "z_try", getattr(row, "z_TRY", np.nan))
                ),
                "z_brl": float(
                    getattr(row, "z_brl", getattr(row, "z_BRL", np.nan))
                ),
                "contributors": contributors,
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def reserve_clock(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"entry_time", "exit_time"}
    if not required.issubset(candidates.columns):
        raise ValueError("unexpected RFXS2 candidate schema")
    clock = candidates.sort_values(
        ["entry_time"]
        + (["candidate_id"] if "candidate_id" in candidates.columns else []),
        kind="mergesort",
    ).reset_index(drop=True)
    reserved: list[bool] = []
    prior_exit: pd.Timestamp | None = None
    for row in clock.itertuples(index=False):
        entry = _utc(row.entry_time)
        exit_time = _utc(row.exit_time)
        accept = prior_exit is None or entry >= prior_exit
        reserved.append(accept)
        if accept:
            prior_exit = exit_time
    clock["reserved"] = reserved
    clock["suppressed_by_overlap"] = ~clock["reserved"]
    return clock


def accepted_for_split(
    clock: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    start_time = _utc(start)
    end_time = _utc(end)
    if start_time >= end_time:
        raise ValueError("split start must precede end")
    entry_time = pd.to_datetime(clock["entry_time"], utc=True, errors="raise")
    exit_time = pd.to_datetime(clock["exit_time"], utc=True, errors="raise")
    expected_hold = pd.Timedelta(minutes=SupportConfig.bar_minutes * SupportConfig.hold_bars)
    if not (exit_time - entry_time).eq(expected_hold).all():
        raise ValueError("RFXS2 clock does not contain exactly 576 five-minute bars")
    if "source_day" in clock:
        source_day = pd.to_datetime(clock["source_day"], utc=True, errors="raise")
        if not entry_time.eq(source_day + pd.Timedelta(days=1, minutes=5)).all():
            raise ValueError("RFXS2 source-to-entry clock changed")
    if "decision_time" in clock:
        decision_time = pd.to_datetime(
            clock["decision_time"], utc=True, errors="raise"
        )
        if "source_day" in clock and not decision_time.eq(
            source_day + pd.Timedelta(days=1)
        ).all():
            raise ValueError("RFXS2 source-to-decision clock changed")
        if not entry_time.eq(decision_time + pd.Timedelta(minutes=5)).all():
            raise ValueError("RFXS2 decision-to-entry clock changed")
    reserved = (
        clock["reserved"].astype(bool)
        if "reserved" in clock
        else pd.Series(True, index=clock.index)
    )
    mask = (
        reserved
        & entry_time.ge(start_time)
        & exit_time.le(end_time)
    )
    if "source_day" in clock:
        mask &= pd.to_datetime(clock["source_day"], utc=True).ge(start_time)
    if "decision_time" in clock:
        mask &= pd.to_datetime(clock["decision_time"], utc=True).ge(start_time)
    return clock.loc[mask].copy()


def deterministic_random_side(entry_time: str | pd.Timestamp) -> int:
    timestamp = _utc(entry_time).isoformat()
    digest = hashlib.sha256(
        f"RFXS2-576-random-side-20260720|{timestamp}".encode()
    ).digest()
    return 1 if digest[0] < 128 else -1


def _clone_primary(
    primary: pd.DataFrame, *, clock_name: str, side_kind: str
) -> pd.DataFrame:
    clone = primary.copy()
    clone["clock_name"] = clock_name
    clone["candidate_id"] = clone["source_day"].map(
        lambda value: f"{clock_name}|{_utc(value).isoformat()}"
    )
    if side_kind == "flip":
        clone["side"] = -clone["side"].astype(int)
    elif side_kind == "random":
        clone["side"] = clone["entry_time"].map(deterministic_random_side)
    else:
        raise ValueError(f"unknown clone side kind: {side_kind}")
    return clone


def build_all_clocks(
    features: pd.DataFrame, cfg: SupportConfig = SupportConfig()
) -> dict[str, pd.DataFrame]:
    definitions = {
        "primary": "primary_state",
        "eur_only": "eur_state",
        "try_only": "try_state",
        "brl_only": "brl_state",
        "three_book_sign_only": "three_book_sign_state",
        "btc_return_shadow": "btc_return_state",
        "stale_one_day": "stale_primary_state",
    }
    clocks = {
        name: reserve_clock(
            build_candidates(
                features, clock_name=name, state_column=column, cfg=cfg
            )
        )
        for name, column in definitions.items()
    }
    clocks["direction_flip"] = _clone_primary(
        clocks["primary"], clock_name="direction_flip", side_kind="flip"
    )
    clocks["deterministic_random_side"] = _clone_primary(
        clocks["primary"],
        clock_name="deterministic_random_side",
        side_kind="random",
    )
    return clocks


def _entry_period_counts(events: pd.DataFrame, frequency: str) -> dict[str, int]:
    if events.empty:
        return {}
    entry = pd.to_datetime(events["entry_time"], utc=True).dt.tz_localize(None)
    counts = entry.dt.to_period(frequency).astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def support_metrics(
    events: pd.DataFrame,
    split_name: str | None = None,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    split_name = split_name or split
    if split_name not in {"train", "selection"}:
        raise ValueError("support split must be train or selection")
    count = int(len(events))
    sides = events["side"].astype(int) if count else pd.Series(dtype=int)
    month_counts = _entry_period_counts(events, "M")
    quarter_counts = _entry_period_counts(events, "Q")
    year_counts = _entry_period_counts(events, "Y")
    max_month_share = max(month_counts.values()) / count if count else None
    contributor_values = events.get("contributors")
    if contributor_values is None and count:
        contributor_values = events.apply(
            lambda row: _contributors(
                row,
                state=int(row["state"]),
                threshold=SupportConfig.threshold,
            ),
            axis=1,
        )
    if contributor_values is None:
        contributor_values = pd.Series(dtype=str)
    region_shares = {
        region: (
            float(
                contributor_values.str.split("+")
                .map(lambda values: region in values)
                .mean()
            )
            if count
            else 0.0
        )
        for region in REGION_Z_COLUMNS
    }
    entry = (
        pd.to_datetime(events["entry_time"], utc=True)
        if count
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    if split_name == "train":
        half_counts = {
            "2021H1": int(((entry >= _utc("2021-01-01")) & (entry < _utc("2021-07-01"))).sum()),
            "2021H2": int(((entry >= _utc("2021-07-01")) & (entry < _utc("2022-01-01"))).sum()),
            "2022H1": int(((entry >= _utc("2022-01-01")) & (entry < _utc("2022-07-01"))).sum()),
            "2022H2": int(((entry >= _utc("2022-07-01")) & (entry < _utc("2023-01-01"))).sum()),
        }
        required_quarters = (
            "2021Q2",
            "2021Q3",
            "2021Q4",
            "2022Q1",
            "2022Q2",
            "2022Q3",
            "2022Q4",
        )
        gates = {
            "accepted_events_at_least_50": count >= 50,
            "2021_events_at_least_18": year_counts.get("2021", 0) >= 18,
            "2022_events_at_least_24": year_counts.get("2022", 0) >= 24,
            "long_events_at_least_15": int((sides == 1).sum()) >= 15,
            "short_events_at_least_15": int((sides == -1).sum()) >= 15,
            "required_quarters_at_least_4": all(
                quarter_counts.get(quarter, 0) >= 4
                for quarter in required_quarters
            ),
            "maximum_entry_month_share_at_most_20pct": (
                max_month_share is not None and max_month_share <= 0.20
            ),
            "each_region_contributes_at_least_40pct": all(
                share >= 0.40 for share in region_shares.values()
            ),
        }
    else:
        half_counts = {
            "2023H1": int((entry < _utc("2023-07-01")).sum()),
            "2023H2": int((entry >= _utc("2023-07-01")).sum()),
        }
        required_quarters = ("2023Q1", "2023Q2", "2023Q3", "2023Q4")
        gates = {
            "accepted_events_at_least_24": count >= 24,
            "each_half_at_least_10": all(value >= 10 for value in half_counts.values()),
            "long_events_at_least_8": int((sides == 1).sum()) >= 8,
            "short_events_at_least_8": int((sides == -1).sum()) >= 8,
            "every_quarter_at_least_4": all(
                quarter_counts.get(quarter, 0) >= 4
                for quarter in required_quarters
            ),
            "maximum_entry_month_share_at_most_25pct": (
                max_month_share is not None and max_month_share <= 0.25
            ),
            "each_region_contributes_at_least_40pct": all(
                share >= 0.40 for share in region_shares.values()
            ),
        }
    output = {
        "accepted_events": count,
        "long_events": int((sides == 1).sum()),
        "short_events": int((sides == -1).sum()),
        "year_counts": year_counts,
        "half_counts": half_counts,
        "quarter_counts": quarter_counts,
        "month_counts": month_counts,
        "maximum_entry_month_share": max_month_share,
        "region_contribution_shares": region_shares,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }
    output["long_count"] = output["long_events"]
    output["short_count"] = output["short_events"]
    output["region_contribution_share"] = output["region_contribution_shares"]
    output["passes_support"] = output["all_gates_pass"]
    return output


def spearman_abs(left: Iterable[float], right: Iterable[float]) -> float:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(pd.Series(list(left)), errors="coerce"),
            "right": pd.to_numeric(pd.Series(list(right)), errors="coerce"),
        }
    ).dropna()
    pair = pair.loc[np.isfinite(pair["left"]) & np.isfinite(pair["right"])]
    if len(pair) < 3:
        return float("nan")
    ranks = pair.rank(method="average")
    if ranks["left"].nunique() < 2 or ranks["right"].nunique() < 2:
        return float("nan")
    value = float(ranks["left"].corr(ranks["right"], method="pearson"))
    return abs(value) if np.isfinite(value) else float("nan")


def _entry_values(values: Iterable[Any] | pd.DataFrame) -> Iterable[Any]:
    if not isinstance(values, pd.DataFrame):
        return values
    frame = values
    if "clock_name" in frame:
        frame = frame.loc[frame["clock_name"].eq("primary")]
    for column in ("entry_time", "entry_ts"):
        if column in frame:
            return frame[column]
    if frame.empty:
        return []
    raise ValueError("entry timestamp column is missing")


def _timestamp_set(values: Iterable[Any] | pd.DataFrame) -> set[int]:
    result: set[int] = set()
    for value in _entry_values(values):
        result.add(int(_utc(value).value))
    return result


def exact_jaccard(
    left: Iterable[Any] | pd.DataFrame,
    right: Iterable[Any] | pd.DataFrame,
) -> float:
    left_set = _timestamp_set(left)
    right_set = _timestamp_set(right)
    union = left_set | right_set
    if not union:
        return float("nan")
    return len(left_set & right_set) / len(union)


def _normalize_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "side"])
    entry_column = next(
        (column for column in ("entry_time", "entry_ts") if column in frame),
        None,
    )
    exit_column = next(
        (column for column in ("exit_time", "exit_ts") if column in frame),
        None,
    )
    side_column = next(
        (column for column in ("side", "direction") if column in frame),
        None,
    )
    if entry_column is None or exit_column is None or side_column is None:
        raise ValueError("interval comparator schema is incomplete")
    output = frame[[entry_column, exit_column, side_column]].rename(
        columns={
            entry_column: "entry_time",
            exit_column: "exit_time",
            side_column: "side",
        }
    )
    side_map = {"long": 1, "short": -1, "LONG": 1, "SHORT": -1}
    output["side"] = output["side"].map(
        lambda value: side_map.get(value, value)
    )
    return output


def _exposure_vector(
    intervals: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: pd.Timedelta,
) -> np.ndarray:
    size_float = (end - start) / frequency
    size = int(size_float)
    if size_float != size or size <= 0:
        raise ValueError("exposure horizon is not an exact positive grid")
    exposure = np.zeros(size, dtype=float)
    for row in intervals.itertuples(index=False):
        entry = _utc(row.entry_time)
        exit_time = _utc(row.exit_time)
        side = int(row.side)
        if side not in {-1, 1}:
            raise ValueError("exposure side must be +/-1")
        if entry < start or exit_time > end or entry >= exit_time:
            raise ValueError("exposure interval is outside its horizon")
        left_float = (entry - start) / frequency
        right_float = (exit_time - start) / frequency
        left = int(left_float)
        right = int(right_float)
        if left_float != left or right_float != right:
            raise ValueError("exposure interval is not aligned to the grid")
        if np.any(exposure[left:right] != 0.0):
            raise ValueError("exposure intervals overlap")
        exposure[left:right] = side
    return exposure


def signed_exposure_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    frequency: str = "5min",
) -> float:
    start_time = _utc(start)
    end_time = _utc(end)
    step = pd.Timedelta(frequency)
    left_vector = _exposure_vector(
        _normalize_intervals(left),
        start=start_time,
        end=end_time,
        frequency=step,
    )
    right_vector = _exposure_vector(
        _normalize_intervals(right),
        start=start_time,
        end=end_time,
        frequency=step,
    )
    if np.std(left_vector) == 0.0 or np.std(right_vector) == 0.0:
        return float("nan")
    value = float(np.corrcoef(left_vector, right_vector)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _contained_intervals(
    frame: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    start_time = _utc(start)
    end_time = _utc(end)
    output = frame.copy()
    output["entry_time"] = pd.to_datetime(output["entry_time"], utc=True, errors="raise")
    output["exit_time"] = pd.to_datetime(output["exit_time"], utc=True, errors="raise")
    return output.loc[
        output["entry_time"].ge(start_time) & output["exit_time"].le(end_time)
    ].copy()


def _normalize_fqpr(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"clock_name", "entry_time", "exit_time", "side"}
    if not required.issubset(frame.columns):
        raise ValueError("FQPR comparator schema changed")
    primary = frame.loc[frame["clock_name"].eq("primary"), list(required)].copy()
    side_map = {"LONG": 1, "SHORT": -1}
    primary["side"] = primary["side"].map(side_map)
    if primary["side"].isna().any():
        raise ValueError("FQPR comparator side changed")
    return primary[["entry_time", "exit_time", "side"]]


def _normalize_sddr(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate", "control", "entry_time", "exit_time", "side"}
    if not required.issubset(frame.columns):
        raise ValueError("SDDR comparator schema changed")
    primary = frame.loc[
        frame["candidate"].eq("SDDR-12") & frame["control"].eq("primary"),
        ["entry_time", "exit_time", "side"],
    ].copy()
    primary["side"] = pd.to_numeric(primary["side"], errors="raise").astype(int)
    if not primary["side"].isin([-1, 1]).all():
        raise ValueError("SDDR comparator side changed")
    return primary


def _novelty(
    features: pd.DataFrame,
    primary: pd.DataFrame,
    fqpr: pd.DataFrame,
    sddr: pd.DataFrame,
    cfg: SupportConfig,
) -> dict[str, Any]:
    train_mask = features["date"].ge(_utc(cfg.train_start)) & features["date"].lt(
        _utc(cfg.train_end)
    )
    selection_mask = features["date"].ge(
        _utc(cfg.selection_start)
    ) & features["date"].lt(_utc(cfg.selection_end))
    train_spearman = spearman_abs(
        features.loc[train_mask, "common_z"],
        features.loc[train_mask, "btc_return_z"],
    )
    selection_spearman = spearman_abs(
        features.loc[selection_mask, "common_z"],
        features.loc[selection_mask, "btc_return_z"],
    )

    primary_fqpr = accepted_for_split(primary, cfg.fqpr_start, cfg.fqpr_end)
    fqpr_common = _contained_intervals(fqpr, cfg.fqpr_start, cfg.fqpr_end)
    fqpr_jaccard = exact_jaccard(
        primary_fqpr["entry_time"], fqpr_common["entry_time"]
    )
    fqpr_corr = signed_exposure_correlation(
        primary_fqpr,
        fqpr_common,
        start=cfg.fqpr_start,
        end=cfg.fqpr_end,
    )

    primary_sddr = accepted_for_split(primary, cfg.sddr_start, cfg.sddr_end)
    sddr_common = _contained_intervals(sddr, cfg.sddr_start, cfg.sddr_end)
    sddr_jaccard = exact_jaccard(
        primary_sddr["entry_time"], sddr_common["entry_time"]
    )
    sddr_corr = signed_exposure_correlation(
        primary_sddr,
        sddr_common,
        start=cfg.sddr_start,
        end=cfg.sddr_end,
    )

    values = {
        "train_abs_spearman_common_z_vs_btc_return_z": train_spearman,
        "selection_abs_spearman_common_z_vs_btc_return_z": selection_spearman,
        "fqpr_exact_entry_jaccard": fqpr_jaccard,
        "fqpr_signed_exposure_correlation": fqpr_corr,
        "fqpr_abs_signed_exposure_correlation": abs(fqpr_corr),
        "sddr_exact_entry_jaccard": sddr_jaccard,
        "sddr_signed_exposure_correlation": sddr_corr,
        "sddr_abs_signed_exposure_correlation": abs(sddr_corr),
    }
    gates = {
        "train_abs_spearman_at_most_0_50": (
            np.isfinite(train_spearman) and train_spearman <= cfg.spearman_limit
        ),
        "selection_abs_spearman_at_most_0_50": (
            np.isfinite(selection_spearman)
            and selection_spearman <= cfg.spearman_limit
        ),
        "fqpr_exact_entry_jaccard_at_most_0_20": (
            np.isfinite(fqpr_jaccard) and fqpr_jaccard <= cfg.fqpr_jaccard_limit
        ),
        "fqpr_abs_signed_exposure_correlation_at_most_0_40": (
            np.isfinite(fqpr_corr)
            and abs(fqpr_corr) <= cfg.fqpr_exposure_correlation_limit
        ),
        "sddr_exact_entry_jaccard_at_most_0_10": (
            np.isfinite(sddr_jaccard) and sddr_jaccard <= cfg.sddr_jaccard_limit
        ),
        "sddr_abs_signed_exposure_correlation_at_most_0_40": (
            np.isfinite(sddr_corr)
            and abs(sddr_corr) <= cfg.sddr_exposure_correlation_limit
        ),
    }
    return {"values": values, "gates": gates, "all_gates_pass": all(gates.values())}


def _control_diagnostics(
    clocks: dict[str, pd.DataFrame], cfg: SupportConfig
) -> dict[str, Any]:
    start = cfg.fqpr_start
    end = cfg.fqpr_end
    primary = accepted_for_split(clocks["primary"], start, end)
    diagnostics: dict[str, Any] = {}
    for name, clock in clocks.items():
        accepted = accepted_for_split(clock, start, end)
        month_counts = _entry_period_counts(accepted, "M")
        maximum_month_share = (
            max(month_counts.values()) / len(accepted) if len(accepted) else None
        )
        correlation = signed_exposure_correlation(
            primary,
            accepted,
            start=start,
            end=end,
        )
        diagnostics[name] = {
            "raw_candidates": int(len(clock)),
            "reserved_events": int(clock["reserved"].sum()),
            "accepted_2021_2023": int(len(accepted)),
            "accepted_train": int(
                len(accepted_for_split(clock, cfg.train_start, cfg.train_end))
            ),
            "accepted_selection": int(
                len(
                    accepted_for_split(
                        clock, cfg.selection_start, cfg.selection_end
                    )
                )
            ),
            "long_events": int((accepted["side"].astype(int) == 1).sum()),
            "short_events": int((accepted["side"].astype(int) == -1).sum()),
            "maximum_entry_month_share": maximum_month_share,
            "exact_entry_jaccard_with_primary": exact_jaccard(
                primary["entry_time"], accepted["entry_time"]
            ),
            "signed_exposure_correlation_with_primary": correlation,
        }
    return diagnostics


def evaluate_source_support(
    panel: pd.DataFrame,
    fqpr_frame: pd.DataFrame,
    sddr_frame: pd.DataFrame,
    cfg: SupportConfig = SupportConfig(),
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = derive_features(panel, cfg)
    clocks = build_all_clocks(features, cfg)
    primary = clocks["primary"]
    train = accepted_for_split(primary, cfg.train_start, cfg.train_end)
    selection = accepted_for_split(
        primary, cfg.selection_start, cfg.selection_end
    )
    train_metrics = support_metrics(train, "train")
    selection_metrics = support_metrics(selection, "selection")
    novelty = _novelty(
        features,
        primary,
        _normalize_fqpr(fqpr_frame),
        _normalize_sddr(sddr_frame),
        cfg,
    )
    all_pass = (
        train_metrics["all_gates_pass"]
        and selection_metrics["all_gates_pass"]
        and novelty["all_gates_pass"]
    )
    combined_clocks: list[pd.DataFrame] = []
    for name, clock in clocks.items():
        output = clock.copy()
        output["accepted_split"] = ""
        train_index = accepted_for_split(
            clock, cfg.train_start, cfg.train_end
        ).index
        selection_index = accepted_for_split(
            clock, cfg.selection_start, cfg.selection_end
        ).index
        output.loc[train_index, "accepted_split"] = "train"
        output.loc[selection_index, "accepted_split"] = "selection"
        combined_clocks.append(output)
    clock_output = pd.concat(combined_clocks, ignore_index=True).sort_values(
        ["clock_name", "entry_time", "candidate_id"], kind="mergesort"
    )
    first_feature = features.loc[features["common_z"].notna(), "date"]
    payload = {
        "schema_version": 1,
        "candidate": cfg.candidate,
        "decision": "PASS" if all_pass else "REJECT",
        "config": asdict(cfg),
        "protocol": {
            "source_only": True,
            "execution_ohlc_opened": False,
            "funding_opened": False,
            "post_2023_source_opened": False,
            "future_return_opened": False,
            "pnl_cagr_mdd_opened": False,
            "outcomes_opened": False,
            "signed_exposure_gate_uses_absolute_pearson_magnitude": True,
        },
        "feature_diagnostics": {
            "source_rows": int(len(features)),
            "first_source_day": features["date"].iloc[0].isoformat(),
            "last_source_day": features["date"].iloc[-1].isoformat(),
            "valid_common_z_days": int(features["common_z"].notna().sum()),
            "first_valid_common_z_day": (
                first_feature.iloc[0].isoformat() if len(first_feature) else None
            ),
        },
        "train": train_metrics,
        "selection": selection_metrics,
        "novelty": novelty,
        "controls": _control_diagnostics(clocks, cfg),
        "all_source_support_gates_pass": bool(all_pass),
        "next_stage_authorized": "strict_evaluator" if all_pass else None,
    }
    return payload, clock_output


def evaluate(
    *,
    source_panel: pd.DataFrame,
    comparators: dict[str, pd.DataFrame] | None = None,
    output_dir: str | Path | None = None,
    write: bool = False,
    cfg: SupportConfig = SupportConfig(),
) -> dict[str, Any]:
    """Injected source-only evaluation seam used by synthetic regression tests."""
    comparators = comparators or {}
    if {"fqpr", "sddr"}.issubset(comparators):
        payload, clocks = evaluate_source_support(
            source_panel, comparators["fqpr"], comparators["sddr"], cfg
        )
    else:
        features = derive_features(source_panel, cfg)
        all_clocks = build_all_clocks(features, cfg)
        primary = all_clocks["primary"]
        payload = {
            "candidate": cfg.candidate,
            "protocol": {
                "source_only": True,
                "comparators_injected": False,
                "outcomes_opened": False,
            },
            "train": support_metrics(
                accepted_for_split(primary, cfg.train_start, cfg.train_end),
                "train",
            ),
            "selection": support_metrics(
                accepted_for_split(
                    primary, cfg.selection_start, cfg.selection_end
                ),
                "selection",
            ),
        }
        nonempty_clocks = [clock for clock in all_clocks.values() if not clock.empty]
        clocks = (
            pd.concat(nonempty_clocks, ignore_index=True)
            if nonempty_clocks
            else pd.DataFrame()
        )
    if write:
        if output_dir is None:
            raise ValueError("injected evaluation write requires output_dir")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rfxs2_injected_support.json").write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
        )
        clocks.to_csv(directory / "rfxs2_injected_clocks.csv", index=False)
    return payload


def _verify_static_inputs() -> None:
    for path, expected in STATIC_INPUT_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen RFXS2 support input changed: {path}")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest.get("combined_sha256") != STATIC_INPUT_SHA256[str(SOURCE_PANEL)]:
        raise ValueError("RFXS2 source manifest no longer binds the panel")
    protocol = manifest.get("protocol", {})
    if (
        protocol.get("execution_ohlc_opened") is not False
        or protocol.get("funding_opened") is not False
        or protocol.get("outcomes_opened") is not False
    ):
        raise ValueError("RFXS2 source manifest crossed the outcome boundary")


def _committed_evaluator() -> tuple[str, str]:
    repository_root = Path(__file__).resolve().parents[1]
    source_path = Path(__file__).resolve()
    relative = source_path.relative_to(repository_root).as_posix()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
        committed = subprocess.run(
            ["git", "show", f"{head}:{relative}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ValueError("cannot bind RFXS2 support evaluator to Git") from exc
    current_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if hashlib.sha256(committed).hexdigest() != current_hash:
        raise ValueError("RFXS2 support evaluator is not committed at HEAD")
    return head, current_hash


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return _utc(value).isoformat()
    return value


def run(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    clocks_output: str | Path = DEFAULT_CLOCK_OUTPUT,
    cfg: SupportConfig = SupportConfig(),
) -> dict[str, Any]:
    _verify_static_inputs()
    evaluator_commit, evaluator_hash = _committed_evaluator()
    panel = pd.read_csv(SOURCE_PANEL)
    fqpr = pd.read_csv(FQPR_CLOCKS)
    sddr = pd.read_csv(SDDR_CLOCKS)
    payload, clocks = evaluate_source_support(panel, fqpr, sddr, cfg)
    payload["source_commit"] = SOURCE_COMMIT
    payload["evaluator"] = {
        "commit": evaluator_commit,
        "path": str(EVALUATOR_SOURCE),
        "sha256": evaluator_hash,
    }
    payload["static_input_sha256"] = STATIC_INPUT_SHA256
    output_path = Path(output)
    clocks_path = Path(clocks_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clocks_path.parent.mkdir(parents=True, exist_ok=True)
    clocks.to_csv(
        clocks_path,
        index=False,
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )
    payload["clock_output"] = str(clocks_path)
    payload["clock_sha256"] = _sha256(clocks_path)
    safe_payload = _json_safe(payload)
    output_path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    safe_payload["output"] = str(output_path)
    safe_payload["output_sha256"] = _sha256(output_path)
    return safe_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clocks-output", default=str(DEFAULT_CLOCK_OUTPUT))
    args = parser.parse_args()
    result = run(output=args.output, clocks_output=args.clocks_output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "train": result["train"],
                "selection": result["selection"],
                "novelty": result["novelty"],
                "clock_sha256": result["clock_sha256"],
                "output_sha256": result["output_sha256"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
