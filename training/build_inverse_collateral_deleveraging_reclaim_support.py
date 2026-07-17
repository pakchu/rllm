"""Freeze outcome-blind ICDR-144 support and falsification clocks.

This module reads only official Binance positioning metrics and their timestamp
grid.  It never loads execution OHLC, post-entry returns, funding cash flow,
CAGR, or drawdown.  The purge quantile is selected from 2021-2022 support only;
2023 can only accept or reject that frozen quantile and cannot choose a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_inverse_collateral_deleveraging_reclaim as prereg


PREREGISTRATION_COMMIT = "a46e13c86daa706e01c28c6c186e57ca9ff93866"
PREREGISTRATION_SOURCE_SHA256 = (
    "dcc1abb4a0d3adc667cb594cec711ca2e9d0d05069ef74b950c96a4b1e913531"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/inverse-collateral-deleveraging-reclaim-preregistration-2026-07-17.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "fd2cca63cad2631ec64cbb6bc3b9c6282424fdd5528744e7befba21619e62417"
)
PREREGISTRATION_RESULT = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_RESULT_SHA256 = (
    "2e6907185c36d16b2a82d87e57c689caa97910bb6d9e04d098a1e9e40dd95107"
)
DEFAULT_OUTPUT = (
    "results/inverse_collateral_deleveraging_reclaim_support_2026-07-17.json"
)
DEFAULT_CLOCK = "results/inverse_collateral_deleveraging_reclaim_clocks_2026-07-17.csv"
START = pd.Timestamp("2021-07-08")
TRAIN_END = pd.Timestamp("2023-01-01")
END_EXCLUSIVE = pd.Timestamp("2024-01-01")
METRIC_COLUMNS = (
    "um_sum_open_interest_value",
    "um_sum_taker_long_short_vol_ratio",
    "cm_sum_open_interest",
    "cm_sum_taker_long_short_vol_ratio",
)
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "cm_only_oi",
    "no_taker_gap",
    "no_reclaim",
    "no_oi_stop",
    "um_matched",
    "one_hour_signal_delay",
    "one_day_shifted_clock",
    "random_side",
)
JACCARD_CONTROLS = (
    "cm_only_oi",
    "no_taker_gap",
    "no_reclaim",
    "no_oi_stop",
    "um_matched",
    "one_hour_signal_delay",
    "one_day_shifted_clock",
)
CLOCK_COLUMNS = (
    "policy_name",
    "purge_quantile",
    "setup_position",
    "confirmation_position",
    "signal_position",
    "entry_position",
    "exit_position",
    "setup_date",
    "confirmation_date",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "branch",
    "delay_bars",
    "hold_bars",
)
EVENT_COLUMNS = (
    "setup_position",
    "confirmation_position",
    "signal_position",
    "side",
    "branch",
    "delay_bars",
    "hold_bars",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clock_bytes(schedule: pd.DataFrame) -> bytes:
    return (
        schedule[list(CLOCK_COLUMNS)]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (Path(prereg.__file__), PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen ICDR-144 preregistration changed: {path}")
    payload = json.loads(PREREGISTRATION_RESULT.read_text())
    prereg.validate_manifest(payload, verify_sources=False)
    if payload.get("outcomes_opened") is not False:
        raise ValueError("ICDR-144 outcomes opened before support freeze")
    if payload.get("policy") != prereg.policy_payload():
        raise ValueError("ICDR-144 support policy differs from preregistration")
    return payload


def _assert_complete_grid(dates: pd.Series) -> None:
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(
            "ICDR metrics timestamps must be nonempty, unique, and ordered"
        )
    if not dates.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError(
            "ICDR metrics timestamps must form a complete five-minute grid"
        )
    if dates.iloc[0] != START or dates.iloc[-1] != END_EXCLUSIVE - pd.Timedelta(
        minutes=5
    ):
        raise ValueError("ICDR metrics timestamps differ from the frozen support grid")


def quarantine_mask(source_available: pd.Series, post_gap_bars: int) -> pd.Series:
    """Quarantine an unavailable bar and exactly the following configured bars."""
    if post_gap_bars < 0:
        raise ValueError("post-gap quarantine cannot be negative")
    return (
        (~source_available.astype(bool))
        .astype(np.int8)
        .rolling(post_gap_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )


def prior_clean_quantile(
    values: pd.Series,
    eligible: pd.Series,
    *,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling quantile over finite eligible observations strictly before t."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if not 1 <= min_periods <= window:
        raise ValueError("ICDR rolling baseline is invalid")
    prior = pd.to_numeric(values, errors="coerce").where(eligible.astype(bool)).shift(1)
    return prior.rolling(window, min_periods=min_periods).quantile(quantile)


def load_support_frame(
    policy: prereg.Policy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registration = verify_preregistration()
    source = registration["source_contract"]
    for key, hash_key in (
        ("metrics", "metrics_sha256"),
        ("metrics_manifest", "metrics_manifest_sha256"),
    ):
        if _sha256(source[key]) != source[hash_key]:
            raise ValueError(f"ICDR-144 frozen source changed: {source[key]}")

    metrics_manifest = json.loads(Path(source["metrics_manifest"]).read_text())
    protocol = metrics_manifest.get("protocol", {})
    if (
        protocol.get("outcomes_opened") is not False
        or protocol.get("source_only") is not True
    ):
        raise ValueError("ICDR metrics manifest is not outcome-blind")
    if protocol.get("end_exclusive") != "2024-01-01 00:00:00":
        raise ValueError("ICDR metrics manifest touches a sealed interval")

    usecols = ["date", *METRIC_COLUMNS, "source_complete"]
    frame = pd.read_csv(
        source["metrics"],
        compression="gzip",
        usecols=usecols,
        parse_dates=["date"],
    )
    _assert_complete_grid(frame["date"])
    numeric = frame[list(METRIC_COLUMNS)].apply(pd.to_numeric, errors="raise")
    finite = pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=frame.index)
    positive = numeric.gt(0.0).all(axis=1)
    source_available = frame["source_complete"].eq(True) & finite & positive
    frame[list(METRIC_COLUMNS)] = numeric
    frame["source_available"] = source_available
    frame["quarantined"] = quarantine_mask(
        source_available, policy.post_gap_quarantine_bars
    )
    source_metadata = {
        "metrics_sha256": _sha256(source["metrics"]),
        "metrics_manifest_sha256": _sha256(source["metrics_manifest"]),
        "market_expected_sha256_not_opened": source["market_sha256"],
        "market_manifest_expected_sha256_not_opened": source["market_manifest_sha256"],
        "funding_expected_sha256_not_opened": source["funding_sha256"],
        "funding_manifest_expected_sha256_not_opened": source[
            "funding_manifest_sha256"
        ],
        "columns_loaded": usecols,
        "price_or_outcome_columns_loaded": [],
        "rows": int(len(frame)),
        "source_available_rows": int(source_available.sum()),
        "unavailable_rows": int((~source_available).sum()),
        "quarantined_rows": int(frame["quarantined"].sum()),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
    }
    return frame, source_metadata


def build_features(frame: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    """Build source-only causal ICDR state and all frozen lagged thresholds."""
    output = frame[["date", "source_available", "quarantined"]].copy()
    clean = ~frame["quarantined"].astype(bool)
    um_oi = frame["um_sum_open_interest_value"].astype(float)
    cm_oi = frame["cm_sum_open_interest"].astype(float)
    um_taker = frame["um_sum_taker_long_short_vol_ratio"].astype(float)
    cm_taker = frame["cm_sum_taker_long_short_vol_ratio"].astype(float)
    u = np.log(um_oi)
    c = np.log(cm_oi)
    tu_raw = np.log(um_taker)
    tc_raw = np.log(cm_taker)
    du = u - u.shift(policy.oi_change_bars)
    dc = c - c.shift(policy.oi_change_bars)
    du_one = u - u.shift(1)
    dc_one = c - c.shift(1)
    tu = (
        tu_raw.where(clean)
        .rolling(
            policy.taker_smoothing_bars,
            min_periods=policy.taker_smoothing_bars,
        )
        .mean()
    )
    tc = (
        tc_raw.where(clean)
        .rolling(
            policy.taker_smoothing_bars,
            min_periods=policy.taker_smoothing_bars,
        )
        .mean()
    )
    purge = du - dc
    cm_sell = -tc
    taker_gap = tu - tc
    um_sell = -tu
    required_clean = clean & clean.shift(policy.oi_change_bars, fill_value=False)
    derived = pd.concat(
        [du, dc, du_one, dc_one, tu, tc, purge, cm_sell, taker_gap, um_sell], axis=1
    )
    finite = pd.Series(np.isfinite(derived.to_numpy()).all(axis=1), index=frame.index)
    eligible = required_clean & finite
    output = output.assign(
        clean=clean,
        eligible=eligible,
        um_taker_ratio=um_taker,
        cm_taker_ratio=cm_taker,
        du=du,
        dc=dc,
        du_one=du_one,
        dc_one=dc_one,
        tu=tu,
        tc=tc,
        relative_purge=purge,
        cm_sell_stress=cm_sell,
        cm_specific_sell_gap=taker_gap,
        um_sell_stress=um_sell,
    )
    pq = lambda values, q: prior_clean_quantile(  # noqa: E731
        values,
        eligible,
        quantile=q,
        window=policy.baseline_bars,
        min_periods=policy.baseline_min_periods,
    )
    output["cm_sell_q90"] = pq(cm_sell, policy.sell_stress_quantile)
    output["taker_gap_q90"] = pq(taker_gap, policy.sell_gap_quantile)
    output["um_sell_q90"] = pq(um_sell, policy.sell_stress_quantile)
    for quantile in policy.purge_quantiles:
        tag = _quantile_tag(quantile)
        output[f"purge_q{tag}"] = pq(purge, quantile)
        output[f"cm_contraction_q{tag}"] = pq(-dc, quantile)
        output[f"um_contraction_q{tag}"] = pq(-du, quantile)
    return output


def _quantile_tag(quantile: float) -> str:
    return str(int(round(quantile * 1_000))).zfill(3)


def setup_onsets(state: pd.Series) -> np.ndarray:
    active = state.astype(bool)
    onset = active & ~active.shift(1, fill_value=False)
    return np.flatnonzero(onset.to_numpy())


def confirmed_events(
    setup_state: pd.Series,
    confirmation: pd.Series,
    *,
    policy: prereg.Policy,
    branch: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Take the first confirmation while suppressing replacement setups."""
    accepted: list[int] = []
    records: list[dict[str, Any]] = []
    next_setup = 0
    confirmation_values = confirmation.astype(bool).to_numpy()
    rows = len(setup_state)
    for setup in setup_onsets(setup_state):
        setup = int(setup)
        if setup < next_setup:
            continue
        accepted.append(setup)
        end = min(setup + policy.confirmation_window_bars, rows - 1)
        candidates = np.flatnonzero(confirmation_values[setup + 1 : end + 1])
        if candidates.size:
            signal = setup + 1 + int(candidates[0])
            records.append(
                {
                    "setup_position": setup,
                    "confirmation_position": signal,
                    "signal_position": signal,
                    "side": 1,
                    "branch": branch,
                    "delay_bars": policy.execution_delay_bars,
                    "hold_bars": policy.hold_bars,
                }
            )
            next_setup = signal + 1
        else:
            next_setup = end + 1
    return pd.DataFrame(records, columns=EVENT_COLUMNS), np.asarray(
        accepted, dtype=np.int64
    )


def immediate_events_from_setups(
    accepted_setups: np.ndarray,
    *,
    policy: prereg.Policy,
    branch: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    setups = np.asarray(accepted_setups, dtype=np.int64)
    records = [
        {
            "setup_position": int(setup),
            "confirmation_position": -1,
            "signal_position": int(setup),
            "side": 1,
            "branch": branch,
            "delay_bars": policy.execution_delay_bars,
            "hold_bars": policy.hold_bars,
        }
        for setup in setups
    ]
    return pd.DataFrame(records, columns=EVENT_COLUMNS), setups


def _shift_events(events: pd.DataFrame, bars: int, branch: str) -> pd.DataFrame:
    shifted = events.copy()
    shifted["signal_position"] = shifted["signal_position"].astype(np.int64) + bars
    shifted["branch"] = branch
    return shifted


def classify_events(
    features: pd.DataFrame,
    policy: prereg.Policy,
    *,
    purge_quantile: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, int]]:
    if purge_quantile not in policy.purge_quantiles:
        raise ValueError("ICDR purge quantile is outside the frozen grid")
    tag = _quantile_tag(purge_quantile)
    common = (
        features["eligible"].astype(bool)
        & features["cm_sell_q90"].notna()
        & features["taker_gap_q90"].notna()
        & features[f"purge_q{tag}"].notna()
        & features[f"cm_contraction_q{tag}"].notna()
        & features[f"um_contraction_q{tag}"].notna()
        & features["um_sell_q90"].notna()
    )
    cm_sell = features["cm_sell_stress"].ge(features["cm_sell_q90"])
    gap = features["cm_specific_sell_gap"].ge(features["taker_gap_q90"])
    primary_setup = (
        common
        & features["dc"].lt(0.0)
        & features["relative_purge"].ge(features[f"purge_q{tag}"])
        & cm_sell
        & gap
    )
    cm_only_setup = (
        common
        & features["dc"].lt(0.0)
        & (-features["dc"]).ge(features[f"cm_contraction_q{tag}"])
        & cm_sell
        & gap
    )
    no_gap_setup = (
        common
        & features["dc"].lt(0.0)
        & features["relative_purge"].ge(features[f"purge_q{tag}"])
        & cm_sell
    )
    um_setup = (
        common
        & features["du"].lt(0.0)
        & (-features["du"]).ge(features[f"um_contraction_q{tag}"])
        & features["um_sell_stress"].ge(features["um_sell_q90"])
    )
    confirmation_clean = features["clean"].astype(bool) & features["clean"].shift(
        1, fill_value=False
    )
    cm_reclaim_without_oi = (
        confirmation_clean
        & features["cm_taker_ratio"].ge(1.0)
        & features["cm_taker_ratio"].ge(features["um_taker_ratio"])
    )
    cm_reclaim = cm_reclaim_without_oi & features["dc_one"].ge(0.0)
    um_reclaim = (
        confirmation_clean
        & features["um_taker_ratio"].ge(1.0)
        & features["du_one"].ge(0.0)
    )

    primary, primary_setups = confirmed_events(
        primary_setup, cm_reclaim, policy=policy, branch="primary"
    )
    cm_only, cm_only_setups = confirmed_events(
        cm_only_setup, cm_reclaim, policy=policy, branch="cm_only_oi"
    )
    no_gap, no_gap_setups = confirmed_events(
        no_gap_setup, cm_reclaim, policy=policy, branch="no_taker_gap"
    )
    no_reclaim, no_reclaim_setups = immediate_events_from_setups(
        primary_setups, policy=policy, branch="no_reclaim"
    )
    no_oi_stop, no_oi_stop_setups = confirmed_events(
        primary_setup,
        cm_reclaim_without_oi,
        policy=policy,
        branch="no_oi_stop",
    )
    um_matched, um_matched_setups = confirmed_events(
        um_setup, um_reclaim, policy=policy, branch="um_matched"
    )
    direction_flip = primary.copy()
    direction_flip["side"] = -1
    direction_flip["branch"] = "direction_flip"
    delayed = _shift_events(primary, 12, "one_hour_signal_delay")
    shifted = _shift_events(primary, 288, "one_day_shifted_clock")
    random_side = primary.copy()
    if not random_side.empty:
        rng = np.random.default_rng(20_260_717)
        random_side["side"] = rng.choice(
            np.array([-1, 1], dtype=np.int8), size=len(random_side)
        )
    random_side["branch"] = "random_side"
    events = {
        "primary": primary,
        "direction_flip": direction_flip,
        "cm_only_oi": cm_only,
        "no_taker_gap": no_gap,
        "no_reclaim": no_reclaim,
        "no_oi_stop": no_oi_stop,
        "um_matched": um_matched,
        "one_hour_signal_delay": delayed,
        "one_day_shifted_clock": shifted,
        "random_side": random_side,
    }
    setup_positions = {
        "primary": primary_setups,
        "cm_only_oi": cm_only_setups,
        "no_taker_gap": no_gap_setups,
        "no_reclaim": no_reclaim_setups,
        "no_oi_stop": no_oi_stop_setups,
        "um_matched": um_matched_setups,
    }
    raw_counts = {
        "primary_active_bars": int(primary_setup.sum()),
        "primary_setup_onsets": int(len(setup_onsets(primary_setup))),
        "primary_accepted_setups": int(len(primary_setups)),
        "primary_confirmed_setups": int(len(primary)),
        "cm_only_accepted_setups": int(len(cm_only_setups)),
        "no_taker_gap_accepted_setups": int(len(no_gap_setups)),
        "um_matched_accepted_setups": int(len(um_matched_setups)),
    }
    if set(events) != set(POLICY_NAMES):
        raise RuntimeError("ICDR policy clock set is incomplete")
    return events, setup_positions, raw_counts


def nonoverlapping_schedule(
    events: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    policy_name: str,
    purge_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    next_free = 0
    dates = frame["date"]
    ordered = events.sort_values(["signal_position", "setup_position"])
    for event in ordered.itertuples(index=False):
        setup = int(event.setup_position)
        confirmation = int(event.confirmation_position)
        signal = int(event.signal_position)
        entry = signal + int(event.delay_bars)
        exit_position = entry + int(event.hold_bars)
        if min(setup, signal, entry) < 0 or exit_position >= len(frame):
            continue
        if entry < next_free:
            continue
        confirmation_date = ""
        if confirmation >= 0:
            if confirmation >= len(frame):
                continue
            confirmation_date = str(dates.iloc[confirmation])
        rows.append(
            {
                "policy_name": policy_name,
                "purge_quantile": purge_quantile,
                "setup_position": setup,
                "confirmation_position": confirmation,
                "signal_position": signal,
                "entry_position": entry,
                "exit_position": exit_position,
                "setup_date": str(dates.iloc[setup]),
                "confirmation_date": confirmation_date,
                "signal_date": str(dates.iloc[signal]),
                "entry_date": str(dates.iloc[entry]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": int(event.side),
                "branch": str(event.branch),
                "delay_bars": int(event.delay_bars),
                "hold_bars": int(event.hold_bars),
            }
        )
        next_free = exit_position
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def build_schedules(
    events: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
    *,
    purge_quantile: float,
) -> dict[str, pd.DataFrame]:
    return {
        name: nonoverlapping_schedule(
            events[name],
            frame,
            policy_name=name,
            purge_quantile=purge_quantile,
        )
        for name in POLICY_NAMES
    }


def _contained_schedule(
    schedule: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if schedule.empty:
        return schedule.copy()
    setup = pd.to_datetime(schedule["setup_date"], errors="raise")
    confirmation_text = schedule["confirmation_date"].fillna("").astype(str)
    confirmation = pd.to_datetime(
        confirmation_text.where(confirmation_text.ne("")), errors="coerce"
    )
    signal = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_date = pd.to_datetime(schedule["exit_date"], errors="raise")
    confirmation_contained = confirmation.isna() | (
        confirmation.ge(start) & confirmation.lt(end)
    )
    mask = (
        setup.ge(start)
        & setup.lt(end)
        & confirmation_contained
        & signal.ge(start)
        & signal.lt(end)
        & entry.ge(start)
        & entry.lt(end)
        & exit_date.ge(start)
        & exit_date.lt(end)
    )
    return schedule.loc[mask].reset_index(drop=True)


def _setup_rate(
    accepted_setups: np.ndarray,
    confirmed_events_frame: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    confirmation_window_bars: int,
) -> dict[str, Any]:
    dates = frame["date"]
    accepted = np.asarray(accepted_setups, dtype=np.int64)
    if accepted.size:
        inside = (dates.iloc[accepted].to_numpy() >= np.datetime64(start)) & (
            accepted + confirmation_window_bars < len(frame)
        )
        accepted_end = np.minimum(accepted + confirmation_window_bars, len(frame) - 1)
        inside &= dates.iloc[accepted_end].to_numpy() < np.datetime64(end)
        accepted = accepted[inside]
    if confirmed_events_frame.empty:
        confirmed = np.asarray([], dtype=np.int64)
    else:
        confirmed = confirmed_events_frame["setup_position"].to_numpy(np.int64)
        confirmed = confirmed[np.isin(confirmed, accepted)]
    denominator = int(len(accepted))
    numerator = int(len(confirmed))
    rate = float(numerator / denominator) if denominator else 0.0
    return {
        "accepted_setups_with_full_confirmation_window": denominator,
        "confirmed_setups": numerator,
        "confirmation_rate": rate,
    }


def _month_concentration(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {"maximum_single_month_share": 1.0, "maximum_month": None}
    month_counts = (
        pd.to_datetime(schedule["entry_date"]).dt.to_period("M").value_counts()
    )
    return {
        "maximum_single_month_share": float(month_counts.max() / len(schedule)),
        "maximum_month": str(month_counts.idxmax()),
    }


def entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_set = set(left["entry_date"].astype(str))
    right_set = set(right["entry_date"].astype(str))
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _window_support(
    schedules: dict[str, pd.DataFrame],
    events: dict[str, pd.DataFrame],
    setup_positions: dict[str, np.ndarray],
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    policy: prereg.Policy,
) -> dict[str, Any]:
    contained = {
        name: _contained_schedule(schedule, start, end)
        for name, schedule in schedules.items()
    }
    primary = contained["primary"]
    jaccard = {
        name: entry_jaccard(primary, contained[name]) for name in JACCARD_CONTROLS
    }
    result = {
        "start": str(start),
        "end_exclusive": str(end),
        "primary_nonoverlap_events": int(len(primary)),
        "control_nonoverlap_events": {
            name: int(len(contained[name]))
            for name in POLICY_NAMES
            if name != "primary"
        },
        "confirmation": _setup_rate(
            setup_positions["primary"],
            events["primary"],
            frame,
            start=start,
            end=end,
            confirmation_window_bars=policy.confirmation_window_bars,
        ),
        **_month_concentration(primary),
        "entry_jaccard": jaccard,
    }
    return result


def _train_gates(support: dict[str, Any], policy: prereg.Policy) -> dict[str, bool]:
    frozen = prereg.build_manifest()["support_calibration"]
    confirmation = support["confirmation"]["confirmation_rate"]
    jaccard = support["entry_jaccard"]
    return {
        "train_events_at_least_100": (
            support["primary_nonoverlap_events"] >= frozen["minimum_nonoverlap_train"]
        ),
        "2021_partial_events_at_least_20": (
            support["subperiod_counts"]["2021_partial"]
            >= frozen["minimum_2021_partial"]
        ),
        "2022_events_at_least_50": (
            support["subperiod_counts"]["2022"] >= frozen["minimum_2022"]
        ),
        "confirmation_rate_between_0_05_and_0_80": (
            frozen["minimum_confirmation_rate"]
            <= confirmation
            <= frozen["maximum_confirmation_rate"]
        ),
        "maximum_month_share_at_most_0_15": (
            support["maximum_single_month_share"]
            <= frozen["maximum_single_month_share"]
        ),
        **{
            f"{name}_entry_jaccard_at_most_{limit}": jaccard[name] <= limit
            for name, limit in frozen["maximum_signal_jaccard"].items()
        },
        "frozen_delay": all(
            schedule["delay_bars"].eq(policy.execution_delay_bars).all()
            for schedule in support["_contained"].values()
        ),
        "frozen_hold": all(
            schedule["hold_bars"].eq(policy.hold_bars).all()
            for schedule in support["_contained"].values()
        ),
    }


def _validation_gates(support: dict[str, Any]) -> dict[str, bool]:
    frozen = prereg.build_manifest()["support_calibration"]
    confirmation = support["confirmation"]["confirmation_rate"]
    jaccard = support["entry_jaccard"]
    return {
        "2023_events_at_least_75": (
            support["primary_nonoverlap_events"] >= frozen["minimum_2023"]
        ),
        "2023_h1_events_at_least_30": (
            support["subperiod_counts"]["2023_h1"] >= frozen["minimum_each_2023_half"]
        ),
        "2023_h2_events_at_least_30": (
            support["subperiod_counts"]["2023_h2"] >= frozen["minimum_each_2023_half"]
        ),
        "confirmation_rate_between_0_05_and_0_80": (
            frozen["minimum_confirmation_rate"]
            <= confirmation
            <= frozen["maximum_confirmation_rate"]
        ),
        "maximum_month_share_at_most_0_15": (
            support["maximum_single_month_share"]
            <= frozen["maximum_single_month_share"]
        ),
        **{
            f"{name}_entry_jaccard_at_most_{limit}": jaccard[name] <= limit
            for name, limit in frozen["maximum_signal_jaccard"].items()
        },
    }


def _strip_internal(support: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in support.items() if not key.startswith("_")}


def calculate_train_support(
    schedules: dict[str, pd.DataFrame],
    events: dict[str, pd.DataFrame],
    setup_positions: dict[str, np.ndarray],
    frame: pd.DataFrame,
    policy: prereg.Policy,
) -> dict[str, Any]:
    support = _window_support(
        schedules,
        events,
        setup_positions,
        frame,
        start=START,
        end=TRAIN_END,
        policy=policy,
    )
    support["_contained"] = {
        name: _contained_schedule(schedule, START, TRAIN_END)
        for name, schedule in schedules.items()
    }
    support["subperiod_counts"] = {
        "2021_partial": int(
            len(
                _contained_schedule(
                    schedules["primary"], START, pd.Timestamp("2022-01-01")
                )
            )
        ),
        "2022": int(
            len(
                _contained_schedule(
                    schedules["primary"], pd.Timestamp("2022-01-01"), TRAIN_END
                )
            )
        ),
    }
    support["gates"] = _train_gates(support, policy)
    support["passes"] = bool(all(support["gates"].values()))
    return support


def calculate_validation_support(
    schedules: dict[str, pd.DataFrame],
    events: dict[str, pd.DataFrame],
    setup_positions: dict[str, np.ndarray],
    frame: pd.DataFrame,
    policy: prereg.Policy,
) -> dict[str, Any]:
    support = _window_support(
        schedules,
        events,
        setup_positions,
        frame,
        start=TRAIN_END,
        end=END_EXCLUSIVE,
        policy=policy,
    )
    support["subperiod_counts"] = {
        "2023_h1": int(
            len(
                _contained_schedule(
                    schedules["primary"], TRAIN_END, pd.Timestamp("2023-07-01")
                )
            )
        ),
        "2023_h2": int(
            len(
                _contained_schedule(
                    schedules["primary"],
                    pd.Timestamp("2023-07-01"),
                    END_EXCLUSIVE,
                )
            )
        ),
    }
    support["gates"] = _validation_gates(support)
    support["passes"] = bool(all(support["gates"].values()))
    return support


def run_support(policy: prereg.Policy) -> dict[str, Any]:
    if policy != prereg.Policy():
        raise ValueError("ICDR-144 policy is frozen; mutation is forbidden")
    frame, source = load_support_frame(policy)
    features = build_features(frame, policy)
    train_mask = frame["date"].lt(TRAIN_END)
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    train_features = features.loc[train_mask].reset_index(drop=True)
    train_candidates: list[dict[str, Any]] = []
    selected_quantile: float | None = None
    for quantile in sorted(policy.purge_quantiles, reverse=True):
        events, setup_positions, raw_counts = classify_events(
            train_features, policy, purge_quantile=quantile
        )
        schedules = build_schedules(events, train_frame, purge_quantile=quantile)
        train_support = calculate_train_support(
            schedules, events, setup_positions, train_frame, policy
        )
        train_candidates.append(
            {
                "purge_quantile": quantile,
                "raw_counts": raw_counts,
                "train_support": _strip_internal(train_support),
            }
        )
        if selected_quantile is None and train_support["passes"]:
            selected_quantile = quantile

    selected_train: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    selected_schedules: dict[str, pd.DataFrame] | None = None
    support_decision = "reject_before_returns_no_train_support_q"
    rejection_reasons: list[str] = []
    if selected_quantile is not None:
        events, setup_positions, _ = classify_events(
            features, policy, purge_quantile=selected_quantile
        )
        selected_schedules = build_schedules(
            events, frame, purge_quantile=selected_quantile
        )
        selected_train = next(
            candidate["train_support"]
            for candidate in train_candidates
            if candidate["purge_quantile"] == selected_quantile
        )
        replayed_train = _strip_internal(
            calculate_train_support(
                selected_schedules, events, setup_positions, frame, policy
            )
        )
        if replayed_train != selected_train:
            raise RuntimeError(
                "ICDR selected train clock changed after 2023 was loaded"
            )
        validation_full = calculate_validation_support(
            selected_schedules, events, setup_positions, frame, policy
        )
        validation = _strip_internal(validation_full)
        if validation_full["passes"]:
            support_decision = "pass_open_stage1_train_only"
        else:
            support_decision = "reject_before_returns_2023_support"
            rejection_reasons = [
                name for name, passed in validation_full["gates"].items() if not passed
            ]
    else:
        for candidate in train_candidates:
            failed = [
                name
                for name, passed in candidate["train_support"]["gates"].items()
                if not passed
            ]
            rejection_reasons.append(
                f"Q={candidate['purge_quantile']}:" + ",".join(failed)
            )

    if selected_schedules is None:
        combined_clock = pd.DataFrame(columns=CLOCK_COLUMNS)
    else:
        combined_clock = pd.concat(
            [selected_schedules[name] for name in POLICY_NAMES], ignore_index=True
        ).sort_values(["policy_name", "entry_position"], ignore_index=True)
    clock_content = _clock_bytes(combined_clock)
    core: dict[str, Any] = {
        "protocol": {
            "version": "inverse_collateral_deleveraging_reclaim_support_v1",
            "outcomes_opened": False,
            "price_return_funding_loaded": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "selection_uses_only_2021_2022_support": True,
            "2023_can_reselect_or_fallback_q": False,
            "sealed_2024_plus_opened": False,
        },
        "policy": prereg.policy_payload(),
        "source": source,
        "feature_contract": {
            "thresholds_strictly_lagged": True,
            "threshold_window_bars": policy.baseline_bars,
            "threshold_minimum_prior_bars": policy.baseline_min_periods,
            "current_plus_following_gap_quarantine_bars": (
                policy.post_gap_quarantine_bars
            ),
            "entry_delay_bars": policy.execution_delay_bars,
            "hold_bars": policy.hold_bars,
            "price_signal_columns": [],
        },
        "threshold_availability": {
            column: int(features[column].notna().sum())
            for column in features.columns
            if column.endswith("q90")
            or column.startswith("purge_q")
            or column.startswith("cm_contraction_q")
            or column.startswith("um_contraction_q")
        },
        "train_selection_candidates_descending_q": train_candidates,
        "selected_purge_quantile": selected_quantile,
        "selected_train_support": selected_train,
        "selected_q_2023_support_validation": validation,
        "support_decision": support_decision,
        "rejection_reasons": rejection_reasons,
        "clock": {
            "path": DEFAULT_CLOCK,
            "sha256": hashlib.sha256(clock_content).hexdigest(),
            "rows": int(len(combined_clock)),
            "columns": list(CLOCK_COLUMNS),
            "counts_by_policy": {
                name: int(combined_clock["policy_name"].eq(name).sum())
                for name in POLICY_NAMES
            },
        },
    }
    return {
        **core,
        "manifest_hash": _canonical_hash(core),
        "_combined_clock": combined_clock,
    }


def _write_once(path: str | Path, content: bytes) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite frozen ICDR artifact: {output}")
        return "verified_existing"
    with output.open("xb") as handle:
        handle.write(content)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    payload = run_support(prereg.Policy())
    schedule = payload.pop("_combined_clock")
    clock_content = _clock_bytes(schedule)
    if hashlib.sha256(clock_content).hexdigest() != payload["clock"]["sha256"]:
        raise RuntimeError("ICDR clock hash changed during serialization")
    payload["clock"]["path"] = args.clock
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = _canonical_hash(core)
    clock_status = _write_once(args.clock, clock_content)
    result_content = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    result_status = _write_once(args.output, result_content)
    print(
        json.dumps(
            {
                "result_status": result_status,
                "clock_status": clock_status,
                "support_decision": payload["support_decision"],
                "selected_purge_quantile": payload["selected_purge_quantile"],
                "selected_train_support": payload["selected_train_support"],
                "selected_q_2023_support_validation": payload[
                    "selected_q_2023_support_validation"
                ],
                "rejection_reasons": payload["rejection_reasons"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
                "clock": args.clock,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
