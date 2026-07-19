"""Build outcome-blind CMSR-36 features, clocks, support, and novelty gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_coinm_calendar_curve_compression as calendar_curve
from training import preregister_coinm_next_maturity_shock_relay as p
from training import preregister_coinm_roll_migration_alpha as old_roll


PREREGISTRATION = Path(p.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "44a1178110509708d7e391eb766fd02e9cd4b32bd1c790f03172e26a48eb98ea"
)
PREREGISTRATION_COMMIT = "09b1249"
DEFAULT_OUTPUT = Path("results/coinm_next_maturity_shock_relay_support_2026-07-19.json")
DEFAULT_CLOCK = Path("data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz")

STATIC_COMPARATORS = {
    "PSR-30/6": {
        "path": "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
        "sha256": "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6",
        "entry_column": "entry_time",
        "filter_column": None,
        "filter_value": None,
    },
    "CCPR-H4": {
        "path": "results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv",
        "sha256": "2a864ec2b616a3118bf9ffa44f99f96fbe19e79d82870f21a0d7d9010d5c993a",
        "entry_column": "entry_time",
        "filter_column": "control",
        "filter_value": "primary",
    },
    "CLBR-24": {
        "path": "data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz",
        "sha256": "df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0",
        "entry_column": "entry_time",
        "filter_column": None,
        "filter_value": None,
    },
    "EBLR-60/30": {
        "path": "data/eth_btc_liquidation_relay_clocks_2023_2024.csv.gz",
        "sha256": "b4b35a0e9ae0cf26bf08df67b5c2fc832393c638c97f5b91a86894ee693b430e",
        "entry_column": "entry_time",
        "filter_column": None,
        "filter_value": None,
    },
    "CIPA-48": {
        "path": "data/cross_collateral_inventory_pressure_absorption_clock_2021_2023.csv.gz",
        "sha256": "a96d06ecda35fd7f0f75a8015ab907e280c4d4b8c06620a9da3d874adb6523f9",
        "entry_column": "entry_time",
        "filter_column": "control",
        "filter_value": "primary",
    },
}


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("CMSR-36 timestamp is NaT")
    return cast(pd.Timestamp, result)


def _timedelta(**kwargs: Any) -> pd.Timedelta:
    result = pd.Timedelta(**kwargs)
    if result is pd.NaT:
        raise ValueError("CMSR-36 timedelta is NaT")
    return cast(pd.Timedelta, result)


TimeWindow = tuple[pd.Timestamp, pd.Timestamp]
FIT: TimeWindow = (_timestamp("2020-08-01"), _timestamp("2023-01-01"))
TEST_SUPPORT: TimeWindow = (_timestamp("2023-01-01"), _timestamp("2024-01-01"))
WINDOWS: dict[str, TimeWindow] = {
    "fit": FIT,
    "fit_2020h2": (_timestamp("2020-08-01"), _timestamp("2021-01-01")),
    "fit_2021h1": (_timestamp("2021-01-01"), _timestamp("2021-07-01")),
    "fit_2021h2": (_timestamp("2021-07-01"), _timestamp("2022-01-01")),
    "fit_2022h1": (_timestamp("2022-01-01"), _timestamp("2022-07-01")),
    "fit_2022h2": (_timestamp("2022-07-01"), _timestamp("2023-01-01")),
    "test_support": TEST_SUPPORT,
    "test_2023h1": (_timestamp("2023-01-01"), _timestamp("2023-07-01")),
    "test_2023h2": (_timestamp("2023-07-01"), _timestamp("2024-01-01")),
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_preregistration() -> dict[str, Any]:
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("CMSR-36 preregistration hash mismatch")
    report = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    p.validate_manifest(report, verify_sources=False)
    source = report["source_contract"]
    if _sha256(source["source"]) != source["source_sha256"]:
        raise ValueError("CMSR-36 source hash mismatch")
    if _sha256(source["source_manifest"]) != source["source_manifest_sha256"]:
        raise ValueError("CMSR-36 source manifest hash mismatch")
    return report


def load_source(manifest: dict[str, Any]) -> pd.DataFrame:
    return old_roll.load_source(manifest["source_contract"]["source"])


def causal_quantile(
    values: pd.Series,
    pair: pd.Series,
    quantile: float,
    *,
    shift: int,
    window: int,
    min_periods: int,
) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    grouped = pd.DataFrame({"value": values, "pair": pair}).groupby(
        "pair", sort=False, dropna=False
    )
    for _, group in grouped:
        output.loc[group.index] = (
            _series(group, "value")
            .shift(shift)
            .rolling(window, min_periods=min_periods)
            .quantile(quantile)
        )
    return output


def build_feature_state(source: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    policy = manifest["policy"]
    path = int(policy["path_bars"])
    edge = int(policy["share_edge_bars"])
    pair = (
        _series(source, "front_symbol").astype(str)
        + "|"
        + _series(source, "next_symbol").astype(str)
    )
    valid = _series(source, "feature_valid").astype(bool)
    path_valid = cast(pd.Series, valid.rolling(path, min_periods=path).sum()).eq(
        path
    ) & pair.eq(pair.shift(path - 1))
    front_volume = _series(source, "front_volume").where(valid)
    next_volume = _series(source, "next_volume").where(valid)
    next_share = next_volume / (front_volume + next_volume).replace(0.0, np.nan)
    front_share = 1.0 - next_share

    def slope(series: pd.Series) -> pd.Series:
        return (
            series.rolling(edge, min_periods=edge).median()
            - series.shift(path - edge).rolling(edge, min_periods=edge).median()
        )

    def flow(leg: str) -> pd.Series:
        volume = _series(source, f"{leg}_volume").where(valid)
        buy = _series(source, f"{leg}_taker_buy_volume").where(valid)
        numerator = cast(
            pd.Series, (2.0 * buy - volume).rolling(path, min_periods=path).sum()
        )
        denominator = cast(
            pd.Series, volume.rolling(path, min_periods=path).sum()
        ).replace(0.0, np.nan)
        return numerator / denominator

    def path_return(leg: str) -> pd.Series:
        return cast(
            pd.Series,
            np.log(
                _series(source, f"{leg}_close")
                / _series(source, f"{leg}_open").shift(path - 1)
            ),
        )

    state = pd.DataFrame(index=source.index)
    state["pair"] = pair
    state["path_valid"] = path_valid
    state["next_share_slope"] = slope(next_share).where(path_valid)
    state["front_share_slope"] = slope(front_share).where(path_valid)
    state["next_flow"] = flow("next").where(path_valid)
    state["front_flow"] = flow("front").where(path_valid)
    state["next_return"] = path_return("next").where(path_valid)
    state["front_return"] = path_return("front").where(path_valid)
    state["next_lead_shock"] = _series(state, "next_return") - _series(
        state, "front_return"
    )
    state["front_lead_shock"] = -_series(state, "next_lead_shock")
    shift = int(policy["prior_nonoverlap_shift_bars"])
    window = int(policy["prior_window_bars"])
    minimum = int(policy["prior_min_periods"])
    threshold_specs = {
        "next_share_slope": float(policy["share_slope_quantile"]),
        "next_flow_abs": float(policy["next_flow_abs_quantile"]),
        "next_lead_shock_abs": float(policy["lead_shock_abs_quantile"]),
        "front_share_slope": float(policy["share_slope_quantile"]),
        "front_flow_abs": float(policy["next_flow_abs_quantile"]),
        "front_lead_shock_abs": float(policy["lead_shock_abs_quantile"]),
    }
    values = {
        "next_share_slope": _series(state, "next_share_slope"),
        "next_flow_abs": _series(state, "next_flow").abs(),
        "next_lead_shock_abs": _series(state, "next_lead_shock").abs(),
        "front_share_slope": _series(state, "front_share_slope"),
        "front_flow_abs": _series(state, "front_flow").abs(),
        "front_lead_shock_abs": _series(state, "front_lead_shock").abs(),
    }
    for name, quantile in threshold_specs.items():
        state[f"threshold_{name}"] = causal_quantile(
            values[name],
            pair,
            quantile,
            shift=shift,
            window=window,
            min_periods=minimum,
        )
    return state


def build_flags(
    source: pd.DataFrame, state: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, pd.Series]:
    policy = manifest["policy"]
    valid = _series(state, "path_valid").astype(bool)
    next_flow = _series(state, "next_flow")
    next_return = _series(state, "next_return")
    front_return = _series(state, "front_return")
    next_lead = _series(state, "next_lead_shock")
    next_sign = cast(pd.Series, np.sign(next_flow))
    next_return_sign = cast(pd.Series, np.sign(next_return))
    next_lead_sign = cast(pd.Series, np.sign(next_lead))
    accepted = (
        next_sign.ne(0.0)
        & next_sign.eq(next_return_sign)
        & front_return.abs().le(
            float(policy["front_to_next_return_abs_max"]) * next_return.abs()
        )
    )
    share_gate = _series(state, "next_share_slope").ge(
        _series(state, "threshold_next_share_slope")
    )
    flow_gate = next_flow.abs().ge(_series(state, "threshold_next_flow_abs"))
    lead_gate = next_lead.abs().ge(
        _series(state, "threshold_next_lead_shock_abs")
    ) & next_sign.eq(next_lead_sign)

    front_flow = _series(state, "front_flow")
    front_ret = _series(state, "front_return")
    next_ret = _series(state, "next_return")
    front_lead = _series(state, "front_lead_shock")
    front_sign = cast(pd.Series, np.sign(front_flow))
    front_return_sign = cast(pd.Series, np.sign(front_ret))
    front_lead_sign = cast(pd.Series, np.sign(front_lead))
    front_mirror = (
        valid
        & _series(state, "front_share_slope").ge(
            _series(state, "threshold_front_share_slope")
        )
        & front_flow.abs().ge(_series(state, "threshold_front_flow_abs"))
        & front_lead.abs().ge(_series(state, "threshold_front_lead_shock_abs"))
        & front_sign.ne(0.0)
        & front_sign.eq(front_return_sign)
        & front_sign.eq(front_lead_sign)
        & next_ret.abs().le(
            float(policy["front_to_next_return_abs_max"]) * front_ret.abs()
        )
    )
    required_hours = (
        float(policy["delivery_buffer_hours"])
        + 5.0 * float(policy["hold_bars"]) / 60.0
        + 5.0 * float(policy["entry_delay_from_signal_bars"]) / 60.0
    )
    delivery_safe = _series(source, "front_hours_to_delivery").ge(
        required_hours
    ) & _series(source, "next_hours_to_delivery").ge(required_hours)
    return {
        "primary": valid
        & delivery_safe
        & share_gate
        & flow_gate
        & accepted
        & lead_gate,
        "no_share_transition": valid & delivery_safe & flow_gate & accepted & lead_gate,
        "no_lead_shock": valid & delivery_safe & share_gate & flow_gate & accepted,
        "front_led_mirror": front_mirror & delivery_safe,
    }


def _onset(flag: pd.Series, pair: pd.Series) -> pd.Series:
    current = flag.fillna(False).astype(bool)
    return current & ~(current.shift(1, fill_value=False) & pair.eq(pair.shift(1)))


def build_clock(
    source: pd.DataFrame, state: pd.DataFrame, manifest: dict[str, Any]
) -> pd.DataFrame:
    flags = build_flags(source, state, manifest)
    pair = _series(state, "pair")
    sides = {
        "primary": cast(pd.Series, np.sign(_series(state, "next_flow"))),
        "no_share_transition": cast(pd.Series, np.sign(_series(state, "next_flow"))),
        "no_lead_shock": cast(pd.Series, np.sign(_series(state, "next_flow"))),
        "front_led_mirror": cast(pd.Series, np.sign(_series(state, "front_flow"))),
    }
    delay = _timedelta(minutes=5 * manifest["policy"]["entry_delay_from_signal_bars"])
    hold = _timedelta(minutes=5 * manifest["policy"]["hold_bars"])
    rows: list[dict[str, Any]] = []
    for control, flag in flags.items():
        next_allowed = FIT[0]
        for position in np.flatnonzero(_onset(flag, pair).to_numpy(dtype=bool)):
            signal = _timestamp(_series(source, "signal_bar_open_utc").iloc[position])
            entry = _timestamp(signal + delay)
            exit_time = _timestamp(entry + hold)
            if entry < next_allowed or exit_time >= TEST_SUPPORT[1]:
                continue
            next_allowed = exit_time
            rows.append(
                {
                    "control": control,
                    "signal_time": signal,
                    "feature_available_time": _timestamp(
                        _series(source, "feature_available_time_utc").iloc[position],
                    ),
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side": int(sides[control].iloc[position]),
                    "pair": str(pair.iloc[position]),
                    "next_share_slope": float(
                        _series(state, "next_share_slope").iloc[position]
                    ),
                    "next_flow": float(_series(state, "next_flow").iloc[position]),
                    "next_return": float(_series(state, "next_return").iloc[position]),
                    "front_return": float(
                        _series(state, "front_return").iloc[position]
                    ),
                    "next_lead_shock": float(
                        _series(state, "next_lead_shock").iloc[position]
                    ),
                }
            )
    clock = (
        pd.DataFrame(rows).sort_values(["control", "entry_time"]).reset_index(drop=True)
    )
    if not bool(_series(clock, "side").isin((-1, 1)).all()):
        raise ValueError("CMSR-36 emitted an invalid side")
    return clock


def _window(frame: pd.DataFrame, bounds: TimeWindow) -> pd.DataFrame:
    start, end = bounds
    return frame.loc[
        _series(frame, "signal_time").ge(start)
        & _series(frame, "entry_time").ge(start)
        & _series(frame, "exit_time").lt(end)
    ].copy()


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    side = _series(frame, "side")
    months = cast(
        pd.Series, _series(frame, "entry_time").dt.to_period("M").value_counts()
    )
    pairs = cast(pd.Series, _series(frame, "pair").value_counts())
    total = len(frame)
    longs, shorts = int(side.eq(1).sum()), int(side.eq(-1).sum())
    return {
        "events": total,
        "longs": longs,
        "shorts": shorts,
        "minimum_side_share": min(longs, shorts) / total if total else 0.0,
        "maximum_month_share": float(months.max() / total) if total else 0.0,
        "maximum_pair_share": float(pairs.max() / total) if total else 0.0,
        "month_counts": {str(k): int(v) for k, v in months.sort_index().items()},
        "pair_counts": {str(k): int(v) for k, v in pairs.sort_index().items()},
    }


def support_summaries(primary: pd.DataFrame) -> dict[str, Any]:
    return {
        name: _summary(_window(primary, bounds)) for name, bounds in WINDOWS.items()
    }


def support_checks(
    summary: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, bool]:
    gate = manifest["support_gate"]
    fit_halves = [
        summary[f"fit_{year}{half}"]["events"]
        for year, half in (
            (2020, "h2"),
            (2021, "h1"),
            (2021, "h2"),
            (2022, "h1"),
            (2022, "h2"),
        )
    ]
    return {
        "fit_events": summary["fit"]["events"] >= gate["minimum_fit_events"],
        "fit_half_years": min(fit_halves) >= gate["minimum_each_fit_half_year"],
        "2023_events": summary["test_support"]["events"] >= gate["minimum_2023_events"],
        "2023_halves": min(
            summary["test_2023h1"]["events"], summary["test_2023h2"]["events"]
        )
        >= gate["minimum_each_2023_half"],
        "fit_side_balance": summary["fit"]["minimum_side_share"]
        >= gate["minimum_each_side_share"],
        "2023_side_balance": summary["test_support"]["minimum_side_share"]
        >= gate["minimum_each_side_share"],
        "fit_month_concentration": summary["fit"]["maximum_month_share"]
        <= gate["fit_max_month_share"],
        "2023_month_concentration": summary["test_support"]["maximum_month_share"]
        <= gate["test_max_month_share"],
        "fit_pair_concentration": summary["fit"]["maximum_pair_share"]
        <= gate["fit_max_pair_share"],
        "2023_pair_concentration": summary["test_support"]["maximum_pair_share"]
        <= gate["test_max_pair_share"],
    }


def _overlap(
    left: pd.Series, right: pd.Series, tolerance: pd.Timedelta
) -> dict[str, Any]:
    left = cast(pd.Series, pd.to_datetime(left, utc=True)).dt.tz_convert(None)
    right = cast(pd.Series, pd.to_datetime(right, utc=True)).dt.tz_convert(None)
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    near = (
        sum(((right - value).abs() <= tolerance).any() for value in left) / len(left)
        if len(left)
        else 0.0
    )
    return {
        "new_events": int(len(left)),
        "comparator_events": int(len(right)),
        "exact_jaccard": float(len(left_set & right_set) / len(union))
        if union
        else 0.0,
        "new_near_fraction": float(near),
        "tolerance_minutes": int(tolerance.total_seconds() // 60),
    }


def _slice_times(times: pd.Series, bounds: TimeWindow) -> pd.Series:
    values = cast(pd.Series, pd.to_datetime(times, utc=True)).dt.tz_convert(None)
    return values.loc[values.ge(bounds[0]) & values.lt(bounds[1])]


def old_roll_novelty(source: pd.DataFrame, primary: pd.DataFrame) -> dict[str, Any]:
    state = old_roll.build_signal_state(source)
    active, side = old_roll.candidate_clock(source, state, old_roll.CANDIDATES[0])
    schedule = old_roll.nonoverlapping_schedule(
        source,
        active,
        side,
        old_roll.CANDIDATES[0],
        start=FIT[0],
        end=TEST_SUPPORT[1],
    )
    old_signal = pd.to_datetime(_series(schedule, "signal_bar_open"))
    output: dict[str, Any] = {}
    for name, bounds in (("fit", FIT), ("test_support", TEST_SUPPORT)):
        new = _series(_window(primary, bounds), "signal_time")
        old = _slice_times(old_signal, bounds)
        output[name] = _overlap(new, old, _timedelta(minutes=10))
    return output


def calendar_curve_entries(source_path: str) -> pd.Series:
    if (
        _sha256("training/preregister_coinm_calendar_curve_compression.py")
        != "a3f6ce9991c2b9a63c0c8a79c70bf6bf0005d1972afff760bb3317e2bf4d135d"
    ):
        raise ValueError("CMSR calendar-curve comparator code changed")
    source = calendar_curve.load_source(source_path)
    state = calendar_curve.build_signal_state(source)
    active, side = calendar_curve.candidate_clock(source, state)
    schedule = calendar_curve.nonoverlapping_schedule(
        source, state, active, side, start=FIT[0], end=TEST_SUPPORT[1]
    )
    return pd.to_datetime(_series(schedule, "entry_time"))


def other_novelty(
    primary: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    comparators: dict[str, pd.Series] = {
        "COINM-CALENDAR": calendar_curve_entries(manifest["source_contract"]["source"])
    }
    for name, spec in STATIC_COMPARATORS.items():
        path = str(spec["path"])
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"CMSR comparator hash mismatch: {name}")
        frame = pd.read_csv(path)
        if spec["filter_column"] is not None:
            frame = frame.loc[
                _series(frame, str(spec["filter_column"])).eq(spec["filter_value"])
            ]
        comparators[name] = pd.to_datetime(
            _series(frame, str(spec["entry_column"])), utc=True, errors="raise"
        ).dt.tz_convert(None)
    output: dict[str, dict[str, Any]] = {}
    for name, times in comparators.items():
        output[name] = {}
        for split, bounds in (("fit", FIT), ("test_support", TEST_SUPPORT)):
            new = _series(_window(primary, bounds), "entry_time")
            shared = _slice_times(times, bounds)
            output[name][split] = _overlap(new, shared, _timedelta(hours=6))
    return output


def novelty_checks(
    old: dict[str, Any], other: dict[str, dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, bool]:
    gate = manifest["support_gate"]
    checks: dict[str, bool] = {}
    for split, result in old.items():
        checks[f"old_roll_exact_{split}"] = (
            result["exact_jaccard"] <= gate["old_roll_exact_signal_jaccard_max"]
        )
        checks[f"old_roll_near_{split}"] = (
            result["new_near_fraction"] <= gate["old_roll_near_10m_containment_max"]
        )
    for name, splits in other.items():
        for split, result in splits.items():
            if result["comparator_events"] == 0:
                continue
            checks[f"{name}_exact_{split}"] = (
                result["exact_jaccard"] <= gate["other_clock_exact_entry_jaccard_max"]
            )
            checks[f"{name}_near_{split}"] = (
                result["new_near_fraction"]
                <= gate["other_clock_near_6h_containment_max"]
            )
    return checks


def _write_clock(frame: pd.DataFrame, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serial = frame.copy()
    for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        serial[column] = _series(serial, column).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serial.to_csv(
        output,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    return _sha256(output)


def build(
    output_path: str | Path = DEFAULT_OUTPUT,
    clock_path: str | Path = DEFAULT_CLOCK,
) -> dict[str, Any]:
    manifest = load_preregistration()
    source = load_source(manifest)
    state = build_feature_state(source, manifest)
    clock = build_clock(source, state, manifest)
    primary = clock.loc[_series(clock, "control").eq("primary")].copy()
    support = support_summaries(primary)
    support_gate_results = support_checks(support, manifest)
    old_novelty = old_roll_novelty(source, primary)
    independent_novelty = other_novelty(primary, manifest)
    novelty_gate_results = novelty_checks(old_novelty, independent_novelty, manifest)
    all_checks = {**support_gate_results, **novelty_gate_results}
    clock_sha = _write_clock(clock, clock_path)
    core: dict[str, Any] = {
        "protocol_version": "coinm_next_maturity_shock_relay_support_v1",
        "policy_id": "CMSR-36",
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREGISTRATION_COMMIT,
            "manifest_hash": manifest["manifest_hash"],
        },
        "source": {
            "path": manifest["source_contract"]["source"],
            "sha256": manifest["source_contract"]["source_sha256"],
            "rows": int(len(source)),
            "valid_rows": int(_series(source, "feature_valid").sum()),
            "execution_btcusdt_rows_loaded": 0,
            "funding_rows_loaded": 0,
        },
        "feature_support": {
            "valid_paths": int(_series(state, "path_valid").sum()),
            "first_valid_path": str(
                _series(source, "signal_bar_open_utc")
                .loc[_series(state, "path_valid")]
                .min()
            ),
        },
        "clock": {"path": str(clock_path), "sha256": clock_sha, "rows": len(clock)},
        "support": support,
        "support_checks": support_gate_results,
        "old_roll_novelty": old_novelty,
        "independent_clock_novelty": independent_novelty,
        "novelty_checks": novelty_gate_results,
        "all_checks": all_checks,
        "support_passed": bool(all(all_checks.values())),
        "failed_checks": [name for name, passed in all_checks.items() if not passed],
        "advance_to_train_outcomes": bool(all(all_checks.values())),
        "sealed_outcome_windows": ["train_2020_2022", "test_2023", "2024_plus"],
    }
    report = {**core, "manifest_hash": _canonical_hash(core)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    report = build(args.output, args.clock)
    print(
        json.dumps(
            {
                "support_passed": report["support_passed"],
                "failed": report["failed_checks"],
            }
        )
    )


if __name__ == "__main__":
    main()
