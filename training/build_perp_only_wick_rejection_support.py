"""Build the outcome-blind POWR-12 signal and control clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_perp_only_wick_rejection as prereg


PREREGISTRATION_COMMIT = "e168d6ff560d429138f586e625ba54e1c9710c97"
PREREGISTRATION_SOURCE_SHA256 = (
    "e267285e9e342f5fad8c966ae79f6988ac0f2ba619919038d840c48dc903a5a4"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/perp-only-wick-rejection-preregistration-2026-07-17.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "549d30103e0a7ddfecab07c523b49a929d505f08a38ba211d423846dcc51b87c"
)
PREREGISTRATION_RESULT = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_RESULT_SHA256 = (
    "1e373a75936d1bc68c488d80c9562e54d132c4274a803a8ca5e76f7675892719"
)
DEFAULT_OUTPUT = "results/perp_only_wick_rejection_support_2026-07-17.json"
DEFAULT_CLOCK = "results/perp_only_wick_rejection_clock_2026-07-17.csv"
SELECTION_END = pd.Timestamp("2024-01-01")
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "spot_only_wick",
    "common_wick",
    "basis_free_perp_wick",
    "one_bar_delayed_entry",
    "stale_spot_1h",
    "stale_spot_1d",
)
CLOCK_COLUMNS = (
    "signal_position",
    "entry_position",
    "exit_position",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "branch",
    "entry_delay_bars",
    "hold_bars",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (Path(prereg.__file__), PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen POWR-12 preregistration changed: {path}")
    payload = json.loads(PREREGISTRATION_RESULT.read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise ValueError("POWR-12 outcomes opened before support freeze")
    if payload["policy"] != asdict(prereg.Policy()):
        raise ValueError("POWR-12 support policy differs from preregistration")
    return payload


def _validate_minute_source(frame: pd.DataFrame, *, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} minute source is empty")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{label} minute timestamps are invalid")
    if not frame["date"].dt.second.eq(0).all() or not frame["date"].dt.microsecond.eq(0).all():
        raise ValueError(f"{label} timestamps are not minute aligned")
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{label} contains invalid OHLC")
    opens, highs, lows, closes = values.T
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError(f"{label} violates OHLC invariants")


def aggregate_five_minute(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    _validate_minute_source(frame, label=prefix.rstrip("_"))
    indexed = frame.set_index("date")
    grouped = indexed.resample("5min", label="left", closed="left")
    output = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_count=("open", "count"),
    ).reset_index()
    output["complete"] = output["minute_count"].eq(5)
    rename = {
        column: f"{prefix}{column}"
        for column in ("open", "high", "low", "close", "minute_count", "complete")
    }
    return output.rename(columns=rename)


def load_support_frame(
    policy: prereg.Policy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = verify_preregistration()
    source = payload["source_contract"]
    for path, expected in (
        (source["source_manifest"], source["source_manifest_sha256"]),
        (source["perp"], source["perp_sha256"]),
        (source["spot"], source["spot_sha256"]),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"POWR-12 source hash changed: {path}")
    manifest = json.loads(Path(source["source_manifest"]).read_text())
    if manifest.get("outputs", {}).get("perp", {}).get("sha256") != source["perp_sha256"]:
        raise ValueError("POWR-12 perp manifest hash differs")
    if manifest.get("outputs", {}).get("spot", {}).get("sha256") != source["spot_sha256"]:
        raise ValueError("POWR-12 spot manifest hash differs")
    if manifest.get("database_snapshot_is_point_in_time") is not False:
        raise ValueError("POWR-12 source provenance boundary changed")

    usecols = ["date", "open", "high", "low", "close"]
    perp_1m = pd.read_csv(
        source["perp"], compression="gzip", usecols=usecols, parse_dates=["date"]
    )
    spot_1m = pd.read_csv(
        source["spot"], compression="gzip", usecols=usecols, parse_dates=["date"]
    )
    if len(perp_1m) != source["perp_rows"] or len(spot_1m) != source["spot_rows"]:
        raise ValueError("POWR-12 source row count differs from freeze")
    if perp_1m["date"].max() >= SELECTION_END or spot_1m["date"].max() >= SELECTION_END:
        raise ValueError("POWR-12 support source contains 2024+ rows")
    perp = aggregate_five_minute(perp_1m, prefix="perp_")
    spot = aggregate_five_minute(spot_1m, prefix="spot_")
    frame = perp.merge(spot, on="date", how="outer", validate="one_to_one").sort_values(
        "date"
    ).reset_index(drop=True)
    expected = pd.date_range("2020-01-01", "2023-12-31 23:55", freq="5min")
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("POWR-12 five-minute grid is incomplete")
    frame["joint_complete"] = frame["perp_complete"].fillna(False) & frame[
        "spot_complete"
    ].fillna(False)
    metadata = {
        "source_manifest_sha256": _sha256(source["source_manifest"]),
        "perp_sha256": _sha256(source["perp"]),
        "spot_sha256": _sha256(source["spot"]),
        "database_snapshot_is_point_in_time": False,
        "perp_1m_rows": int(len(perp_1m)),
        "spot_1m_rows": int(len(spot_1m)),
        "five_minute_rows": int(len(frame)),
        "perp_incomplete_five_minute_bars": int((~frame["perp_complete"]).sum()),
        "spot_incomplete_five_minute_bars": int((~frame["spot_complete"]).sum()),
        "joint_complete_five_minute_bars": int(frame["joint_complete"].sum()),
        "signal_ohlc_columns_loaded": [
            "date",
            "perp_open",
            "perp_high",
            "perp_low",
            "perp_close",
            "spot_open",
            "spot_high",
            "spot_low",
            "spot_close",
        ],
        "post_signal_outcomes_loaded": False,
        "future_trade_returns_computed": False,
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
    }
    return frame, metadata


def prior_clean_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = clean.astype(bool) & numeric.notna() & np.isfinite(numeric)
    observed = numeric.loc[valid]
    thresholds = (
        observed.shift(1)
        .rolling(window, min_periods=min_periods)
        .quantile(quantile)
    )
    output = pd.Series(np.nan, index=values.index, dtype=float)
    output.loc[thresholds.index] = thresholds.to_numpy(float)
    return output


def _wick_features(frame: pd.DataFrame) -> dict[str, pd.Series]:
    values: dict[str, pd.Series] = {}
    for venue in ("perp", "spot"):
        open_ = pd.to_numeric(frame[f"{venue}_open"], errors="coerce")
        high = pd.to_numeric(frame[f"{venue}_high"], errors="coerce")
        low = pd.to_numeric(frame[f"{venue}_low"], errors="coerce")
        close = pd.to_numeric(frame[f"{venue}_close"], errors="coerce")
        values[f"{venue}_upper"] = np.log(high / pd.concat([open_, close], axis=1).max(axis=1))
        values[f"{venue}_lower"] = np.log(pd.concat([open_, close], axis=1).min(axis=1) / low)
        values[f"{venue}_body"] = np.log(close / open_)
    values["upper_excess"] = (values["perp_upper"] - values["spot_upper"]).clip(lower=0.0)
    values["lower_excess"] = (values["perp_lower"] - values["spot_lower"]).clip(lower=0.0)
    return values


def _branch_signal(
    upper: pd.Series,
    lower: pd.Series,
    *,
    upper_side: int = -1,
    lower_side: int = 1,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    both = upper & lower
    upper_only = upper & ~both
    lower_only = lower & ~both
    side = pd.Series(0, index=upper.index, dtype=np.int8)
    side.loc[upper_only] = upper_side
    side.loc[lower_only] = lower_side
    branch = pd.Series("none", index=upper.index, dtype="string")
    branch.loc[upper_only] = "upper_rejection"
    branch.loc[lower_only] = "lower_rejection"
    return side, branch, both


def classify_signals(
    frame: pd.DataFrame,
    policy: prereg.Policy,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    clean = frame["joint_complete"].astype(bool)
    wick = _wick_features(frame)
    floor = policy.minimum_perp_wick_bp / 10_000.0
    q = policy.wick_excess_quantile

    def prior(values: pd.Series, valid: pd.Series = clean) -> pd.Series:
        return prior_clean_quantile(
            values,
            valid,
            quantile=q,
            window=policy.baseline_bars,
            min_periods=policy.baseline_min_periods,
        )

    upper_excess_q = prior(wick["upper_excess"])
    lower_excess_q = prior(wick["lower_excess"])
    primary_upper = (
        clean
        & wick["upper_excess"].ge(upper_excess_q)
        & wick["perp_upper"].ge(floor)
        & wick["spot_upper"].le(policy.maximum_spot_to_perp_wick_ratio * wick["perp_upper"])
        & wick["perp_body"].le(0.0)
    )
    primary_lower = (
        clean
        & wick["lower_excess"].ge(lower_excess_q)
        & wick["perp_lower"].ge(floor)
        & wick["spot_lower"].le(policy.maximum_spot_to_perp_wick_ratio * wick["perp_lower"])
        & wick["perp_body"].ge(0.0)
    )
    primary_side, primary_branch, primary_both = _branch_signal(
        primary_upper, primary_lower
    )

    perp_upper_q = prior(wick["perp_upper"])
    perp_lower_q = prior(wick["perp_lower"])
    spot_upper_q = prior(wick["spot_upper"])
    spot_lower_q = prior(wick["spot_lower"])
    spot_upper = (
        clean
        & wick["spot_upper"].ge(spot_upper_q)
        & wick["spot_upper"].ge(floor)
        & wick["spot_body"].le(0.0)
    )
    spot_lower = (
        clean
        & wick["spot_lower"].ge(spot_lower_q)
        & wick["spot_lower"].ge(floor)
        & wick["spot_body"].ge(0.0)
    )
    spot_side, spot_branch, _ = _branch_signal(spot_upper, spot_lower)
    common_upper = (
        clean
        & wick["perp_upper"].ge(perp_upper_q)
        & wick["perp_upper"].ge(floor)
        & wick["spot_upper"].ge(policy.maximum_spot_to_perp_wick_ratio * wick["perp_upper"])
        & wick["perp_body"].le(0.0)
    )
    common_lower = (
        clean
        & wick["perp_lower"].ge(perp_lower_q)
        & wick["perp_lower"].ge(floor)
        & wick["spot_lower"].ge(policy.maximum_spot_to_perp_wick_ratio * wick["perp_lower"])
        & wick["perp_body"].ge(0.0)
    )
    common_side, common_branch, _ = _branch_signal(common_upper, common_lower)
    basis_upper = (
        clean
        & wick["perp_upper"].ge(perp_upper_q)
        & wick["perp_upper"].ge(floor)
        & wick["perp_body"].le(0.0)
    )
    basis_lower = (
        clean
        & wick["perp_lower"].ge(perp_lower_q)
        & wick["perp_lower"].ge(floor)
        & wick["perp_body"].ge(0.0)
    )
    basis_side, basis_branch, _ = _branch_signal(basis_upper, basis_lower)

    stale_signals: dict[str, tuple[pd.Series, pd.Series]] = {}
    for name, lag in (("stale_spot_1h", 12), ("stale_spot_1d", 288)):
        stale_clean = clean & clean.shift(lag, fill_value=False)
        stale_upper = wick["spot_upper"].shift(lag)
        stale_lower = wick["spot_lower"].shift(lag)
        upper_excess = (wick["perp_upper"] - stale_upper).clip(lower=0.0)
        lower_excess = (wick["perp_lower"] - stale_lower).clip(lower=0.0)
        upper = (
            stale_clean
            & upper_excess.ge(prior(upper_excess, stale_clean))
            & wick["perp_upper"].ge(floor)
            & stale_upper.le(policy.maximum_spot_to_perp_wick_ratio * wick["perp_upper"])
            & wick["perp_body"].le(0.0)
        )
        lower = (
            stale_clean
            & lower_excess.ge(prior(lower_excess, stale_clean))
            & wick["perp_lower"].ge(floor)
            & stale_lower.le(policy.maximum_spot_to_perp_wick_ratio * wick["perp_lower"])
            & wick["perp_body"].ge(0.0)
        )
        stale_side, stale_branch, _ = _branch_signal(upper, lower)
        stale_signals[name] = (stale_side, stale_branch)

    def frame_for(
        side: pd.Series,
        branch: pd.Series,
        *,
        entry_delay: int,
        branch_prefix: str,
    ) -> pd.DataFrame:
        active = side.ne(0)
        return pd.DataFrame(
            {
                "date": frame["date"],
                "side": side.astype(np.int8),
                "branch": np.where(active, branch_prefix + ":" + branch.astype(str), "none"),
                "entry_delay_bars": np.where(active, entry_delay, 0).astype(np.int16),
                "hold_bars": np.where(active, policy.hold_bars, 0).astype(np.int16),
            }
        )

    signals = {
        "primary": frame_for(
            primary_side,
            primary_branch,
            entry_delay=policy.entry_delay_bars,
            branch_prefix="primary",
        ),
        "direction_flip": frame_for(
            -primary_side,
            primary_branch,
            entry_delay=policy.entry_delay_bars,
            branch_prefix="direction_flip",
        ),
        "spot_only_wick": frame_for(
            spot_side,
            spot_branch,
            entry_delay=policy.entry_delay_bars,
            branch_prefix="spot_only",
        ),
        "common_wick": frame_for(
            common_side,
            common_branch,
            entry_delay=policy.entry_delay_bars,
            branch_prefix="common",
        ),
        "basis_free_perp_wick": frame_for(
            basis_side,
            basis_branch,
            entry_delay=policy.entry_delay_bars,
            branch_prefix="basis_free",
        ),
        "one_bar_delayed_entry": frame_for(
            primary_side,
            primary_branch,
            entry_delay=policy.entry_delay_bars + 1,
            branch_prefix="delayed",
        ),
        **{
            name: frame_for(
                side,
                branch,
                entry_delay=policy.entry_delay_bars,
                branch_prefix=name,
            )
            for name, (side, branch) in stale_signals.items()
        },
    }
    diagnostics = {
        **wick,
        "upper_excess_threshold": upper_excess_q,
        "lower_excess_threshold": lower_excess_q,
        "primary_upper": primary_upper & ~primary_both,
        "primary_lower": primary_lower & ~primary_both,
        "primary_both_rejected": primary_both,
    }
    return signals, diagnostics


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = SELECTION_END,
) -> pd.DataFrame:
    start_timestamp = frame["date"].min() if start is None else pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("POWR-12 schedule start must precede end")
    dates = frame["date"]
    period = dates.ge(start_timestamp).to_numpy(bool) & dates.lt(end_timestamp).to_numpy(bool)
    joint = frame["joint_complete"].to_numpy(bool)
    sides = signal["side"].to_numpy(np.int8)
    delays = signal["entry_delay_bars"].to_numpy(np.int64)
    holds = signal["hold_bars"].to_numpy(np.int64)
    branches = signal["branch"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    previous_exit = -1
    for signal_position in np.flatnonzero(sides):
        entry_position = int(signal_position + int(delays[signal_position]))
        exit_position = int(entry_position + int(holds[signal_position]))
        if signal_position + 1 >= len(frame) or exit_position >= len(frame):
            continue
        if entry_position < previous_exit:
            continue
        if not (
            period[signal_position]
            and period[entry_position]
            and period[exit_position]
        ):
            continue
        # Signal and the fully observed latency bucket are causal health gates.
        # Spot availability during the future held path is deliberately ignored.
        if not joint[signal_position] or not joint[signal_position + 1]:
            continue
        rows.append(
            {
                "signal_position": int(signal_position),
                "entry_position": entry_position,
                "exit_position": exit_position,
                "signal_date": str(dates.iloc[signal_position]),
                "entry_date": str(dates.iloc[entry_position]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": int(sides[signal_position]),
                "branch": str(branches[signal_position]),
                "entry_delay_bars": int(delays[signal_position]),
                "hold_bars": int(holds[signal_position]),
            }
        )
        previous_exit = exit_position
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _support(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {"nonoverlap_total": 0, "passes_support": False}
    dates = pd.to_datetime(schedule["entry_date"], errors="raise")
    by_year = {str(year): int(dates.dt.year.eq(year).sum()) for year in range(2020, 2024)}
    train = int(dates.lt("2023-01-01").sum())
    selection = int(dates.ge("2023-01-01").sum())
    h1 = int(((dates >= "2023-01-01") & (dates < "2023-07-01")).sum())
    h2 = int(((dates >= "2023-07-01") & (dates < "2024-01-01")).sum())
    total = int(len(schedule))
    long_share = float(schedule["side"].eq(1).mean())
    short_share = float(schedule["side"].eq(-1).mean())
    upper_share = float(schedule["branch"].str.endswith("upper_rejection").mean())
    lower_share = float(schedule["branch"].str.endswith("lower_rejection").mean())
    month_share = float(dates.dt.to_period("M").value_counts().max() / total)
    passes = (
        train >= 500
        and all(by_year[str(year)] >= 80 for year in (2020, 2021, 2022))
        and selection >= 40
        and h1 >= 20
        and h2 >= 10
        and 0.35 <= long_share <= 0.65
        and 0.35 <= short_share <= 0.65
        and upper_share >= 0.20
        and lower_share >= 0.20
        and month_share <= 0.12
    )
    return {
        "nonoverlap_total": total,
        "train_2020_2022": train,
        "selection_2023": selection,
        "by_year": by_year,
        "2023_h1": h1,
        "2023_h2": h2,
        "long_share": long_share,
        "short_share": short_share,
        "upper_branch_share": upper_share,
        "lower_branch_share": lower_share,
        "maximum_single_month_share": month_share,
        "passes_support": bool(passes),
    }


def _entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    a = set(left["entry_position"].astype(int))
    b = set(right["entry_position"].astype(int))
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def run_support() -> tuple[dict[str, Any], pd.DataFrame]:
    policy = prereg.Policy()
    frame, source = load_support_frame(policy)
    signals, diagnostics = classify_signals(frame, policy)
    schedules = {
        name: nonoverlapping_schedule(signals[name], frame) for name in POLICY_NAMES
    }
    primary = schedules["primary"]
    metrics = _support(primary)
    result = {
        "protocol": {
            "name": "POWR-12 — Perp-Only Wick Rejection",
            "support_only": True,
            "outcomes_opened": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "selection_end_exclusive": str(SELECTION_END),
            "post_signal_returns_computed": False,
            "entry": "signal_position+3 after completed latency bucket",
            "future_hold_spot_missingness_filters_entries": False,
        },
        "policy": asdict(policy),
        "source": source,
        "feature_support": {
            "bars_with_both_thresholds": int(
                (
                    diagnostics["upper_excess_threshold"].notna()
                    & diagnostics["lower_excess_threshold"].notna()
                ).sum()
            ),
            "raw_upper": int(diagnostics["primary_upper"].sum()),
            "raw_lower": int(diagnostics["primary_lower"].sum()),
            "raw_both_rejected": int(diagnostics["primary_both_rejected"].sum()),
            "raw_primary": int(signals["primary"]["side"].ne(0).sum()),
        },
        "support": metrics,
        "controls": {
            name: {
                "raw_signal_count": int(signals[name]["side"].ne(0).sum()),
                "nonoverlap_count": int(len(schedule)),
                "entry_jaccard_with_primary": _entry_jaccard(primary, schedule),
            }
            for name, schedule in schedules.items()
            if name != "primary"
        },
        "support_decision": "pass" if metrics["passes_support"] else "reject_before_returns",
        "clock": {
            "path": DEFAULT_CLOCK,
            "columns": list(CLOCK_COLUMNS),
            "rows": int(len(primary)),
        },
    }
    return result, primary


def _write_once(path: str | Path, content: bytes) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite frozen POWR-12 artifact: {output}")
        return "verified_existing"
    with output.open("xb") as handle:
        handle.write(content)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    if args.clock != DEFAULT_CLOCK:
        raise ValueError("POWR-12 clock path is frozen")
    result, clock = run_support()
    clock_bytes = clock.to_csv(index=False, lineterminator="\n").encode("utf-8")
    result["clock"]["sha256"] = hashlib.sha256(clock_bytes).hexdigest()
    result_bytes = (
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    clock_status = _write_once(args.clock, clock_bytes)
    result_status = _write_once(args.output, result_bytes)
    print(
        json.dumps(
            {
                "clock_status": clock_status,
                "result_status": result_status,
                "outcomes_opened": False,
                "support_decision": result["support_decision"],
                "support": result["support"],
                "clock_sha256": result["clock"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
