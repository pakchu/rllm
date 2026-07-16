"""Build the outcome-blind LVRT-R0 event clock and support diagnostics.

The loader intentionally reads only timestamps plus the six preregistered
aggTrade features.  It verifies the kline archive hash, but reads no open,
high, low, close, volume, future return, CAGR, or drawdown value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_liquidity_vacuum_replenishment as prereg


PREREGISTRATION_COMMIT = "625f8df32d1ea4a227f1e0322d82584c48eba290"
PREREGISTRATION_SOURCE_SHA256 = (
    "ba5b9996439a40dedfaa2402e62ec84129aa0aa5e0c6c5881398364b0d31719e"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "a13c91246deee802ecb524b89cde9375190fdb6871a89a4f5d21d28274953329"
)
PREREGISTRATION_RESULT_SHA256 = (
    "967ad4493652afdac5be0d37ed41675bf8803ae14391723b2ef7691ce2ba2582"
)
FEATURE_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
AUDIT_SHA256 = (
    "5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/liquidity-vacuum-replenishment-r0-preregistration-2026-07-17.md"
)
PREREGISTRATION_RESULT = Path(prereg.DEFAULT_OUTPUT)
DEFAULT_OUTPUT = "results/liquidity_vacuum_replenishment_support_2026-07-17.json"
DEFAULT_CLOCK = "results/liquidity_vacuum_replenishment_clock_2026-07-17.csv"
SELECTION_END = pd.Timestamp("2024-01-01")
FEATURE_COLUMNS = (
    "agg_trade_count",
    "signed_quote_notional",
    "signed_price_response",
    "interarrival_burstiness",
    "event_notional_hhi",
    "micro_log_return",
)
CLOCK_COLUMNS = (
    "setup_position",
    "signal_position",
    "entry_position",
    "exit_position",
    "setup_date",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "branch",
    "hold_bars",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    checks = (
        (Path(prereg.__file__), PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
    )
    for path, expected in checks:
        if _sha256(path) != expected:
            raise ValueError(f"frozen LVRT-R0 preregistration changed: {path}")
    payload = json.loads(PREREGISTRATION_RESULT.read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise ValueError("LVRT-R0 outcomes were opened before support freeze")
    if payload["policy"] != asdict(prereg.Policy()):
        raise ValueError("LVRT-R0 support policy differs from preregistration")
    return payload


def _assert_complete_grid(dates: pd.Series) -> None:
    if dates.empty:
        raise ValueError("market timestamp grid is empty")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("market timestamps must be unique and monotonic")
    deltas = dates.diff().dropna()
    if len(deltas) and not deltas.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("market timestamps must form a complete five-minute grid")


def quarantine_mask(
    source_available: pd.Series,
    source_gap_day: pd.Series,
    post_gap_bars: int,
) -> pd.Series:
    if post_gap_bars < 0:
        raise ValueError("post-gap quarantine cannot be negative")
    invalid = (~source_available.astype(bool)) | source_gap_day.astype(bool)
    return (
        invalid.astype(np.int8)
        .rolling(post_gap_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )


def load_support_frame(
    policy: prereg.Policy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preregistration = verify_preregistration()
    source = preregistration["source_contract"]
    frozen_hashes = (
        (source["features"], source["feature_sha256"]),
        (source["feature_manifest"], FEATURE_MANIFEST_SHA256),
        (source["market"], source["market_sha256"]),
        (source["market_manifest"], MARKET_MANIFEST_SHA256),
        (source["source_audit"], AUDIT_SHA256),
    )
    for path, expected in frozen_hashes:
        if _sha256(path) != expected:
            raise ValueError(f"LVRT-R0 source hash changed: {path}")

    feature_manifest = json.loads(Path(source["feature_manifest"]).read_text())
    market_manifest = json.loads(Path(source["market_manifest"]).read_text())
    audit = json.loads(Path(source["source_audit"]).read_text())
    if feature_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("aggTrade manifest opened outcomes")
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("market manifest opened outcomes")
    if audit.get("passed") is not True or audit.get("failed_checks"):
        raise ValueError("aggTrade source-integrity audit did not pass")
    if feature_manifest.get("combined_sha256") != source["feature_sha256"]:
        raise ValueError("feature manifest combined hash differs from preregistration")
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("market manifest combined hash differs from preregistration")

    # Do not parse any market outcome column in this stage.
    market = pd.read_csv(
        source["market"], compression="gzip", usecols=["date"], parse_dates=["date"]
    )
    features = pd.read_csv(
        source["features"],
        compression="gzip",
        usecols=["date", *FEATURE_COLUMNS],
        parse_dates=["date"],
    )
    _assert_complete_grid(market["date"])
    if features["date"].duplicated().any() or not features["date"].is_monotonic_increasing:
        raise ValueError("aggTrade timestamps must be unique and monotonic")
    if market["date"].max() >= SELECTION_END or features["date"].max() >= SELECTION_END:
        raise ValueError("support source contains 2024+ rows")
    frame = market.merge(features, on="date", how="left", validate="one_to_one")
    frame["source_available"] = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
    gap_days = set(audit["manifest_diagnostics"]["source_gap_days"])
    frame["source_gap_day"] = frame["date"].dt.strftime("%Y-%m-%d").isin(gap_days)
    frame["quarantined"] = quarantine_mask(
        frame["source_available"],
        frame["source_gap_day"],
        policy.post_gap_quarantine_bars,
    )
    metadata = {
        "feature_sha256": _sha256(source["features"]),
        "feature_manifest_sha256": _sha256(source["feature_manifest"]),
        "market_sha256": _sha256(source["market"]),
        "market_manifest_sha256": _sha256(source["market_manifest"]),
        "audit_sha256": _sha256(source["source_audit"]),
        "audit_passed": True,
        "source_gap_days": sorted(gap_days),
        "market_columns_loaded": ["date"],
        "feature_columns_loaded": ["date", *FEATURE_COLUMNS],
        "price_or_outcome_columns_loaded": [],
        "market_rows": int(len(market)),
        "feature_rows": int(len(features)),
        "missing_feature_bars": int((~frame["source_available"]).sum()),
        "quarantined_bars": int(frame["quarantined"].sum()),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
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
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 1 <= min_periods <= window:
        raise ValueError("rolling window/minimum are invalid")
    numeric = pd.to_numeric(values, errors="coerce")
    valid = clean.astype(bool) & numeric.notna() & np.isfinite(numeric)
    observed = numeric.loc[valid]
    observed_threshold = (
        observed.shift(1)
        .rolling(window, min_periods=min_periods)
        .quantile(quantile)
    )
    output = pd.Series(np.nan, index=values.index, dtype=float)
    output.loc[observed_threshold.index] = observed_threshold.to_numpy(float)
    return output


def classify_signals(
    frame: pd.DataFrame,
    policy: prereg.Policy,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    clean = ~frame["quarantined"].astype(bool)
    count = pd.to_numeric(frame["agg_trade_count"], errors="coerce")
    flow = pd.to_numeric(frame["signed_quote_notional"], errors="coerce")
    response = pd.to_numeric(frame["signed_price_response"], errors="coerce")
    burst = pd.to_numeric(frame["interarrival_burstiness"], errors="coerce")
    hhi = pd.to_numeric(frame["event_notional_hhi"], errors="coerce")
    micro_return = pd.to_numeric(frame["micro_log_return"], errors="coerce")
    direction = pd.Series(np.sign(flow), index=frame.index, dtype=float)
    active = clean & count.ge(policy.minimum_agg_trade_count)
    burst_threshold = prior_clean_quantile(
        burst,
        clean,
        quantile=policy.setup_quantile,
        window=policy.baseline_bars,
        min_periods=policy.baseline_min_periods,
    )
    hhi_threshold = prior_clean_quantile(
        hhi,
        clean,
        quantile=policy.setup_quantile,
        window=policy.baseline_bars,
        min_periods=policy.baseline_min_periods,
    )
    setup = (
        active
        & direction.ne(0.0)
        & response.gt(0.0)
        & burst.ge(burst_threshold)
        & hhi.ge(hhi_threshold)
    )
    setup_previous = setup.shift(1, fill_value=False)
    setup_direction = direction.shift(1)
    confirmation = (
        active
        & direction.eq(-setup_direction)
        & (setup_direction * micro_return).lt(0.0)
    )
    primary = setup_previous & confirmation
    primary_side = pd.Series(
        np.where(primary, -setup_direction, 0), index=frame.index, dtype=np.int8
    )

    def signal_frame(
        mask: pd.Series,
        side: pd.Series,
        *,
        branch: str,
        lookback_bars: int,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": frame["date"],
                "side": np.where(mask, side, 0).astype(np.int8),
                "branch": np.where(mask, branch, "none"),
                "hold_bars": np.where(mask, policy.hold_bars, 0).astype(np.int16),
                "lookback_bars": np.where(mask, lookback_bars, 0).astype(np.int16),
            }
        )

    rng = np.random.default_rng(20_260_717)
    permuted_direction = pd.Series(
        rng.permutation(direction.fillna(0.0).to_numpy(float)), index=frame.index
    )
    permuted_confirmation = (
        active
        & permuted_direction.eq(-setup_direction)
        & (setup_direction * micro_return).lt(0.0)
    )
    shifted_setup = setup.shift(288, fill_value=False)
    shifted_setup_direction = direction.shift(288)
    one_day_confirmation = (
        active
        & direction.eq(-shifted_setup_direction)
        & (shifted_setup_direction * micro_return).lt(0.0)
    )
    one_day = shifted_setup & one_day_confirmation
    one_day_side = pd.Series(
        np.where(one_day, -shifted_setup_direction, 0),
        index=frame.index,
        dtype=np.int8,
    )
    no_reversal_side = pd.Series(
        np.where(setup, -direction, 0), index=frame.index, dtype=np.int8
    )
    delayed = primary.shift(1, fill_value=False)
    delayed_side = primary_side.shift(1, fill_value=0).astype(np.int8)
    permuted = setup_previous & permuted_confirmation
    permuted_side = pd.Series(
        np.where(permuted, -setup_direction, 0), index=frame.index, dtype=np.int8
    )

    signals = {
        "primary": signal_frame(
            primary,
            primary_side,
            branch="lvrt_r0",
            lookback_bars=1,
        ),
        "direction_flip": signal_frame(
            primary,
            -primary_side,
            branch="direction_flip",
            lookback_bars=1,
        ),
        "no_reversal_confirmation": signal_frame(
            setup,
            no_reversal_side,
            branch="no_reversal_confirmation",
            lookback_bars=0,
        ),
        "one_bar_extra_delay": signal_frame(
            delayed,
            delayed_side,
            branch="one_bar_extra_delay",
            lookback_bars=2,
        ),
        "one_day_shifted_setup": signal_frame(
            one_day,
            one_day_side,
            branch="one_day_shifted_setup",
            lookback_bars=288,
        ),
        "sign_permuted_confirmation": signal_frame(
            permuted,
            permuted_side,
            branch="sign_permuted_confirmation",
            lookback_bars=1,
        ),
    }
    diagnostics = {
        "clean": clean,
        "direction": direction,
        "burst_threshold": burst_threshold,
        "hhi_threshold": hhi_threshold,
        "setup": setup,
        "confirmation": confirmation,
        "primary": primary,
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
        raise ValueError("schedule start must precede end")
    dates = frame["date"]
    period = dates.ge(start_timestamp).to_numpy(bool) & dates.lt(end_timestamp).to_numpy(bool)
    quarantined = frame["quarantined"].to_numpy(bool)
    sides = signal["side"].to_numpy(np.int8)
    # Production clocks exceed int16 row positions after roughly four months.
    # Widen before any position arithmetic even though individual hold/lookback
    # values themselves are small.
    holds = signal["hold_bars"].to_numpy(np.int64)
    lookbacks = signal["lookback_bars"].to_numpy(np.int64)
    branches = signal["branch"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    previous_exit = -1
    for signal_position in np.flatnonzero(sides):
        setup_position = int(signal_position - int(lookbacks[signal_position]))
        entry_position = int(signal_position + 1)
        exit_position = int(entry_position + int(holds[signal_position]))
        if setup_position < 0 or exit_position >= len(frame):
            continue
        if entry_position < previous_exit:
            continue
        if not period[setup_position : exit_position + 1].all():
            continue
        if quarantined[setup_position : exit_position + 1].any():
            continue
        rows.append(
            {
                "setup_position": setup_position,
                "signal_position": int(signal_position),
                "entry_position": entry_position,
                "exit_position": exit_position,
                "setup_date": str(dates.iloc[setup_position]),
                "signal_date": str(dates.iloc[signal_position]),
                "entry_date": str(dates.iloc[entry_position]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": int(sides[signal_position]),
                "branch": str(branches[signal_position]),
                "hold_bars": int(holds[signal_position]),
            }
        )
        previous_exit = exit_position
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _support(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {
            "nonoverlap_total": 0,
            "by_year": {str(year): 0 for year in range(2020, 2024)},
            "2023_h1": 0,
            "2023_h2": 0,
            "long_share": 0.0,
            "short_share": 0.0,
            "maximum_single_month_share": 0.0,
            "passes_support": False,
        }
    dates = pd.to_datetime(schedule["entry_date"], errors="raise")
    by_year = {
        str(year): int(dates.dt.year.eq(year).sum()) for year in range(2020, 2024)
    }
    h1 = int(((dates >= "2023-01-01") & (dates < "2023-07-01")).sum())
    h2 = int(((dates >= "2023-07-01") & (dates < "2024-01-01")).sum())
    total = int(len(schedule))
    long_share = float(schedule["side"].eq(1).mean())
    short_share = float(schedule["side"].eq(-1).mean())
    month_counts = dates.dt.to_period("M").value_counts()
    maximum_month_share = float(month_counts.max() / total)
    passes = (
        total >= 250
        and all(count >= 40 for count in by_year.values())
        and h1 >= 20
        and h2 >= 20
        and 0.25 <= long_share <= 0.75
        and 0.25 <= short_share <= 0.75
        and maximum_month_share <= 0.20
    )
    return {
        "nonoverlap_total": total,
        "by_year": by_year,
        "2023_h1": h1,
        "2023_h2": h2,
        "long_share": long_share,
        "short_share": short_share,
        "maximum_single_month_share": maximum_month_share,
        "passes_support": bool(passes),
    }


def _entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_entries = set(left["entry_position"].astype(int))
    right_entries = set(right["entry_position"].astype(int))
    union = left_entries | right_entries
    return float(len(left_entries & right_entries) / len(union)) if union else 0.0


def run_support(policy: prereg.Policy | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    frozen = prereg.Policy() if policy is None else policy
    if frozen != prereg.Policy():
        raise ValueError("LVRT-R0 support policy is frozen")
    frame, source = load_support_frame(frozen)
    signals, diagnostics = classify_signals(frame, frozen)
    schedules = {
        name: nonoverlapping_schedule(signal, frame)
        for name, signal in signals.items()
    }
    primary = schedules["primary"]
    support = _support(primary)
    controls = {
        name: {
            "raw_signal_count": int(signals[name]["side"].ne(0).sum()),
            "nonoverlap_count": int(len(schedule)),
            "entry_jaccard_with_primary": _entry_jaccard(primary, schedule),
        }
        for name, schedule in schedules.items()
        if name != "primary"
    }
    finite_thresholds = diagnostics["burst_threshold"].notna() & diagnostics[
        "hhi_threshold"
    ].notna()
    result = {
        "protocol": {
            "name": "LVRT-R0 — Liquidity Vacuum Replenishment Transition",
            "support_only": True,
            "outcomes_opened": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "selection_end_exclusive": str(SELECTION_END),
            "market_values_loaded": False,
            "entry": "next 5m timestamp after two completed feature buckets",
            "hold": f"fixed {frozen.hold_bars} bars",
        },
        "policy": asdict(frozen),
        "source": source,
        "feature_support": {
            "bars_with_both_prior_thresholds": int(finite_thresholds.sum()),
            "raw_setup_count": int(diagnostics["setup"].sum()),
            "raw_confirmation_count": int(diagnostics["confirmation"].sum()),
            "raw_primary_count": int(diagnostics["primary"].sum()),
        },
        "support": support,
        "controls": controls,
        "support_decision": "pass" if support["passes_support"] else "reject_before_returns",
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
            raise RuntimeError(f"refusing to overwrite frozen LVRT-R0 artifact: {output}")
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
        raise ValueError("LVRT-R0 clock path is frozen")
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
