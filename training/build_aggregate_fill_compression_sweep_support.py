"""Freeze outcome-blind AFCS-144 event and falsification clocks.

Only timestamps and the preregistered contemporaneous aggTrade features are
parsed.  The market file is hash-verified but only its ``date`` column is read;
no open, high, low, close, return, funding PnL, CAGR, or MDD is available here.
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

from training import preregister_aggregate_fill_compression_sweep as prereg


PREREGISTRATION_COMMIT = "0b4ac0bc35f3a8c2f7b2e79f4409b4375ee9911b"
PREREGISTRATION_SOURCE_SHA256 = (
    "12499270059c9bf22a400765fbcd7f2a0aec7462e1f473f275212cac1f3269ff"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/aggregate-fill-compression-sweep-preregistration-2026-07-17.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "88dc59755f8ff491922012b09abcb820571406cffd7aed8b970c8b3f445756f8"
)
PREREGISTRATION_RESULT = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_RESULT_SHA256 = (
    "3b4cb52c4b4b6bca59b98422e1114513e408078e746542259adeafbe9fe2a3fe"
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
DEFAULT_OUTPUT = "results/aggregate_fill_compression_sweep_support_2026-07-17.json"
DEFAULT_CLOCK = "results/aggregate_fill_compression_sweep_clock_2026-07-17.csv"
END_EXCLUSIVE = pd.Timestamp("2024-01-01")
FEATURE_COLUMNS = (
    "agg_trade_count",
    "underlying_trades_per_agg_event",
    "quote_notional",
    "signed_quote_notional",
    "flow_coherence",
    "signed_price_response",
)
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "no_compression",
    "no_coherence",
    "no_aligned_response",
    "one_hour_signal_delay",
    "one_day_shifted_clock",
    "random_side",
)
CLOCK_COLUMNS = (
    "origin_position",
    "signal_position",
    "entry_position",
    "exit_position",
    "origin_date",
    "signal_date",
    "entry_date",
    "exit_date",
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


def _clock_sha256(schedule: pd.DataFrame) -> str:
    data = schedule[list(CLOCK_COLUMNS)].to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (Path(prereg.__file__), PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen AFCS-144 preregistration changed: {path}")
    payload = json.loads(PREREGISTRATION_RESULT.read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise ValueError("AFCS-144 outcomes opened before support freeze")
    if payload["policy"] != asdict(prereg.Policy()):
        raise ValueError("AFCS-144 support policy differs from preregistration")
    return payload


def _assert_complete_grid(dates: pd.Series) -> None:
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("market timestamps must be nonempty, unique, and monotonic")
    if not dates.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
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
    registration = verify_preregistration()
    source = registration["source_contract"]
    for path, expected in (
        (source["features"], source["feature_sha256"]),
        (source["feature_manifest"], FEATURE_MANIFEST_SHA256),
        (source["market"], source["market_sha256"]),
        (source["market_manifest"], MARKET_MANIFEST_SHA256),
        (source["source_audit"], AUDIT_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"AFCS-144 source hash changed: {path}")

    feature_manifest = json.loads(Path(source["feature_manifest"]).read_text())
    market_manifest = json.loads(Path(source["market_manifest"]).read_text())
    audit = json.loads(Path(source["source_audit"]).read_text())
    if feature_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("aggTrade source manifest opened outcomes")
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("market source manifest opened outcomes")
    if audit.get("passed") is not True or audit.get("failed_checks"):
        raise ValueError("aggTrade source audit did not pass")
    if feature_manifest.get("combined_sha256") != source["feature_sha256"]:
        raise ValueError("feature manifest combined hash changed")
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("market manifest combined hash changed")

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
        raise ValueError("aggTrade feature timestamps are invalid")
    if market["date"].max() >= END_EXCLUSIVE or features["date"].max() >= END_EXCLUSIVE:
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
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    numeric = pd.to_numeric(values, errors="coerce")
    eligible = clean.astype(bool) & numeric.notna()
    positions = np.flatnonzero(eligible.to_numpy())
    history = numeric.iloc[positions].reset_index(drop=True)
    dense_threshold = (
        history.rolling(window, min_periods=min_periods)
        .quantile(quantile)
        .shift(1)
    )
    output = pd.Series(np.nan, index=values.index, dtype=float)
    output.iloc[positions] = dense_threshold.to_numpy(float)
    return output


def _episode_start(state: pd.Series, reset_bars: int) -> pd.Series:
    if reset_bars < 1:
        raise ValueError("episode_reset_bars must be positive")
    prior_active = (
        state.astype(np.int8)
        .shift(1, fill_value=0)
        .rolling(reset_bars, min_periods=1)
        .max()
        .astype(bool)
    )
    return state.astype(bool) & ~prior_active


def _signal_frame(
    frame: pd.DataFrame,
    active: pd.Series,
    *,
    branch: str,
    side: pd.Series,
    policy: prereg.Policy,
    signal_shift: int = 0,
) -> pd.DataFrame:
    rows = len(frame)
    origin = np.flatnonzero(_episode_start(active, policy.episode_reset_bars).to_numpy())
    signals = origin + signal_shift
    inside = signals < rows
    origin = origin[inside]
    signals = signals[inside]
    output = pd.DataFrame(
        {
            "origin_position": np.full(rows, -1, dtype=np.int64),
            "side": np.zeros(rows, dtype=np.int8),
            "branch": np.full(rows, "none", dtype=object),
            "delay_bars": np.zeros(rows, dtype=np.int16),
            "hold_bars": np.zeros(rows, dtype=np.int16),
        }
    )
    if len(signals):
        chosen_side = side.iloc[origin].to_numpy(np.int8)
        output.loc[signals, "origin_position"] = origin
        output.loc[signals, "side"] = chosen_side
        output.loc[signals, "branch"] = branch
        output.loc[signals, "delay_bars"] = policy.execution_delay_bars
        output.loc[signals, "hold_bars"] = policy.hold_bars
    return output


def classify_signals(
    frame: pd.DataFrame,
    policy: prereg.Policy,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    clean = ~frame["quarantined"].astype(bool)
    thresholds = {
        "compression": prior_clean_quantile(
            frame["underlying_trades_per_agg_event"],
            clean,
            quantile=policy.compression_quantile,
            window=policy.baseline_bars,
            min_periods=policy.baseline_min_periods,
        ),
        "coherence": prior_clean_quantile(
            frame["flow_coherence"],
            clean,
            quantile=policy.coherence_quantile,
            window=policy.baseline_bars,
            min_periods=policy.baseline_min_periods,
        ),
        "response": prior_clean_quantile(
            frame["signed_price_response"],
            clean,
            quantile=policy.response_quantile,
            window=policy.baseline_bars,
            min_periods=policy.baseline_min_periods,
        ),
        "activity": prior_clean_quantile(
            frame["quote_notional"],
            clean,
            quantile=policy.activity_quantile,
            window=policy.baseline_bars,
            min_periods=policy.baseline_min_periods,
        ),
    }
    common = (
        clean
        & frame["source_available"].astype(bool)
        & frame["agg_trade_count"].ge(policy.minimum_agg_trade_count)
        & frame["signed_quote_notional"].ne(0.0)
        & frame["quote_notional"].ge(thresholds["activity"])
        & thresholds["compression"].notna()
        & thresholds["coherence"].notna()
        & thresholds["response"].notna()
        & thresholds["activity"].notna()
    )
    compression = frame["underlying_trades_per_agg_event"].ge(
        thresholds["compression"]
    )
    coherence = frame["flow_coherence"].ge(thresholds["coherence"])
    response = frame["signed_price_response"].ge(thresholds["response"]) & frame[
        "signed_price_response"
    ].gt(0.0)
    states = {
        "primary": common & compression & coherence & response,
        "no_compression": common & coherence & response,
        "no_coherence": common & compression & response,
        "no_aligned_response": common & compression & coherence,
    }
    base_side = np.sign(frame["signed_quote_notional"]).fillna(0).astype(np.int8)
    primary = _signal_frame(
        frame,
        states["primary"],
        branch="afcs_144",
        side=base_side,
        policy=policy,
    )
    signals = {
        "primary": primary,
        "direction_flip": primary.assign(side=-primary["side"]),
        "no_compression": _signal_frame(
            frame,
            states["no_compression"],
            branch="no_compression",
            side=base_side,
            policy=policy,
        ),
        "no_coherence": _signal_frame(
            frame,
            states["no_coherence"],
            branch="no_coherence",
            side=base_side,
            policy=policy,
        ),
        "no_aligned_response": _signal_frame(
            frame,
            states["no_aligned_response"],
            branch="no_aligned_response",
            side=base_side,
            policy=policy,
        ),
        "one_hour_signal_delay": _signal_frame(
            frame,
            states["primary"],
            branch="one_hour_signal_delay",
            side=base_side,
            policy=policy,
            signal_shift=12,
        ),
        "one_day_shifted_clock": _signal_frame(
            frame,
            states["primary"],
            branch="one_day_shifted_clock",
            side=base_side,
            policy=policy,
            signal_shift=288,
        ),
    }
    rng = np.random.default_rng(20_260_717)
    random_side = primary.copy()
    mask = random_side["side"].ne(0)
    random_side.loc[mask, "side"] = rng.choice(
        np.array([-1, 1], dtype=np.int8), size=int(mask.sum())
    )
    random_side.loc[mask, "branch"] = "random_side"
    signals["random_side"] = random_side
    if set(signals) != set(POLICY_NAMES):
        raise RuntimeError("AFCS-144 policy clock set is incomplete")
    diagnostics = {**thresholds, **states}
    return signals, diagnostics


def _half_boundaries(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first_year = int(frame["date"].dt.year.min())
    last_year = int(frame["date"].dt.year.max())
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for year in range(first_year, last_year + 1):
        segments.append((pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-07-01")))
        segments.append((pd.Timestamp(f"{year}-07-01"), pd.Timestamp(f"{year + 1}-01-01")))
    return segments


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    dates = frame["date"]
    quarantined = frame["quarantined"].to_numpy(bool)
    rows: list[dict[str, Any]] = []
    segments = _half_boundaries(frame) if segments is None else segments
    for start, end in segments:
        next_free = -1
        in_segment = dates.ge(start) & dates.lt(end) & signal["side"].ne(0)
        for signal_position in np.flatnonzero(in_segment.to_numpy()):
            origin_position = int(signal.loc[signal_position, "origin_position"])
            delay = int(signal.loc[signal_position, "delay_bars"])
            hold = int(signal.loc[signal_position, "hold_bars"])
            entry_position = signal_position + delay
            exit_position = entry_position + hold
            if signal_position < next_free or origin_position < 0:
                continue
            if exit_position >= len(frame):
                continue
            positions = [origin_position, signal_position, entry_position, exit_position]
            if not all(start <= dates.iloc[position] < end for position in positions):
                continue
            left = min(origin_position, signal_position)
            if quarantined[left : exit_position + 1].any():
                continue
            rows.append(
                {
                    "origin_position": origin_position,
                    "signal_position": signal_position,
                    "entry_position": entry_position,
                    "exit_position": exit_position,
                    "origin_date": str(dates.iloc[origin_position]),
                    "signal_date": str(dates.iloc[signal_position]),
                    "entry_date": str(dates.iloc[entry_position]),
                    "exit_date": str(dates.iloc[exit_position]),
                    "side": int(signal.loc[signal_position, "side"]),
                    "branch": str(signal.loc[signal_position, "branch"]),
                    "delay_bars": delay,
                    "hold_bars": hold,
                }
            )
            next_free = exit_position
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _support(schedule: pd.DataFrame) -> dict[str, Any]:
    entry = pd.to_datetime(schedule["entry_date"], errors="raise")
    side = schedule["side"].astype(int)
    train = entry.lt("2023-01-01")
    selection = entry.ge("2023-01-01") & entry.lt("2024-01-01")
    yearly = {str(year): int((entry.dt.year == year).sum()) for year in range(2020, 2024)}
    halves = {
        f"{year}_h{half}": int(
            ((entry.dt.year == year) & ((entry.dt.month <= 6) if half == 1 else (entry.dt.month >= 7))).sum()
        )
        for year in range(2020, 2024)
        for half in (1, 2)
    }
    total = int(len(schedule))
    train_total = int(train.sum())
    selection_total = int(selection.sum())
    long_share = float((side > 0).mean()) if total else 0.0
    short_share = float((side < 0).mean()) if total else 0.0
    month_counts = entry.dt.to_period("M").value_counts()
    max_month_share = float(month_counts.max() / total) if total else 1.0
    policy = prereg.Policy()
    support = prereg.build_manifest()["support_freeze_before_returns"]
    passes = (
        train_total >= support["minimum_nonoverlap_total_2020_2022"]
        and min(yearly[str(year)] for year in range(2020, 2023))
        >= support["minimum_nonoverlap_each_year_2020_2022"]
        and min(halves[f"{year}_h{half}"] for year in range(2020, 2023) for half in (1, 2))
        >= support["minimum_nonoverlap_each_half_2020_2022"]
        and selection_total >= support["minimum_nonoverlap_2023"]
        and min(halves["2023_h1"], halves["2023_h2"])
        >= support["minimum_nonoverlap_each_2023_half"]
        and support["minimum_each_side_share"] <= long_share <= support["maximum_each_side_share"]
        and support["minimum_each_side_share"] <= short_share <= support["maximum_each_side_share"]
        and max_month_share <= support["maximum_single_month_share"]
        and schedule["delay_bars"].eq(policy.execution_delay_bars).all()
        and schedule["hold_bars"].eq(policy.hold_bars).all()
    )
    return {
        "nonoverlap_total": total,
        "train_2020_2022": train_total,
        "selection_2023": selection_total,
        "yearly": yearly,
        "halves": halves,
        "long_count": int((side > 0).sum()),
        "short_count": int((side < 0).sum()),
        "long_share": long_share,
        "short_share": short_share,
        "maximum_single_month_share": max_month_share,
        "passes_support": bool(passes),
    }


def run_support(policy: prereg.Policy) -> dict[str, Any]:
    if policy != prereg.Policy():
        raise ValueError("AFCS-144 policy is frozen; mutation is forbidden")
    frame, source = load_support_frame(policy)
    signals, diagnostics = classify_signals(frame, policy)
    schedules = {name: nonoverlapping_schedule(signals[name], frame) for name in POLICY_NAMES}
    support = _support(schedules["primary"])
    controls = {
        name: {
            "nonoverlap_count": int(len(schedule)),
            "clock_sha256": _clock_sha256(schedule),
        }
        for name, schedule in schedules.items()
        if name != "primary"
    }
    core: dict[str, Any] = {
        "protocol": {
            "version": "aggregate_fill_compression_sweep_support_v1",
            "outcomes_opened": False,
            "price_or_return_loaded": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_window_end_exclusive": "2024-01-01",
        },
        "policy": asdict(policy),
        "source": source,
        "threshold_availability": {
            name: int(values.notna().sum())
            for name, values in diagnostics.items()
            if name in {"compression", "coherence", "response", "activity"}
        },
        "raw_state_counts": {
            name: int(values.sum())
            for name, values in diagnostics.items()
            if name in {"primary", "no_compression", "no_coherence", "no_aligned_response"}
        },
        "support": support,
        "primary_clock_sha256": _clock_sha256(schedules["primary"]),
        "controls": controls,
        "support_decision": "pass" if support["passes_support"] else "reject_before_returns",
    }
    return {
        **core,
        "manifest_hash": _canonical_hash(core),
        "_primary_schedule": schedules["primary"],
    }


def _write_once(path: str | Path, content: bytes) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite frozen AFCS-144 artifact: {output}")
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
    schedule = payload.pop("_primary_schedule")
    clock_bytes = schedule.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hashlib.sha256(clock_bytes).hexdigest() != payload["primary_clock_sha256"]:
        raise RuntimeError("AFCS-144 clock hash changed during serialization")
    clock_status = _write_once(args.clock, clock_bytes)
    result_bytes = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    result_status = _write_once(args.output, result_bytes)
    print(
        json.dumps(
            {
                "result_status": result_status,
                "clock_status": clock_status,
                "support_decision": payload["support_decision"],
                "support": payload["support"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
                "clock": args.clock,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
