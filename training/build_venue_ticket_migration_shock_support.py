"""Freeze outcome-blind VTMS-288 event and falsification clocks.

Only timestamps and preregistered contemporaneous Spot/USD-M features are
parsed.  The execution market is hash-verified but only its ``date`` column is
read; future OHLC, funding, return, CAGR, and drawdown are unavailable here.
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

from training import preregister_venue_ticket_migration_shock as prereg


PREREGISTRATION_COMMIT = "95b28513004536e7337f298ae400a9c4af5a7d78"
PREREGISTRATION_SOURCE_SHA256 = (
    "1748e233e079534945f57b45e66628008068b261a5652d18a29a7327768e68ce"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/venue-ticket-migration-shock-preregistration-2026-07-17.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "32f445a5c3bac4e231e19a67075efadaae6b6693c1feb3bc63e36903b917e28f"
)
PREREGISTRATION_RESULT = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_RESULT_SHA256 = (
    "04fff22aae17d8cba1a94dc5fb08746c2501013a26a353daa901fa06b548a8ed"
)
DEFAULT_OUTPUT = "results/venue_ticket_migration_shock_support_2026-07-17.json"
DEFAULT_CLOCK = "results/venue_ticket_migration_shock_clock_2026-07-17.csv"
END_EXCLUSIVE = pd.Timestamp("2024-01-01")
SPOT_COLUMNS = (
    "source_complete",
    "mean_trade_notional",
    "signed_quote_notional",
    "flow_coherence",
    "signed_price_response",
)
PERP_COLUMNS = (
    "agg_trade_count",
    "underlying_trade_count",
    "quote_notional",
    "signed_quote_notional",
    "flow_coherence",
    "signed_price_response",
)
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "no_ticket_level",
    "no_ticket_shock",
    "no_coherence",
    "no_price_acceptance",
    "other_venue_side",
    "one_hour_signal_delay",
    "one_day_shifted_clock",
    "random_side",
)
COMPONENT_CONTROLS = (
    "no_ticket_level",
    "no_ticket_shock",
    "no_coherence",
    "no_price_acceptance",
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
    content = (
        schedule[list(CLOCK_COLUMNS)]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(content).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (Path(prereg.__file__), PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen VTMS-288 preregistration changed: {path}")
    payload = json.loads(PREREGISTRATION_RESULT.read_text())
    prereg.validate_manifest(payload)
    if payload.get("outcomes_opened") is not False:
        raise ValueError("VTMS-288 outcomes opened before support freeze")
    if payload.get("policy") != asdict(prereg.Policy()):
        raise ValueError("VTMS-288 support policy differs from preregistration")
    return payload


def _assert_complete_grid(dates: pd.Series) -> None:
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("market timestamps must be nonempty, unique, and monotonic")
    if not dates.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("market timestamps must form a complete five-minute grid")


def quarantine_mask(source_available: pd.Series, post_gap_bars: int) -> pd.Series:
    if post_gap_bars < 0:
        raise ValueError("post-gap quarantine cannot be negative")
    return (
        (~source_available.astype(bool))
        .astype(np.int8)
        .rolling(post_gap_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )


def _perp_gap_days(audit: dict[str, Any]) -> set[str]:
    return set(audit["manifest_diagnostics"]["source_gap_days"])


def load_support_frame(
    policy: prereg.Policy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registration = verify_preregistration()
    source = registration["source_contract"]
    source_files = (
        (source["spot_features"], source["spot_feature_sha256"]),
        (source["spot_manifest"], source["spot_manifest_sha256"]),
        (source["spot_audit"], source["spot_audit_sha256"]),
        (source["perp_features"], source["perp_feature_sha256"]),
        (source["perp_manifest"], source["perp_manifest_sha256"]),
        (source["perp_audit"], source["perp_audit_sha256"]),
        (source["market"], source["market_sha256"]),
        (source["market_manifest"], source["market_manifest_sha256"]),
    )
    for path, expected in source_files:
        if _sha256(path) != expected:
            raise ValueError(f"VTMS-288 source hash changed: {path}")

    spot_manifest = json.loads(Path(source["spot_manifest"]).read_text())
    perp_manifest = json.loads(Path(source["perp_manifest"]).read_text())
    market_manifest = json.loads(Path(source["market_manifest"]).read_text())
    spot_audit = json.loads(Path(source["spot_audit"]).read_text())
    perp_audit = json.loads(Path(source["perp_audit"]).read_text())
    for label, manifest in (
        ("spot", spot_manifest),
        ("perp", perp_manifest),
        ("market", market_manifest),
    ):
        if manifest.get("protocol", {}).get("outcomes_opened") is not False:
            raise ValueError(f"VTMS-288 {label} source manifest opened outcomes")
    if spot_audit.get("outcomes_opened") is not False:
        raise ValueError("VTMS-288 Spot audit opened outcomes")
    if spot_audit.get("decision") != "pass_with_fail_closed_quarantine":
        raise ValueError("VTMS-288 Spot audit did not pass")
    if perp_audit.get("passed") is not True or perp_audit.get("failed_checks"):
        raise ValueError("VTMS-288 USD-M audit did not pass")
    if spot_manifest.get("combined_sha256") != source["spot_feature_sha256"]:
        raise ValueError("VTMS-288 Spot manifest combined hash changed")
    if perp_manifest.get("combined_sha256") != source["perp_feature_sha256"]:
        raise ValueError("VTMS-288 USD-M manifest combined hash changed")
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("VTMS-288 market manifest combined hash changed")

    market = pd.read_csv(
        source["market"], compression="gzip", usecols=["date"], parse_dates=["date"]
    )
    spot = pd.read_csv(
        source["spot_features"],
        compression="gzip",
        usecols=["date", *SPOT_COLUMNS],
        parse_dates=["date"],
    ).rename(columns={column: f"spot_{column}" for column in SPOT_COLUMNS})
    perp = pd.read_csv(
        source["perp_features"],
        compression="gzip",
        usecols=["date", *PERP_COLUMNS],
        parse_dates=["date"],
    )
    _assert_complete_grid(market["date"])
    for label, features in (("Spot", spot), ("USD-M", perp)):
        if (
            features["date"].duplicated().any()
            or not features["date"].is_monotonic_increasing
        ):
            raise ValueError(f"VTMS-288 {label} feature timestamps are invalid")
        if features["date"].max() >= END_EXCLUSIVE:
            raise ValueError(f"VTMS-288 {label} support source contains 2024+")
    if market["date"].max() >= END_EXCLUSIVE:
        raise ValueError("VTMS-288 market support source contains 2024+")

    frame = market.merge(perp, on="date", how="left", validate="one_to_one")
    frame = frame.merge(spot, on="date", how="left", validate="one_to_one")
    spot_numeric = [
        f"spot_{column}" for column in SPOT_COLUMNS if column != "source_complete"
    ]
    spot_available = frame["spot_source_complete"].eq(True).fillna(False) & frame[
        spot_numeric
    ].notna().all(axis=1)
    gap_days = _perp_gap_days(perp_audit)
    perp_available = frame[list(PERP_COLUMNS)].notna().all(axis=1) & ~frame[
        "date"
    ].dt.strftime("%Y-%m-%d").isin(gap_days)
    frame["spot_quarantined"] = quarantine_mask(
        spot_available, policy.post_gap_quarantine_bars
    )
    frame["perp_quarantined"] = quarantine_mask(
        perp_available, policy.post_gap_quarantine_bars
    )
    frame["quarantined"] = frame["spot_quarantined"] | frame["perp_quarantined"]
    frame["spot_available"] = spot_available
    frame["perp_available"] = perp_available
    source_metadata = {
        "spot_feature_sha256": _sha256(source["spot_features"]),
        "spot_manifest_sha256": _sha256(source["spot_manifest"]),
        "spot_audit_sha256": _sha256(source["spot_audit"]),
        "perp_feature_sha256": _sha256(source["perp_features"]),
        "perp_manifest_sha256": _sha256(source["perp_manifest"]),
        "perp_audit_sha256": _sha256(source["perp_audit"]),
        "market_sha256": _sha256(source["market"]),
        "market_manifest_sha256": _sha256(source["market_manifest"]),
        "market_columns_loaded": ["date"],
        "spot_columns_loaded": ["date", *SPOT_COLUMNS],
        "perp_columns_loaded": ["date", *PERP_COLUMNS],
        "price_or_outcome_columns_loaded": [],
        "market_rows": int(len(market)),
        "spot_rows": int(len(spot)),
        "perp_rows": int(len(perp)),
        "spot_missing_or_incomplete_bars": int((~spot_available).sum()),
        "perp_missing_or_gap_bars": int((~perp_available).sum()),
        "joint_quarantined_bars": int(frame["quarantined"].sum()),
        "perp_source_gap_days": sorted(gap_days),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
    }
    return frame, source_metadata


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
    if not 1 <= min_periods <= window:
        raise ValueError("VTMS-288 rolling baseline is invalid")
    return (
        pd.to_numeric(values, errors="coerce")
        .where(clean.astype(bool))
        .shift(1)
        .rolling(window, min_periods=min_periods)
        .quantile(quantile)
    )


def _episode_start(state: pd.Series, reset_bars: int) -> pd.Series:
    if reset_bars < 1:
        raise ValueError("episode reset must be positive")
    previous = (
        state.astype(np.int8)
        .shift(1, fill_value=0)
        .rolling(reset_bars, min_periods=1)
        .max()
        .astype(bool)
    )
    return state.astype(bool) & ~previous


def _signal_frame(
    frame: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    *,
    high_side: pd.Series,
    low_side: pd.Series,
    label: str,
    policy: prereg.Policy,
    signal_shift: int = 0,
) -> pd.DataFrame:
    rows = len(frame)
    active = high.astype(bool) | low.astype(bool)
    origin = np.flatnonzero(
        _episode_start(active, policy.episode_reset_bars).to_numpy()
    )
    signals = origin + signal_shift
    inside = signals < rows
    origin = origin[inside]
    signals = signals[inside]
    side_values = np.where(
        high.iloc[origin].to_numpy(bool),
        high_side.iloc[origin].to_numpy(np.int8),
        low_side.iloc[origin].to_numpy(np.int8),
    )
    branch_values = np.where(
        high.iloc[origin].to_numpy(bool),
        f"{label}_spot_dominant",
        f"{label}_perp_dominant",
    )
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
        output.loc[signals, "origin_position"] = origin
        output.loc[signals, "side"] = side_values
        output.loc[signals, "branch"] = branch_values
        output.loc[signals, "delay_bars"] = policy.execution_delay_bars
        output.loc[signals, "hold_bars"] = policy.hold_bars
    return output


def classify_signals(
    frame: pd.DataFrame,
    policy: prereg.Policy,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    clean = ~frame["quarantined"].astype(bool)
    spot_ticket = pd.to_numeric(frame["spot_mean_trade_notional"], errors="coerce")
    perp_ticket = pd.to_numeric(frame["quote_notional"], errors="coerce").divide(
        pd.to_numeric(frame["underlying_trade_count"], errors="coerce").replace(
            0.0, np.nan
        )
    )
    ticket_ratio = np.log(spot_ticket / perp_ticket)
    ticket_change = ticket_ratio - ticket_ratio.shift(policy.ticket_change_bars)
    spot_coherence = pd.to_numeric(frame["spot_flow_coherence"], errors="coerce")
    perp_coherence = pd.to_numeric(frame["flow_coherence"], errors="coerce")
    spot_response = pd.to_numeric(frame["spot_signed_price_response"], errors="coerce")
    perp_response = pd.to_numeric(frame["signed_price_response"], errors="coerce")
    spot_flow = pd.to_numeric(frame["spot_signed_quote_notional"], errors="coerce")
    perp_flow = pd.to_numeric(frame["signed_quote_notional"], errors="coerce")
    pq = lambda values, quantile: prior_clean_quantile(  # noqa: E731
        values,
        clean,
        quantile=quantile,
        window=policy.baseline_bars,
        min_periods=policy.baseline_min_periods,
    )
    thresholds = {
        "ticket_high": pq(ticket_ratio, policy.ticket_level_quantile),
        "ticket_low": pq(ticket_ratio, 1.0 - policy.ticket_level_quantile),
        "change_high": pq(ticket_change, policy.ticket_change_quantile),
        "change_low": pq(ticket_change, 1.0 - policy.ticket_change_quantile),
        "spot_coherence": pq(spot_coherence, policy.coherence_quantile),
        "perp_coherence": pq(perp_coherence, policy.coherence_quantile),
        "spot_response": pq(spot_response, policy.response_quantile),
        "perp_response": pq(perp_response, policy.response_quantile),
    }
    current_and_anchor_clean = clean & clean.shift(
        policy.ticket_change_bars, fill_value=False
    )
    common = (
        current_and_anchor_clean
        & frame["agg_trade_count"].ge(policy.minimum_perp_agg_trade_count)
        & ticket_ratio.notna()
        & ticket_change.notna()
        & spot_flow.ne(0.0)
        & perp_flow.ne(0.0)
        & pd.concat(list(thresholds.values()), axis=1).notna().all(axis=1)
    )
    ticket_high = ticket_ratio.ge(thresholds["ticket_high"])
    ticket_low = ticket_ratio.le(thresholds["ticket_low"])
    shock_high = ticket_change.ge(thresholds["change_high"])
    shock_low = ticket_change.le(thresholds["change_low"])
    spot_coherent = spot_coherence.ge(thresholds["spot_coherence"])
    perp_coherent = perp_coherence.ge(thresholds["perp_coherence"])
    spot_accepted = spot_response.ge(thresholds["spot_response"])
    perp_accepted = perp_response.ge(thresholds["perp_response"])
    states = {
        "primary": (
            common & ticket_high & shock_high & spot_coherent & spot_accepted,
            common & ticket_low & shock_low & perp_coherent & perp_accepted,
        ),
        "no_ticket_level": (
            common & shock_high & spot_coherent & spot_accepted,
            common & shock_low & perp_coherent & perp_accepted,
        ),
        "no_ticket_shock": (
            common & ticket_high & spot_coherent & spot_accepted,
            common & ticket_low & perp_coherent & perp_accepted,
        ),
        "no_coherence": (
            common & ticket_high & shock_high & spot_accepted,
            common & ticket_low & shock_low & perp_accepted,
        ),
        "no_price_acceptance": (
            common & ticket_high & shock_high & spot_coherent,
            common & ticket_low & shock_low & perp_coherent,
        ),
    }
    spot_side = np.sign(spot_flow).fillna(0).astype(np.int8)
    perp_side = np.sign(perp_flow).fillna(0).astype(np.int8)
    primary_high, primary_low = states["primary"]
    primary = _signal_frame(
        frame,
        primary_high,
        primary_low,
        high_side=spot_side,
        low_side=perp_side,
        label="primary",
        policy=policy,
    )
    signals = {
        "primary": primary,
        "direction_flip": primary.assign(side=-primary["side"]),
    }
    for name in COMPONENT_CONTROLS:
        high, low = states[name]
        signals[name] = _signal_frame(
            frame,
            high,
            low,
            high_side=spot_side,
            low_side=perp_side,
            label=name,
            policy=policy,
        )
    signals["other_venue_side"] = _signal_frame(
        frame,
        primary_high,
        primary_low,
        high_side=perp_side,
        low_side=spot_side,
        label="other_venue_side",
        policy=policy,
    )
    signals["one_hour_signal_delay"] = _signal_frame(
        frame,
        primary_high,
        primary_low,
        high_side=spot_side,
        low_side=perp_side,
        label="one_hour_signal_delay",
        policy=policy,
        signal_shift=12,
    )
    signals["one_day_shifted_clock"] = _signal_frame(
        frame,
        primary_high,
        primary_low,
        high_side=spot_side,
        low_side=perp_side,
        label="one_day_shifted_clock",
        policy=policy,
        signal_shift=288,
    )
    random_side = primary.copy()
    rng = np.random.default_rng(20_260_717)
    active = random_side["side"].ne(0)
    random_side.loc[active, "side"] = rng.choice(
        np.array([-1, 1], dtype=np.int8), size=int(active.sum())
    )
    random_side.loc[active, "branch"] = "random_side"
    signals["random_side"] = random_side
    if set(signals) != set(POLICY_NAMES):
        raise RuntimeError("VTMS-288 policy clock set is incomplete")
    diagnostics = {
        **thresholds,
        "ticket_ratio": ticket_ratio,
        "ticket_change": ticket_change,
        **{f"{name}_spot": high for name, (high, _) in states.items()},
        **{f"{name}_perp": low for name, (_, low) in states.items()},
    }
    return signals, diagnostics


def _half_boundaries() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-07-01"))
        for year in range(2020, 2024)
    ] + [
        (pd.Timestamp(f"{year}-07-01"), pd.Timestamp(f"{year + 1}-01-01"))
        for year in range(2020, 2024)
    ]


def nonoverlapping_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    dates = frame["date"]
    rows: list[dict[str, Any]] = []
    for start, end in sorted(_half_boundaries()):
        next_free = -1
        candidates = dates.ge(start) & dates.lt(end) & signal["side"].ne(0)
        for signal_position in np.flatnonzero(candidates.to_numpy()):
            origin_position = int(signal.loc[signal_position, "origin_position"])
            delay = int(signal.loc[signal_position, "delay_bars"])
            hold = int(signal.loc[signal_position, "hold_bars"])
            entry_position = signal_position + delay
            exit_position = entry_position + hold
            if signal_position < next_free or origin_position < 0:
                continue
            if exit_position >= len(frame):
                continue
            positions = (
                origin_position,
                signal_position,
                entry_position,
                exit_position,
            )
            if not all(start <= dates.iloc[position] < end for position in positions):
                continue
            if frame["quarantined"].iloc[origin_position]:
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
    yearly = {
        str(year): int(entry.dt.year.eq(year).sum()) for year in range(2020, 2024)
    }
    halves = {
        f"{year}_h{half}": int(
            (
                entry.dt.year.eq(year)
                & (entry.dt.month.le(6) if half == 1 else entry.dt.month.ge(7))
            ).sum()
        )
        for year in range(2020, 2024)
        for half in (1, 2)
    }
    total = int(len(schedule))
    long_share = float(side.gt(0).mean()) if total else 0.0
    short_share = float(side.lt(0).mean()) if total else 0.0
    spot_branch = schedule["branch"].str.endswith("spot_dominant")
    perp_branch = schedule["branch"].str.endswith("perp_dominant")
    spot_share = float(spot_branch.mean()) if total else 0.0
    perp_share = float(perp_branch.mean()) if total else 0.0
    month_counts = entry.dt.to_period("M").value_counts()
    max_month_share = float(month_counts.max() / total) if total else 1.0
    frozen = prereg.build_manifest()["support_freeze_before_returns"]
    policy = prereg.Policy()
    passes = (
        int(train.sum()) >= frozen["minimum_nonoverlap_train_2020_2022"]
        and min(yearly[str(year)] for year in range(2020, 2023))
        >= frozen["minimum_nonoverlap_each_train_year"]
        and int(selection.sum()) >= frozen["minimum_nonoverlap_2023"]
        and min(halves["2023_h1"], halves["2023_h2"])
        >= frozen["minimum_nonoverlap_each_2023_half"]
        and min(long_share, short_share) >= frozen["minimum_each_side_share"]
        and min(spot_share, perp_share) >= frozen["minimum_each_branch_share"]
        and max_month_share <= frozen["maximum_single_month_share"]
        and schedule["delay_bars"].eq(policy.execution_delay_bars).all()
        and schedule["hold_bars"].eq(policy.hold_bars).all()
    )
    return {
        "nonoverlap_total": total,
        "train_2020_2022": int(train.sum()),
        "selection_2023": int(selection.sum()),
        "yearly": yearly,
        "halves": halves,
        "long_count": int(side.gt(0).sum()),
        "short_count": int(side.lt(0).sum()),
        "long_share": long_share,
        "short_share": short_share,
        "spot_branch_count": int(spot_branch.sum()),
        "perp_branch_count": int(perp_branch.sum()),
        "spot_branch_share": spot_share,
        "perp_branch_share": perp_share,
        "maximum_single_month_share": max_month_share,
        "passes_support": bool(passes),
    }


def run_support(policy: prereg.Policy) -> dict[str, Any]:
    if policy != prereg.Policy():
        raise ValueError("VTMS-288 policy is frozen; mutation is forbidden")
    frame, source = load_support_frame(policy)
    signals, diagnostics = classify_signals(frame, policy)
    schedules = {
        name: nonoverlapping_schedule(signals[name], frame) for name in POLICY_NAMES
    }
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
            "version": "venue_ticket_migration_shock_support_v1",
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
            if name
            in {
                "ticket_high",
                "ticket_low",
                "change_high",
                "change_low",
                "spot_coherence",
                "perp_coherence",
                "spot_response",
                "perp_response",
            }
        },
        "raw_state_counts": {
            name: int(values.sum())
            for name, values in diagnostics.items()
            if name.endswith("_spot") or name.endswith("_perp")
        },
        "support": support,
        "primary_clock_sha256": _clock_sha256(schedules["primary"]),
        "controls": controls,
        "support_decision": "pass"
        if support["passes_support"]
        else "reject_before_returns",
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
            raise RuntimeError(
                f"refusing to overwrite frozen VTMS-288 artifact: {output}"
            )
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
        raise RuntimeError("VTMS-288 clock hash changed during serialization")
    clock_status = _write_once(args.clock, clock_bytes)
    result_bytes = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
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
