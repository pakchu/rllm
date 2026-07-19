"""Freeze and run outcome-blind support for LCLR-24.

LCLR-24 observes only completed Coinbase BTC-USD and Binance BTCUSDT
perpetual candles from 15:00 through 16:00 Europe/London.  This module never
loads funding, a post-window execution bar, a return label, PnL, or a 2023+
source row.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


POLICY_ID = "LCLR-24"
LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")

COINBASE_SOURCE = Path("data/coinbase_btcusd_5m_2020_2022.csv.gz")
COINBASE_SHA256 = (
    "07f7a3bddecbbc3724994645b9ac1cd0f391378e0feed421f2c8caa145aab77b"
)
BINANCE_SOURCE = Path(
    "data/coinbase_leadership_binance_5m_2020_2022.csv.gz"
)
BINANCE_SHA256 = (
    "1a06f1f4dbbdafaf885fb03844426eed5d5bad4aa206fa72b88db2cbd98bef94"
)
SOURCE_MANIFEST = Path(
    "results/coinbase_spot_leadership_source_manifest_2026-07-16.json"
)
SOURCE_MANIFEST_SHA256 = (
    "3af321fdcafd0fe6680c4583341b6508124a979fefbf489f8d3376c7ec78a269"
)
SOURCE_MANIFEST_HASH = (
    "243ecba3b9e31548d682084dd5acc2e89c6a24423bce241dd6338a57dd6eefe9"
)
SOURCE_DECISION = Path(
    "docs/london-cash-lead-release-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "1183980eb392687d668af0c5a9d5a2631729952418de94fff54f670f25a4521c"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/london-cash-lead-release-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_london_cash_lead_release.py"
)


@dataclass(frozen=True)
class Config:
    support_output: str = (
        "results/london_cash_lead_release_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/london_cash_lead_release_event_clock_2026-07-20.json"
    )
    lookback_windows: int = 126
    minimum_prior_windows: int = 63
    displacement_quantile: float = 0.50
    coherence_quantile: float = 0.50
    participation_quantile: float = 0.50
    optional_votes_required: int = 2
    final_partition_count: int = 3
    latency_bars: int = 1
    hold_bars: int = 24
    minimum_total: int = 180
    minimum_train_2020_2021: int = 110
    minimum_each_train_year: int = 45
    minimum_test_2022: int = 55
    minimum_each_test_half: int = 22
    minimum_each_quarter: int = 8
    minimum_side_share: float = 0.30
    maximum_quarter_share: float = 0.18


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(cfg: Config) -> None:
    expected = Config(
        support_output=cfg.support_output,
        event_clock_output=cfg.event_clock_output,
    )
    if cfg != expected:
        raise ValueError("LCLR-24 signal and support configuration is frozen")
    anchors = {
        COINBASE_SOURCE: COINBASE_SHA256,
        BINANCE_SOURCE: BINANCE_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
    }
    for path, expected_sha in anchors.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"LCLR frozen source anchor mismatch: {path}")


def _utc_naive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _london_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.tz_convert(LONDON)


def _number(token: str) -> float:
    return float(token) if token.strip() else float("nan")


def read_source_window(
    path: str | Path,
    *,
    numeric_columns: tuple[str, ...],
    complete_column: str | None = None,
) -> pd.DataFrame:
    """Parse non-date fields only for weekday London-window source rows."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    rows: list[dict[str, Any]] = []
    prior_date: pd.Timestamp | None = None
    selected_non_date_rows = 0
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        fieldnames = next(csv.reader([header_line]))
        required = {"date", *numeric_columns}
        if complete_column is not None:
            required.add(complete_column)
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"source columns missing: {sorted(missing)}")
        if not fieldnames or fieldnames[0] != "date":
            raise ValueError("date must be the first source column")
        for raw_line in handle:
            date_token = raw_line.split(",", 1)[0]
            timestamp = _utc_naive(date_token)
            if prior_date is not None and timestamp < prior_date:
                raise RuntimeError("source is not chronological")
            prior_date = timestamp
            if not (pd.Timestamp("2020-01-01") <= timestamp < pd.Timestamp("2023-01-01")):
                continue
            local = _london_timestamp(timestamp)
            if local.weekday() >= 5 or local.hour != 15:
                continue
            if local.minute % 5 or local.second or local.microsecond:
                raise RuntimeError("selected source row is not five-minute aligned")
            values = next(csv.reader([raw_line]))
            selected_non_date_rows += 1
            if len(values) != len(fieldnames):
                raise ValueError("malformed selected source row")
            raw = dict(zip(fieldnames, values))
            row: dict[str, Any] = {
                "date": timestamp,
                "local_date": local.date().isoformat(),
                "local_slot": local.minute // 5,
            }
            for column in numeric_columns:
                row[column] = _number(raw[column])
            if complete_column is not None:
                row[complete_column] = int(raw[complete_column])
            rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError(f"no London source rows in {source}")
    if frame["date"].duplicated().any():
        raise RuntimeError("source contains duplicate London timestamps")
    numeric = frame[list(numeric_columns)]
    if complete_column is None:
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError("complete source contains non-finite London values")
    else:
        if not frame[complete_column].isin([0, 1]).all():
            raise ValueError("source_complete must be binary")
        complete = frame[complete_column].eq(1)
        if not np.isfinite(numeric.loc[complete].to_numpy(float)).all():
            raise ValueError("complete source row contains non-finite values")
        if numeric.loc[~complete].notna().any().any():
            raise ValueError("incomplete source row contains imputed values")
    price_columns = [
        column for column in ("open", "high", "low", "close") if column in numeric
    ]
    finite_prices = numeric[price_columns].stack()
    if not finite_prices.empty and finite_prices.le(0.0).any():
        raise ValueError("source prices must be positive")
    for volume_column in ("volume", "quote_asset_volume"):
        if volume_column in numeric and numeric[volume_column].dropna().lt(0.0).any():
            raise ValueError("source volume must be nonnegative")
    frame.attrs["selected_non_date_rows_parsed"] = selected_non_date_rows
    frame.attrs["outside_window_non_date_rows_parsed"] = 0
    frame.attrs["physical_end_exclusive"] = "2023-01-01"
    return frame


def load_source_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise RuntimeError("LCLR source manifest hash mismatch")
    if manifest.get("end_exclusive") != "2023-01-01":
        raise RuntimeError("LCLR source manifest does not end before 2023")
    outputs = manifest.get("outputs", {})
    if outputs.get("coinbase", {}).get("sha256") != COINBASE_SHA256:
        raise RuntimeError("LCLR Coinbase manifest output mismatch")
    if outputs.get("binance", {}).get("sha256") != BINANCE_SHA256:
        raise RuntimeError("LCLR Binance manifest output mismatch")
    coinbase = read_source_window(
        COINBASE_SOURCE,
        numeric_columns=("open", "high", "low", "close", "volume"),
        complete_column="source_complete",
    )
    binance = read_source_window(
        BINANCE_SOURCE,
        numeric_columns=(
            "open",
            "high",
            "low",
            "close",
            "quote_asset_volume",
        ),
    )
    return coinbase, binance


def _window_metrics(group: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    checked = group.sort_values("local_slot").reset_index(drop=True)
    expected_slots = list(range(12))
    complete = (
        len(checked) == 12
        and checked["local_slot"].tolist() == expected_slots
        and checked["source_complete"].eq(1).all()
    )
    numeric = [
        "cb_open",
        "cb_close",
        "cb_volume",
        "bn_open",
        "bn_close",
        "bn_quote_asset_volume",
    ]
    complete = bool(
        complete and np.isfinite(checked[numeric].to_numpy(float)).all()
    )
    local_date = str(checked["local_date"].iloc[0])
    decision_local = pd.Timestamp(f"{local_date} 16:00", tz=LONDON)
    decision_time = decision_local.tz_convert("UTC").tz_localize(None)
    entry_time = decision_time + pd.Timedelta(minutes=5 * cfg.latency_bars)
    exit_time = entry_time + pd.Timedelta(minutes=5 * cfg.hold_bars)
    base = {
        "window_date": local_date,
        "decision_time": decision_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "source_complete": complete,
    }
    if not complete:
        return {
            **base,
            "cash_return": np.nan,
            "perp_return": np.nan,
            "relative_return": np.nan,
            "cash_efficiency": np.nan,
            "final_cash_return": np.nan,
            "cash_quote_share": np.nan,
        }
    cash_path = np.log(
        np.r_[checked["cb_open"].iloc[0], checked["cb_close"].to_numpy(float)]
    )
    perp_path = np.log(
        np.r_[checked["bn_open"].iloc[0], checked["bn_close"].to_numpy(float)]
    )
    cash_steps = np.diff(cash_path)
    perp_steps = np.diff(perp_path)
    cash_return = float(cash_steps.sum())
    perp_return = float(perp_steps.sum())
    path_length = float(np.abs(cash_steps).sum())
    efficiency = abs(cash_return) / path_length if path_length > 0 else 0.0
    final_return = float(cash_steps[-cfg.final_partition_count :].sum())
    cash_notional = float(
        (checked["cb_volume"] * checked["cb_close"]).sum()
    )
    perp_notional = float(checked["bn_quote_asset_volume"].sum())
    total_notional = cash_notional + perp_notional
    cash_share = cash_notional / total_notional if total_notional > 0 else np.nan
    return {
        **base,
        "cash_return": cash_return,
        "perp_return": perp_return,
        "relative_return": cash_return - perp_return,
        "cash_efficiency": efficiency,
        "final_cash_return": final_return,
        "cash_quote_share": cash_share,
    }


def build_window_panel(
    coinbase: pd.DataFrame,
    binance: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    cb = coinbase.rename(
        columns={
            "open": "cb_open",
            "high": "cb_high",
            "low": "cb_low",
            "close": "cb_close",
            "volume": "cb_volume",
        }
    )
    bn = binance.rename(
        columns={
            "open": "bn_open",
            "high": "bn_high",
            "low": "bn_low",
            "close": "bn_close",
            "quote_asset_volume": "bn_quote_asset_volume",
        }
    )
    merged = cb.merge(
        bn,
        on=["date", "local_date", "local_slot"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    merged = merged.drop(columns="_merge")
    records = [
        _window_metrics(group, cfg)
        for _, group in merged.groupby("local_date", sort=True)
    ]
    panel = pd.DataFrame.from_records(records)
    panel["window_date"] = pd.to_datetime(panel["window_date"])
    if panel["window_date"].duplicated().any() or not panel[
        "window_date"
    ].is_monotonic_increasing:
        raise RuntimeError("LCLR daily window panel is not unique/chronological")
    if not panel["window_date"].dt.weekday.lt(5).all():
        raise RuntimeError("LCLR panel contains a non-weekday London window")
    return panel


def prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    cfg: Config,
) -> pd.Series:
    return (
        values.shift(1)
        .rolling(
            cfg.lookback_windows,
            min_periods=cfg.minimum_prior_windows,
        )
        .quantile(quantile)
    )


def build_signal(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    side = np.sign(panel["cash_return"].fillna(0.0)).astype(np.int8)
    displacement_threshold = prior_quantile(
        panel["relative_return"].abs(),
        quantile=cfg.displacement_quantile,
        cfg=cfg,
    )
    coherence_threshold = prior_quantile(
        panel["cash_efficiency"],
        quantile=cfg.coherence_quantile,
        cfg=cfg,
    )
    participation_threshold = prior_quantile(
        panel["cash_quote_share"],
        quantile=cfg.participation_quantile,
        cfg=cfg,
    )
    aligned_cash_lead = (
        side.ne(0)
        & (side * panel["perp_return"]).gt(0.0)
        & (side * panel["relative_return"]).gt(0.0)
    )
    displacement_vote = panel["relative_return"].abs().ge(
        displacement_threshold
    )
    coherence_vote = panel["cash_efficiency"].ge(coherence_threshold)
    participation_vote = panel["cash_quote_share"].ge(
        participation_threshold
    )
    backload_vote = (side * panel["final_cash_return"]).gt(0.0)
    optional_votes = pd.concat(
        [
            displacement_vote,
            coherence_vote,
            participation_vote,
            backload_vote,
        ],
        axis=1,
    ).sum(axis=1)
    thresholds_ready = (
        displacement_threshold.notna()
        & coherence_threshold.notna()
        & participation_threshold.notna()
    )
    candidate = (
        panel["source_complete"]
        & thresholds_ready
        & aligned_cash_lead
        & optional_votes.ge(cfg.optional_votes_required)
    )
    signal_side = side.where(candidate, 0).astype(np.int8)
    return pd.DataFrame(
        {
            "window_date": panel["window_date"],
            "decision_time": panel["decision_time"],
            "entry_time": panel["entry_time"],
            "exit_time": panel["exit_time"],
            "source_complete": panel["source_complete"],
            "candidate": candidate,
            "side": signal_side,
            "aligned_cash_lead": aligned_cash_lead,
            "displacement_vote": displacement_vote,
            "coherence_vote": coherence_vote,
            "participation_vote": participation_vote,
            "backload_vote": backload_vote,
            "optional_votes": optional_votes.astype(np.int8),
            "displacement_threshold": displacement_threshold,
            "coherence_threshold": coherence_threshold,
            "participation_threshold": participation_threshold,
        }
    )


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    dates = schedule["window_date"]
    counts = {
        "total_2020_2022": int(len(schedule)),
        "train_2020_2021": int(dates.dt.year.le(2021).sum()),
        "train_2020": int(dates.dt.year.eq(2020).sum()),
        "train_2021": int(dates.dt.year.eq(2021).sum()),
        "test_2022": int(dates.dt.year.eq(2022).sum()),
        "test_2022_h1": int(
            (dates.dt.year.eq(2022) & dates.dt.month.le(6)).sum()
        ),
        "test_2022_h2": int(
            (dates.dt.year.eq(2022) & dates.dt.month.ge(7)).sum()
        ),
    }
    period_masks = {
        "all": pd.Series(True, index=schedule.index),
        "train": dates.dt.year.le(2021),
        "test": dates.dt.year.eq(2022),
    }
    side_shares: dict[str, dict[str, float]] = {}
    side_checks: dict[str, bool] = {}
    for name, mask in period_masks.items():
        selected = schedule.loc[mask]
        long_share = float(selected["side"].gt(0).mean()) if len(selected) else 0.0
        short_share = float(selected["side"].lt(0).mean()) if len(selected) else 0.0
        side_shares[name] = {"long": long_share, "short": short_share}
        side_checks[name] = min(long_share, short_share) >= cfg.minimum_side_share
    quarter = dates.dt.to_period("Q").astype(str)
    quarter_counts = {
        key: int(value) for key, value in quarter.value_counts().sort_index().items()
    }
    maximum_quarter_share = (
        max(quarter_counts.values()) / len(schedule) if len(schedule) else 1.0
    )
    checks = {
        "total": counts["total_2020_2022"] >= cfg.minimum_total,
        "train_total": counts["train_2020_2021"] >= cfg.minimum_train_2020_2021,
        "train_2020": counts["train_2020"] >= cfg.minimum_each_train_year,
        "train_2021": counts["train_2021"] >= cfg.minimum_each_train_year,
        "test_total": counts["test_2022"] >= cfg.minimum_test_2022,
        "test_h1": counts["test_2022_h1"] >= cfg.minimum_each_test_half,
        "test_h2": counts["test_2022_h2"] >= cfg.minimum_each_test_half,
        "each_quarter": bool(
            quarter_counts
            and min(quarter_counts.values()) >= cfg.minimum_each_quarter
            and len(quarter_counts) == 12
        ),
        "side_all": side_checks["all"],
        "side_train": side_checks["train"],
        "side_test": side_checks["test"],
        "quarter_concentration": maximum_quarter_share
        <= cfg.maximum_quarter_share,
    }
    return {
        "counts": counts,
        "side_shares": side_shares,
        "quarter_counts": quarter_counts,
        "maximum_quarter_share": float(maximum_quarter_share),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def event_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "window_date",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
        "optional_votes",
        "displacement_vote",
        "coherence_vote",
        "participation_vote",
        "backload_vote",
    ]
    records: list[dict[str, Any]] = []
    for row in schedule[columns].to_dict(orient="records"):
        records.append(
            {
                "window_date": str(row["window_date"]),
                "decision_time": str(row["decision_time"]),
                "entry_time": str(row["entry_time"]),
                "exit_time": str(row["exit_time"]),
                "side": int(row["side"]),
                "optional_votes": int(row["optional_votes"]),
                "displacement_vote": bool(row["displacement_vote"]),
                "coherence_vote": bool(row["coherence_vote"]),
                "participation_vote": bool(row["participation_vote"]),
                "backload_vote": bool(row["backload_vote"]),
            }
        )
    return records


def event_clock_hash(
    events: list[dict[str, Any]],
    *,
    cfg: Config,
    protocol_hash: str,
) -> str:
    return canonical_hash(
        {
            "policy_id": POLICY_ID,
            "config": asdict(cfg),
            "protocol_hash": protocol_hash,
            "source_manifest_hash": SOURCE_MANIFEST_HASH,
            "events": events,
        }
    )


def protocol(cfg: Config) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "source": {
            "coinbase": str(COINBASE_SOURCE),
            "coinbase_sha256": COINBASE_SHA256,
            "binance": str(BINANCE_SOURCE),
            "binance_sha256": BINANCE_SHA256,
            "source_manifest": str(SOURCE_MANIFEST),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "selection_end_exclusive": "2023-01-01",
            "funding_loaded": False,
            "post_window_execution_or_outcome_bars_loaded": False,
        },
        "clock": {
            "timezone": "Europe/London",
            "eligible_weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "source_window": "[15:00,16:00) local, twelve completed five-minute bars",
            "decision": "16:00 local after the 15:55 source bar closes",
            "entry": f"{cfg.latency_bars} complete five-minute latency bar later",
            "hold_bars": cfg.hold_bars,
        },
        "feature": {
            "cash_return": "log Coinbase last close / first open across the source window",
            "perp_return": "log Binance last close / first open across the source window",
            "cash_efficiency": "abs cash return / sum abs Coinbase partition returns",
            "relative_return": "cash return - perp return",
            "cash_quote_share": "Coinbase volume*close / (Coinbase volume*close + Binance quote volume)",
            "final_cash_return": f"sum of the final {cfg.final_partition_count} Coinbase partition returns",
            "strict_prior_reference": (
                f"last {cfg.lookback_windows} earlier source windows; incomplete "
                f"values remain NaN and are ignored; require {cfg.minimum_prior_windows} "
                "finite prior values; current row excluded"
            ),
            "mandatory": "same-direction cash/perp move and cash farther in the cash direction",
            "optional_votes": [
                f"abs relative return >= prior q{cfg.displacement_quantile}",
                f"cash efficiency >= prior q{cfg.coherence_quantile}",
                f"cash quote share >= prior q{cfg.participation_quantile}",
                "final 15-minute cash return has the full-window sign",
            ],
            "optional_votes_required": cfg.optional_votes_required,
            "side": "sign of completed Coinbase window return",
        },
        "support_gate": {
            "minimum_total": cfg.minimum_total,
            "minimum_train_2020_2021": cfg.minimum_train_2020_2021,
            "minimum_each_train_year": cfg.minimum_each_train_year,
            "minimum_test_2022": cfg.minimum_test_2022,
            "minimum_each_test_half": cfg.minimum_each_test_half,
            "minimum_each_quarter": cfg.minimum_each_quarter,
            "minimum_side_share_all_train_test": cfg.minimum_side_share,
            "maximum_quarter_share": cfg.maximum_quarter_share,
            "failure_action": "reject before post-window outcomes; no threshold/hold repair",
        },
        "later_evaluation_contract": {
            "train": ["2020-01-01", "2022-01-01"],
            "test": ["2022-01-01", "2023-01-01"],
            "sealed_sequential": ["2023", "2024", "2025", "2026_ytd"],
            "leverage": 0.5,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "funding": (
                "interior exact-time symmetric; exact entry/exit credits dropped "
                "and debits retained"
            ),
            "cagr": "full split wall clock including warmup and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, exact funding, every held 5m "
                "path, virtual adverse exit fee, and actual exit"
            ),
            "primary_gates_each_train_and_test": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "stress_cost_absolute_return_positive": True,
                "one_bar_delayed_absolute_return_positive": True,
                "mean_gross_underlying_bp_min": 20.0,
                "weekly_cluster_signflip_p_max": 0.10,
            },
            "mechanism_controls": [
                "exact side flip",
                "cash-only rule without mandatory cross-venue lead",
                "same rule on a separately prior-normalized 12:00-13:00 London window",
                "weekend 15:00-16:00 London clock",
            ],
        },
        "frozen_artifacts": {
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(PREREGISTRATION_DOCUMENT),
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
        "research_history_boundary": (
            "candidate-level freeze only; unrelated repository research has seen "
            "the market history, but no LCLR event incidence or post-window outcome "
            "was used to choose this singleton"
        ),
    }


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _validate_config(cfg)
    coinbase, binance = load_source_windows()
    panel = build_window_panel(coinbase, binance, cfg)
    signal = build_signal(panel, cfg)
    schedule = signal.loc[signal["candidate"]].reset_index(drop=True)
    summary = support_summary(schedule, cfg)
    events = event_records(schedule)
    protocol_payload = protocol(cfg)
    protocol_hash = canonical_hash(protocol_payload)
    event_hash = event_clock_hash(
        events,
        cfg=cfg,
        protocol_hash=protocol_hash,
    )
    core = {
        "protocol_version": "london_cash_lead_release_support_v1",
        "protocol": protocol_payload,
        "protocol_hash": protocol_hash,
        "outcomes_opened": False,
        "source_loaded": True,
        "source_audit": {
            "coinbase_rows_parsed": int(len(coinbase)),
            "binance_rows_parsed": int(len(binance)),
            "coinbase_outside_window_non_date_rows_parsed": int(
                coinbase.attrs["outside_window_non_date_rows_parsed"]
            ),
            "binance_outside_window_non_date_rows_parsed": int(
                binance.attrs["outside_window_non_date_rows_parsed"]
            ),
            "funding_rows_loaded": 0,
            "post_window_execution_or_outcome_rows_loaded": 0,
            "rows_at_or_after_2023_loaded": 0,
        },
        "window_support": {
            "windows": int(len(panel)),
            "complete_windows": int(panel["source_complete"].sum()),
            "incomplete_windows": int((~panel["source_complete"]).sum()),
            "threshold_ready_windows": int(
                signal[[
                    "displacement_threshold",
                    "coherence_threshold",
                    "participation_threshold",
                ]].notna().all(axis=1).sum()
            ),
            "candidate_windows": int(len(schedule)),
        },
        "support_gate": summary,
        "event_clock_hash": event_hash,
        "event_clock_written": bool(summary["passed"]),
        "sealed": ["all post-window 2020-2022 outcomes", "2023", "2024", "2025", "2026_ytd"],
        "failure_action": (
            None
            if summary["passed"]
            else "reject before outcomes; no parameter, vote, latency, or hold repair"
        ),
    }
    result = {
        **core,
        "result_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = Path(cfg.support_output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    clock: dict[str, Any] | None = None
    if summary["passed"]:
        clock_core = {
            "protocol_version": "london_cash_lead_release_event_clock_v1",
            "policy_id": POLICY_ID,
            "outcomes_opened": False,
            "support_result_hash": result["result_hash"],
            "protocol_hash": protocol_hash,
            "config": asdict(cfg),
            "source_manifest_hash": SOURCE_MANIFEST_HASH,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "coinbase_sha256": COINBASE_SHA256,
            "binance_sha256": BINANCE_SHA256,
            "event_clock_hash": event_hash,
            "events": events,
        }
        clock = {
            **clock_core,
            "manifest_hash": canonical_hash(clock_core),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        clock_path = Path(cfg.event_clock_output)
        clock_path.parent.mkdir(parents=True, exist_ok=True)
        clock_path.write_text(json.dumps(clock, indent=2, ensure_ascii=False) + "\n")
    return result, clock


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-output", default=defaults.support_output)
    parser.add_argument("--event-clock-output", default=defaults.event_clock_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, clock = run_support(Config(**vars(args)))
    print(
        json.dumps(
            {
                "result_hash": result["result_hash"],
                "support_gate": result["support_gate"],
                "event_clock_written": clock is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
