"""Freeze and run outcome-blind support for IFAR-288.

IFAR-288 uses only BitMEX XBt insurance-fund snapshots and the completed
Binance BTCUSDT price move ending at the same 12:00 UTC snapshot.  It never
loads a post-decision bar, funding, return label, PnL, or a 2023+ source row.
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

import numpy as np
import pandas as pd


POLICY_ID = "IFAR-288"
INSURANCE_SOURCE = Path(
    "data/bitmex_xbt_insurance_fund_2018_2022.csv.gz"
)
INSURANCE_MANIFEST = Path(
    "results/bitmex_xbt_insurance_fund_source_manifest_2026-07-20.json"
)
BINANCE_SOURCE = Path(
    "data/coinbase_leadership_binance_5m_2020_2022.csv.gz"
)
BINANCE_SHA256 = (
    "1a06f1f4dbbdafaf885fb03844426eed5d5bad4aa206fa72b88db2cbd98bef94"
)
BINANCE_MANIFEST = Path(
    "results/coinbase_spot_leadership_source_manifest_2026-07-16.json"
)
BINANCE_MANIFEST_SHA256 = (
    "3af321fdcafd0fe6680c4583341b6508124a979fefbf489f8d3376c7ec78a269"
)
SOURCE_DECISION = Path(
    "docs/bitmex-insurance-fund-absorption-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "37943511af6e88ac52b25b37bac3bfcda262b3132640986ec294ff640b76be02"
)
SOURCE_DOWNLOADER = Path(
    "training/download_bitmex_xbt_insurance_fund.py"
)
SOURCE_DOWNLOADER_SHA256 = (
    "06223bcb0681b1c4c4e4fe8f540fea2f14c4085ba93d2c9ef7e98b09a50a8868"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/bitmex-insurance-fund-absorption-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_bitmex_insurance_fund_absorption.py"
)


@dataclass(frozen=True)
class Config:
    support_output: str = (
        "results/bitmex_insurance_fund_absorption_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/bitmex_insurance_fund_absorption_event_clock_2026-07-20.json"
    )
    fund_lookback_days: int = 365
    minimum_prior_loss_days: int = 20
    fund_loss_quantile: float = 0.50
    price_lookback_days: int = 126
    minimum_prior_price_days: int = 90
    price_move_quantile: float = 0.50
    source_embargo_days: int = 1
    latency_bars: int = 1
    hold_bars: int = 288
    eligibility_start: str = "2020-07-01"
    selection_end_exclusive: str = "2023-01-01"
    minimum_total: int = 50
    minimum_train_2020h2_2021: int = 30
    minimum_train_2020h2: int = 8
    minimum_train_2021: int = 20
    minimum_test_2022: int = 20
    minimum_each_test_half: int = 8
    minimum_each_eligible_quarter: int = 2
    minimum_side_share: float = 0.25
    maximum_quarter_share: float = 0.20


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
        raise ValueError("IFAR-288 signal and support configuration is frozen")
    for path, expected_sha in {
        BINANCE_SOURCE: BINANCE_SHA256,
        BINANCE_MANIFEST: BINANCE_MANIFEST_SHA256,
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        SOURCE_DOWNLOADER: SOURCE_DOWNLOADER_SHA256,
    }.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"IFAR frozen source anchor mismatch: {path}")


def _utc_naive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }


def load_insurance_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(INSURANCE_MANIFEST.read_text())
    if manifest.get("protocol_version") != "bitmex_xbt_insurance_source_v1":
        raise RuntimeError("IFAR insurance source manifest version mismatch")
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("IFAR insurance source manifest hash mismatch")
    expected_config = {
        "output_csv": str(INSURANCE_SOURCE),
        "manifest_output": str(INSURANCE_MANIFEST),
        "start": "2018-01-01",
        "end_exclusive": "2023-01-01",
        "currency": "XBt",
        "page_size": 500,
        "timeout_sec": 30.0,
    }
    if manifest.get("config") != expected_config:
        raise RuntimeError("IFAR insurance source request contract mismatch")
    output = manifest.get("output", {})
    if output.get("path") != str(INSURANCE_SOURCE):
        raise RuntimeError("IFAR insurance source path mismatch")
    if sha256_file(INSURANCE_SOURCE) != output.get("sha256"):
        raise RuntimeError("IFAR insurance source file hash mismatch")
    audit = manifest.get("source_audit", {})
    if not audit.get("complete_daily_noon_utc_grid"):
        raise RuntimeError("IFAR insurance source grid is incomplete")
    if audit.get("rows_selected") != 1826 or audit.get("expected_days") != 1826:
        raise RuntimeError("IFAR insurance source row count mismatch")

    frame = pd.read_csv(INSURANCE_SOURCE)
    if list(frame.columns) != ["date", "wallet_balance_satoshi"]:
        raise ValueError("IFAR insurance source columns mismatch")
    frame["snapshot_time"] = pd.to_datetime(frame.pop("date"), errors="raise")
    frame["wallet_balance_satoshi"] = pd.to_numeric(
        frame["wallet_balance_satoshi"], errors="raise"
    )
    if len(frame) != 1826 or frame["snapshot_time"].duplicated().any():
        raise RuntimeError("IFAR insurance source is not unique/complete")
    if not frame["snapshot_time"].is_monotonic_increasing:
        raise RuntimeError("IFAR insurance source is not chronological")
    expected = pd.date_range(
        "2018-01-01 12:00", "2022-12-31 12:00", freq="1D"
    )
    if not frame["snapshot_time"].equals(pd.Series(expected, name="snapshot_time")):
        raise RuntimeError("IFAR insurance source is not the frozen daily grid")
    if frame["wallet_balance_satoshi"].le(0).any():
        raise RuntimeError("IFAR insurance source contains non-positive balance")
    return frame, manifest


def read_daily_snapshot_prices(path: str | Path) -> pd.DataFrame:
    """Parse price fields only from completed 11:55 UTC five-minute bars."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    rows: list[dict[str, Any]] = []
    prior_date: pd.Timestamp | None = None
    selected_non_date_rows = 0
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        fieldnames = next(csv.reader([handle.readline()]))
        if not fieldnames or fieldnames[0] != "date":
            raise ValueError("date must be the first market source column")
        if "close" not in fieldnames:
            raise ValueError("market source is missing close")
        for raw_line in handle:
            date_token = raw_line.split(",", 1)[0]
            timestamp = _utc_naive(date_token)
            if prior_date is not None and timestamp < prior_date:
                raise RuntimeError("market source is not chronological")
            prior_date = timestamp
            if not (
                pd.Timestamp("2020-01-01")
                <= timestamp
                < pd.Timestamp("2023-01-01")
            ):
                continue
            if not (timestamp.hour == 11 and timestamp.minute == 55):
                continue
            values = next(csv.reader([raw_line]))
            selected_non_date_rows += 1
            if len(values) != len(fieldnames):
                raise ValueError("malformed selected market source row")
            raw = dict(zip(fieldnames, values))
            close = float(raw["close"])
            if not np.isfinite(close) or close <= 0.0:
                raise ValueError("selected market close must be finite and positive")
            rows.append(
                {
                    "snapshot_time": timestamp + pd.Timedelta(minutes=5),
                    "snapshot_price": close,
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError("market source has no daily snapshot closes")
    if frame["snapshot_time"].duplicated().any():
        raise RuntimeError("market source has duplicate daily snapshot closes")
    frame.attrs["selected_non_date_rows_parsed"] = selected_non_date_rows
    frame.attrs["outside_snapshot_non_date_rows_parsed"] = 0
    frame.attrs["rows_at_or_after_2023_loaded"] = 0
    return frame


def prior_positive_quantile(
    values: pd.Series,
    *,
    lookback: int,
    minimum_positive: int,
    quantile: float,
) -> pd.Series:
    def positive_quantile(window: np.ndarray) -> float:
        positive = window[np.isfinite(window) & (window > 0.0)]
        if len(positive) < minimum_positive:
            return float("nan")
        return float(np.quantile(positive, quantile))

    return (
        values.shift(1)
        .rolling(lookback, min_periods=1)
        .apply(positive_quantile, raw=True)
    )


def prior_quantile(
    values: pd.Series,
    *,
    lookback: int,
    minimum: int,
    quantile: float,
) -> pd.Series:
    return (
        values.shift(1)
        .rolling(lookback, min_periods=minimum)
        .quantile(quantile)
    )


def build_signal_panel(
    insurance: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    panel = insurance.merge(
        prices,
        on="snapshot_time",
        how="left",
        validate="one_to_one",
    ).sort_values("snapshot_time", ignore_index=True)
    panel["fund_return"] = np.log(
        panel["wallet_balance_satoshi"]
        / panel["wallet_balance_satoshi"].shift(1)
    )
    panel["fund_loss"] = (-panel["fund_return"]).clip(lower=0.0)
    panel["pre_snapshot_return"] = np.log(
        panel["snapshot_price"] / panel["snapshot_price"].shift(1)
    )
    panel["fund_loss_threshold"] = prior_positive_quantile(
        panel["fund_loss"],
        lookback=cfg.fund_lookback_days,
        minimum_positive=cfg.minimum_prior_loss_days,
        quantile=cfg.fund_loss_quantile,
    )
    panel["price_move_threshold"] = prior_quantile(
        panel["pre_snapshot_return"].abs(),
        lookback=cfg.price_lookback_days,
        minimum=cfg.minimum_prior_price_days,
        quantile=cfg.price_move_quantile,
    )
    panel["decision_time"] = panel["snapshot_time"] + pd.Timedelta(
        days=cfg.source_embargo_days
    )
    panel["entry_time"] = panel["decision_time"] + pd.Timedelta(
        minutes=5 * cfg.latency_bars
    )
    panel["exit_time"] = panel["entry_time"] + pd.Timedelta(
        minutes=5 * cfg.hold_bars
    )
    side = -np.sign(panel["pre_snapshot_return"].fillna(0.0)).astype(np.int8)
    eligible = panel["entry_time"].ge(pd.Timestamp(cfg.eligibility_start)) & panel[
        "entry_time"
    ].lt(pd.Timestamp(cfg.selection_end_exclusive))
    thresholds_ready = panel[
        ["fund_loss_threshold", "price_move_threshold"]
    ].notna().all(axis=1)
    candidate = (
        eligible
        & thresholds_ready
        & panel["fund_return"].lt(0.0)
        & panel["fund_loss"].ge(panel["fund_loss_threshold"])
        & panel["pre_snapshot_return"].abs().ge(panel["price_move_threshold"])
        & side.ne(0)
    )
    panel["eligible"] = eligible
    panel["thresholds_ready"] = thresholds_ready
    panel["candidate"] = candidate
    panel["side"] = side.where(candidate, 0).astype(np.int8)
    return panel


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    entries = schedule["entry_time"]
    train = entries.lt(pd.Timestamp("2022-01-01"))
    test = entries.ge(pd.Timestamp("2022-01-01"))
    h1 = test & entries.dt.month.le(6)
    h2 = test & entries.dt.month.ge(7)
    counts = {
        "total_2020h2_2022": int(len(schedule)),
        "train_2020h2_2021": int(train.sum()),
        "train_2020h2": int((entries.dt.year.eq(2020) & train).sum()),
        "train_2021": int(entries.dt.year.eq(2021).sum()),
        "test_2022": int(test.sum()),
        "test_2022_h1": int(h1.sum()),
        "test_2022_h2": int(h2.sum()),
    }
    side_shares: dict[str, dict[str, float]] = {}
    side_checks: dict[str, bool] = {}
    for name, mask in {
        "all": pd.Series(True, index=schedule.index),
        "train": train,
        "test": test,
    }.items():
        selected = schedule.loc[mask]
        long_share = float(selected["side"].gt(0).mean()) if len(selected) else 0.0
        short_share = float(selected["side"].lt(0).mean()) if len(selected) else 0.0
        side_shares[name] = {"long": long_share, "short": short_share}
        side_checks[name] = min(long_share, short_share) >= cfg.minimum_side_share
    quarters = entries.dt.to_period("Q").astype(str)
    quarter_counts = {
        key: int(value)
        for key, value in quarters.value_counts().sort_index().items()
    }
    expected_quarters = [
        "2020Q3",
        "2020Q4",
        "2021Q1",
        "2021Q2",
        "2021Q3",
        "2021Q4",
        "2022Q1",
        "2022Q2",
        "2022Q3",
        "2022Q4",
    ]
    maximum_quarter_share = (
        max(quarter_counts.values()) / len(schedule) if len(schedule) else 1.0
    )
    checks = {
        "total": counts["total_2020h2_2022"] >= cfg.minimum_total,
        "train_total": counts["train_2020h2_2021"]
        >= cfg.minimum_train_2020h2_2021,
        "train_2020h2": counts["train_2020h2"] >= cfg.minimum_train_2020h2,
        "train_2021": counts["train_2021"] >= cfg.minimum_train_2021,
        "test_total": counts["test_2022"] >= cfg.minimum_test_2022,
        "test_h1": counts["test_2022_h1"] >= cfg.minimum_each_test_half,
        "test_h2": counts["test_2022_h2"] >= cfg.minimum_each_test_half,
        "each_eligible_quarter": all(
            quarter_counts.get(quarter, 0) >= cfg.minimum_each_eligible_quarter
            for quarter in expected_quarters
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
        "expected_quarters": expected_quarters,
        "maximum_quarter_share": float(maximum_quarter_share),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def event_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in schedule[
        ["snapshot_time", "decision_time", "entry_time", "exit_time", "side"]
    ].to_dict(orient="records"):
        records.append(
            {
                "snapshot_time": str(row["snapshot_time"]),
                "decision_time": str(row["decision_time"]),
                "entry_time": str(row["entry_time"]),
                "exit_time": str(row["exit_time"]),
                "side": int(row["side"]),
            }
        )
    return records


def event_clock_hash(
    events: list[dict[str, Any]],
    *,
    cfg: Config,
    protocol_hash: str,
    source_manifest_hash: str,
    source_sha256: str,
) -> str:
    return canonical_hash(
        {
            "policy_id": POLICY_ID,
            "events": events,
            "config": asdict(cfg),
            "protocol_hash": protocol_hash,
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha256,
        }
    )


def protocol(
    cfg: Config,
    insurance_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    insurance_output = (
        insurance_manifest.get("output", {}) if insurance_manifest else {}
    )
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "source": {
            "insurance_endpoint": "https://www.bitmex.com/api/v1/insurance",
            "insurance_official_docs": (
                "https://docs.bitmex.com/api-explorer/get-insurances"
            ),
            "insurance_currency": "XBt",
            "insurance_interval": ["2018-01-01", "2023-01-01"],
            "insurance_source": str(INSURANCE_SOURCE),
            "insurance_source_sha256": insurance_output.get(
                "sha256", "pending_outcome_blind_download"
            ),
            "insurance_manifest": str(INSURANCE_MANIFEST),
            "insurance_manifest_hash": (
                insurance_manifest.get("manifest_hash")
                if insurance_manifest
                else "pending_outcome_blind_download"
            ),
            "binance": str(BINANCE_SOURCE),
            "binance_sha256": BINANCE_SHA256,
            "selection_end_exclusive": cfg.selection_end_exclusive,
            "funding_loaded": False,
            "post_decision_execution_or_outcome_bars_loaded": False,
            "raw_insurance_history_committed": False,
        },
        "clock": {
            "timezone": "UTC",
            "insurance_snapshot": "daily 12:00 UTC",
            "price_observation": (
                "completed Binance 11:55-12:00 UTC five-minute close"
            ),
            "decision": (
                f"snapshot timestamp + {cfg.source_embargo_days} full calendar day"
            ),
            "entry": f"{cfg.latency_bars} complete five-minute latency bar later",
            "hold_bars": cfg.hold_bars,
            "live_fail_closed": (
                "expected source timestamp must actually be observed before decision"
            ),
        },
        "feature": {
            "fund_return": "log current XBt wallet balance / prior daily balance",
            "fund_loss": "max(-fund return, 0)",
            "fund_loss_reference": (
                f"q{cfg.fund_loss_quantile} of positive losses among the last "
                f"{cfg.fund_lookback_days} earlier snapshots; require "
                f"{cfg.minimum_prior_loss_days}; current row excluded"
            ),
            "pre_snapshot_return": (
                "log completed 12:00 price / prior completed 12:00 price"
            ),
            "price_reference": (
                f"q{cfg.price_move_quantile} absolute move among the last "
                f"{cfg.price_lookback_days} earlier observations; require "
                f"{cfg.minimum_prior_price_days}; current row excluded"
            ),
            "mandatory": (
                "negative fund return at least its prior loss threshold and absolute "
                "pre-snapshot BTC return at least its prior move threshold"
            ),
            "side": "opposite the completed pre-snapshot BTC move",
            "threshold_grid": False,
        },
        "support_gate": {
            "eligibility_start": cfg.eligibility_start,
            "minimum_total": cfg.minimum_total,
            "minimum_train_2020h2_2021": cfg.minimum_train_2020h2_2021,
            "minimum_train_2020h2": cfg.minimum_train_2020h2,
            "minimum_train_2021": cfg.minimum_train_2021,
            "minimum_test_2022": cfg.minimum_test_2022,
            "minimum_each_test_half": cfg.minimum_each_test_half,
            "minimum_each_eligible_quarter": cfg.minimum_each_eligible_quarter,
            "minimum_side_share_all_train_test": cfg.minimum_side_share,
            "maximum_quarter_share": cfg.maximum_quarter_share,
            "failure_action": (
                "reject before post-decision outcomes; no threshold, embargo, side, "
                "or hold repair"
            ),
        },
        "later_evaluation_contract": {
            "train": [cfg.eligibility_start, "2022-01-01"],
            "test": ["2022-01-01", cfg.selection_end_exclusive],
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
                "same price rule without the insurance-loss gate",
                "seven-day-stale insurance loss gate",
                "within-calendar-year permutation of insurance loss magnitudes",
            ],
        },
        "frozen_artifacts": {
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "source_downloader": str(SOURCE_DOWNLOADER),
            "source_downloader_sha256": SOURCE_DOWNLOADER_SHA256,
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(
                PREREGISTRATION_DOCUMENT
            ),
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(
                PREREGISTRATION_SOURCE
            ),
        },
        "research_history_boundary": (
            "candidate-level freeze only; unrelated repository research has seen "
            "market history, but no complete IFAR source incidence or post-decision "
            "outcome was used to select this singleton"
        ),
    }


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _validate_config(cfg)
    insurance, insurance_manifest = load_insurance_source()
    prices = read_daily_snapshot_prices(BINANCE_SOURCE)
    panel = build_signal_panel(insurance, prices, cfg)
    schedule = panel.loc[panel["candidate"]].reset_index(drop=True)
    summary = support_summary(schedule, cfg)
    events = event_records(schedule)
    protocol_payload = protocol(cfg, insurance_manifest)
    protocol_hash = canonical_hash(protocol_payload)
    source_sha = str(insurance_manifest["output"]["sha256"])
    source_manifest_hash = str(insurance_manifest["manifest_hash"])
    clock_hash = event_clock_hash(
        events,
        cfg=cfg,
        protocol_hash=protocol_hash,
        source_manifest_hash=source_manifest_hash,
        source_sha256=source_sha,
    )
    core = {
        "protocol_version": "bitmex_insurance_fund_absorption_support_v1",
        "protocol": protocol_payload,
        "protocol_hash": protocol_hash,
        "outcomes_opened": False,
        "source_loaded": True,
        "source_audit": {
            "insurance_rows_parsed": int(len(insurance)),
            "market_snapshot_rows_parsed": int(len(prices)),
            "market_outside_snapshot_non_date_rows_parsed": int(
                prices.attrs["outside_snapshot_non_date_rows_parsed"]
            ),
            "funding_rows_loaded": 0,
            "post_decision_execution_or_outcome_rows_loaded": 0,
            "rows_at_or_after_2023_loaded": 0,
        },
        "window_support": {
            "insurance_days": int(len(panel)),
            "market_days": int(panel["snapshot_price"].notna().sum()),
            "eligible_days": int(panel["eligible"].sum()),
            "threshold_ready_eligible_days": int(
                (panel["eligible"] & panel["thresholds_ready"]).sum()
            ),
            "candidate_days": int(len(schedule)),
        },
        "support_gate": summary,
        "event_clock_hash": clock_hash,
        "event_clock_written": bool(summary["passed"]),
        "sealed": [
            "all post-decision 2020-2022 outcomes",
            "2023",
            "2024",
            "2025",
            "2026_ytd",
        ],
        "failure_action": (
            None
            if summary["passed"]
            else (
                "reject before outcomes; no threshold, embargo, side, or hold repair"
            )
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
            "protocol_version": (
                "bitmex_insurance_fund_absorption_event_clock_v1"
            ),
            "policy_id": POLICY_ID,
            "outcomes_opened": False,
            "support_result_hash": result["result_hash"],
            "protocol_hash": protocol_hash,
            "config": asdict(cfg),
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha,
            "event_clock_hash": clock_hash,
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
    else:
        Path(cfg.event_clock_output).unlink(missing_ok=True)
    return result, clock


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-output", default=Config.support_output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    result, _ = run_support(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
