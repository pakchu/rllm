"""Freeze and run outcome-blind source support for DEHR-72.

This stage reads only the pre-2023 Deribit BTC option-delivery aggregate.  It
never reads Binance bars, funding, post-expiry returns, PnL, or a 2023+ source
row.  The original preregistration existed before source incidence; this
successor discloses the later source-only event-clock diagnostic explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.download_deribit_btc_option_deliveries import (
    Config as SourceConfig,
)


POLICY_ID = "DEHR-72"
SOURCE_DATA = Path(
    "data/deribit_btc_option_delivery_release_2019_2022.csv.gz"
)
SOURCE_MANIFEST = Path(
    "results/deribit_btc_option_delivery_source_manifest_2026-07-20.json"
)
SOURCE_DECISION = Path(
    "docs/deribit-expiry-hedge-release-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "459247767cd26b33fc9df1e2ee5819fda65826455151d943e0e432f1bb414c00"
)
SOURCE_DOWNLOADER = Path(
    "training/download_deribit_btc_option_deliveries.py"
)
SOURCE_DOWNLOADER_SHA256 = (
    "1e698db869ef263b692a950a3ecc4f4fafb834dd99db8476fd4da11bc1852cda"
)
EVENT_CLOCK_CORRECTION = Path(
    "docs/deribit-expiry-event-clock-operational-correction-2026-07-20.md"
)
EVENT_CLOCK_CORRECTION_SHA256 = (
    "3a7068c44d38b35e84a461186c22825d0961269bd2feb334785f6bf3001467a6"
)
ORIGINAL_PREREGISTRATION_ARTIFACT_HASH = (
    "8e0608227cfd85560d284de4c565ba1eb8741f46a54961c710a299a516386891"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/deribit-expiry-hedge-release-support-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_deribit_expiry_hedge_release_support.py"
)

SOURCE_COLUMNS = [
    "expiry_time",
    "delivery_event_time",
    "source_observation_earliest",
    "index_price",
    "option_count",
    "call_count",
    "put_count",
    "itm_call_count",
    "itm_put_count",
    "total_position",
    "call_position",
    "put_position",
    "itm_call_position",
    "itm_put_position",
    "otm_position",
    "atm_position",
    "net_release_position",
    "absolute_release_position",
    "release_side",
    "largest_instrument_share",
    "delivery_delay_seconds",
    "maximum_event_row_span_seconds",
]


@dataclass(frozen=True)
class Config:
    preregistration_output: str = (
        "results/deribit_expiry_hedge_release_support_preregistration_"
        "2026-07-20.json"
    )
    support_output: str = (
        "results/deribit_expiry_hedge_release_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/deribit_expiry_hedge_release_event_clock_2026-07-20.json"
    )
    reference_lookback_days: int = 365
    minimum_prior_expiries: int = 20
    total_position_quantile: float = 0.50
    release_share_quantile: float = 0.70
    source_observation_latency_minutes: int = 65
    entry_latency_bars: int = 1
    hold_bars: int = 72
    eligibility_start: str = "2020-07-01"
    selection_end_exclusive: str = "2023-01-01"
    source_first_at_or_before: str = "2019-03-01"
    source_last_at_or_after: str = "2022-12-25"
    minimum_source_eligible_expiries: int = 500
    minimum_source_expiries_per_month: int = 10
    maximum_source_gap_days: float = 14.0
    minimum_total: int = 120
    minimum_train_2020h2_2021: int = 70
    minimum_train_2020h2: int = 15
    minimum_train_2021: int = 45
    minimum_test_2022: int = 50
    minimum_each_test_half: int = 20
    minimum_each_eligible_quarter: int = 8
    minimum_active_months: int = 27
    minimum_side_share: float = 0.25
    maximum_month_share: float = 0.15


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(cfg: Config) -> None:
    expected = Config(
        preregistration_output=cfg.preregistration_output,
        support_output=cfg.support_output,
        event_clock_output=cfg.event_clock_output,
    )
    if cfg != expected:
        raise ValueError("DEHR-72 signal and support configuration is frozen")
    anchors = {
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        SOURCE_DOWNLOADER: SOURCE_DOWNLOADER_SHA256,
        EVENT_CLOCK_CORRECTION: EVENT_CLOCK_CORRECTION_SHA256,
    }
    for path, expected_sha in anchors.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"DEHR-72 frozen source anchor mismatch: {path}")


def protocol(cfg: Config) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "source": {
            "endpoint": (
                "https://www.deribit.com/api/v2/public/"
                "get_last_settlements_by_currency"
            ),
            "official_docs": (
                "https://docs.deribit.com/api-reference/market-data/"
                "public-get_last_settlements_by_currency"
            ),
            "currency": "BTC",
            "settlement_type": "delivery",
            "interval": ["2019-01-01", "2023-01-01"],
            "aggregate": str(SOURCE_DATA),
            "aggregate_sha256": "pending_outcome_blind_download",
            "manifest": str(SOURCE_MANIFEST),
            "manifest_hash": "pending_outcome_blind_download",
            "fields_used_for_support": [
                "expiry_time",
                "delivery_event_time",
                "source_observation_earliest",
                "total_position",
                "absolute_release_position",
                "net_release_position",
                "release_side",
            ],
            "fields_preserved_for_frozen_controls": [
                "itm_call_count",
                "itm_put_count",
                "itm_call_position",
                "itm_put_position",
            ],
            "fields_explicitly_unused": [
                "post_expiry_btc_price",
                "return",
                "pnl",
                "funding",
                "profit_loss",
                "session_profit_loss",
            ],
            "raw_responses_persisted": False,
            "rows_at_or_after_2023_loaded": False,
        },
        "clock": {
            "timezone": "UTC",
            "event": "Deribit option delivery, normally 08:00 UTC",
            "publication_sla_known": False,
            "historical_observation": (
                "reported delivery event timestamp + "
                f"{cfg.source_observation_latency_minutes} minutes"
            ),
            "live_observation": (
                "two identical canonical delivery sets observed five minutes "
                "apart, never before event + 60 minutes"
            ),
            "entry": (
                f"{cfg.entry_latency_bars} additional completed five-minute "
                "latency bar after source observation"
            ),
            "hold_bars": cfg.hold_bars,
            "hold_minutes": cfg.hold_bars * 5,
            "late_live_source": "delay or cancel; never backdate",
        },
        "feature": {
            "release_share": (
                "absolute net ITM terminal-delta release / total reported "
                "option position at the expiry"
            ),
            "total_position_reference": (
                f"q{cfg.total_position_quantile:.2f} from strictly prior "
                f"expiries in {cfg.reference_lookback_days} calendar days; "
                f"require {cfg.minimum_prior_expiries}"
            ),
            "release_share_reference": (
                f"q{cfg.release_share_quantile:.2f} from the same strictly "
                "prior calendar window"
            ),
            "candidate": (
                "nonzero release side, total position at or above its prior "
                "reference, and release share at or above its prior reference"
            ),
            "side": (
                "long when ITM put position exceeds ITM call position; short "
                "when ITM call position exceeds ITM put position"
            ),
            "threshold_grid": False,
            "repair_after_incidence": False,
        },
        "support_gate": {
            "eligibility": [
                cfg.eligibility_start,
                cfg.selection_end_exclusive,
            ],
            "source_first_at_or_before": cfg.source_first_at_or_before,
            "source_last_at_or_after": cfg.source_last_at_or_after,
            "minimum_source_eligible_expiries": (
                cfg.minimum_source_eligible_expiries
            ),
            "minimum_source_expiries_per_month": (
                cfg.minimum_source_expiries_per_month
            ),
            "maximum_source_gap_days": cfg.maximum_source_gap_days,
            "minimum_total": cfg.minimum_total,
            "minimum_train_2020h2_2021": cfg.minimum_train_2020h2_2021,
            "minimum_train_2020h2": cfg.minimum_train_2020h2,
            "minimum_train_2021": cfg.minimum_train_2021,
            "minimum_test_2022": cfg.minimum_test_2022,
            "minimum_each_test_half": cfg.minimum_each_test_half,
            "minimum_each_eligible_quarter": (
                cfg.minimum_each_eligible_quarter
            ),
            "minimum_active_months": cfg.minimum_active_months,
            "minimum_side_share_all_train_test": cfg.minimum_side_share,
            "maximum_month_share": cfg.maximum_month_share,
            "failure_action": (
                "reject before market outcomes; no threshold, latency, side, "
                "hold, or calendar-support repair"
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
                "interior exact-time symmetric; exact entry/exit credits "
                "dropped and debits retained"
            ),
            "cagr": "full split wall clock including warmup and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, exact funding, every held "
                "5m path, virtual adverse exit fee, and actual exit"
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
                (
                    "expiry-time-only random side on every eligible expiry "
                    "clock"
                ),
                "exact direction flip",
                (
                    "equal-position strike ablation: sign of ITM put count "
                    "minus ITM call count"
                ),
                (
                    "call/put-type ablation: exact candidate clocks with fixed "
                    "alternating side independent of option type"
                ),
                (
                    "deterministic random side on the exact candidate clocks, "
                    "seeded only by expiry timestamp"
                ),
                "release-share gate ablation",
                "total-position gate ablation",
                "one additional five-minute entry delay",
            ],
            "candidate_mean_gross_minus_best_nondirectional_control_bp_min": 5.0,
        },
        "frozen_artifacts": {
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "source_downloader": str(SOURCE_DOWNLOADER),
            "source_downloader_sha256": SOURCE_DOWNLOADER_SHA256,
            "event_clock_correction": str(EVENT_CLOCK_CORRECTION),
            "event_clock_correction_sha256": (
                EVENT_CLOCK_CORRECTION_SHA256
            ),
            "original_preregistration_artifact_hash": (
                ORIGINAL_PREREGISTRATION_ARTIFACT_HASH
            ),
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
            "candidate-level freeze only; unrelated repository research has "
            "seen BTC history. A source-only clock diagnostic opened page, row, "
            "year, and timestamp-offset statistics after the original freeze; "
            "DEHR candidate incidence and all matching post-entry outcomes "
            "remain unopened in this successor artifact"
        ),
    }


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    protocol_payload = protocol(cfg)
    core = {
        "protocol_version": "deribit_expiry_hedge_release_preregistration_v2",
        "protocol": protocol_payload,
        "protocol_hash": canonical_hash(protocol_payload),
        "outcomes_opened": False,
        "source_incidence_opened": True,
        "source_clock_diagnostic_opened": True,
        "candidate_incidence_opened": False,
        "supersedes_artifact_hash": ORIGINAL_PREREGISTRATION_ARTIFACT_HASH,
    }
    artifact = {**core, "artifact_hash": canonical_hash(core)}
    path = Path(cfg.preregistration_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    return artifact


def load_preregistration(cfg: Config) -> dict[str, Any]:
    path = Path(cfg.preregistration_output)
    artifact = json.loads(path.read_text())
    core = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    if canonical_hash(core) != artifact.get("artifact_hash"):
        raise RuntimeError("DEHR-72 preregistration artifact hash mismatch")
    if artifact.get("protocol") != protocol(cfg):
        raise RuntimeError("DEHR-72 preregistration protocol drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("DEHR-72 preregistration opened outcomes")
    if artifact.get("source_incidence_opened") is not True:
        raise RuntimeError("DEHR-72 preregistration hides source diagnostics")
    if artifact.get("source_clock_diagnostic_opened") is not True:
        raise RuntimeError("DEHR-72 preregistration hides the clock diagnostic")
    if artifact.get("candidate_incidence_opened") is not False:
        raise RuntimeError("DEHR-72 preregistration opened candidate incidence")
    if artifact.get("supersedes_artifact_hash") != (
        ORIGINAL_PREREGISTRATION_ARTIFACT_HASH
    ):
        raise RuntimeError("DEHR-72 preregistration predecessor mismatch")
    return artifact


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def validate_source_manifest_metadata(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest.get("protocol_version") != "deribit_btc_option_delivery_source_v3":
        raise RuntimeError("DEHR-72 source manifest version mismatch")
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("DEHR-72 source manifest hash mismatch")
    if manifest.get("config") != asdict(SourceConfig()):
        raise RuntimeError("DEHR-72 source request contract mismatch")
    aggregate = manifest.get("aggregate")
    audit = manifest.get("source_audit")
    if not isinstance(aggregate, dict) or not isinstance(audit, dict):
        raise RuntimeError("DEHR-72 source manifest sections are missing")
    if aggregate.get("path") != str(SOURCE_DATA):
        raise RuntimeError("DEHR-72 source aggregate path mismatch")
    rows = aggregate.get("rows")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or aggregate.get("columns") != SOURCE_COLUMNS
    ):
        raise RuntimeError("DEHR-72 source aggregate metadata mismatch")
    if audit.get("start") != "2019-01-01" or audit.get(
        "end_exclusive"
    ) != "2023-01-01":
        raise RuntimeError("DEHR-72 source audit interval mismatch")
    if audit.get("crossed_start_boundary") is not True:
        raise RuntimeError("DEHR-72 source did not prove lower-bound coverage")
    if audit.get("expiry_events") != rows:
        raise RuntimeError("DEHR-72 source expiry count mismatch")
    try:
        first = pd.Timestamp(audit["first_expiry"])
        last = pd.Timestamp(audit["last_expiry"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DEHR-72 source audit expiry boundary is invalid") from exc
    if first.tzinfo is None or last.tzinfo is None:
        raise RuntimeError("DEHR-72 source audit expiry boundary lacks timezone")
    first = first.tz_convert("UTC")
    last = last.tz_convert("UTC")
    if (
        first < pd.Timestamp("2019-01-01", tz="UTC")
        or last >= pd.Timestamp("2023-01-01", tz="UTC")
        or first > last
    ):
        raise RuntimeError("DEHR-72 source audit opened a forbidden year")
    allowed_years = {"2019", "2020", "2021", "2022"}
    for key in ["rows_by_year", "expiries_by_year"]:
        values = audit.get(key)
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f"DEHR-72 source audit {key} is missing")
        if not set(values).issubset(allowed_years):
            raise RuntimeError(f"DEHR-72 source audit {key} opened a forbidden year")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in values.values()
        ):
            raise RuntimeError(f"DEHR-72 source audit {key} is invalid")
    if sum(audit["expiries_by_year"].values()) != rows:
        raise RuntimeError("DEHR-72 source yearly expiry counts do not reconcile")
    availability = manifest.get("causal_availability")
    if availability != {
        "deribit_publication_sla_known": False,
        "source_observation_rule": (
            "delivery_event_time + 65 minutes after two identical canonical "
            "delivery sets observed five minutes apart"
        ),
        "source_observation_latency_seconds": 3900,
        "earliest_next_5m_entry_latency_seconds": 4200,
    }:
        raise RuntimeError("DEHR-72 source causal availability mismatch")
    return aggregate, audit


def validate_source_frame(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if list(frame.columns) != SOURCE_COLUMNS:
        raise RuntimeError("DEHR-72 source aggregate columns mismatch")
    checked = frame.copy()
    for column in [
        "expiry_time",
        "delivery_event_time",
        "source_observation_earliest",
    ]:
        checked[column] = pd.to_datetime(checked[column], utc=True, errors="raise")
    numeric_columns = [column for column in SOURCE_COLUMNS if column not in {
        "expiry_time", "delivery_event_time", "source_observation_earliest"
    }]
    for column in numeric_columns:
        checked[column] = pd.to_numeric(checked[column], errors="raise")
        if not np.isfinite(checked[column].to_numpy(float)).all():
            raise RuntimeError(f"DEHR-72 source {column} is not finite")
    if checked.empty or checked["expiry_time"].duplicated().any():
        raise RuntimeError("DEHR-72 source expiry clock is empty or duplicated")
    if not checked["expiry_time"].is_monotonic_increasing:
        raise RuntimeError("DEHR-72 source expiry clock is not chronological")
    start = pd.Timestamp("2019-01-01", tz="UTC")
    end = pd.Timestamp(cfg.selection_end_exclusive, tz="UTC")
    if checked["expiry_time"].min() < start or checked["expiry_time"].max() >= end:
        raise RuntimeError("DEHR-72 source escaped the frozen pre-2023 interval")
    expected_observation = checked["delivery_event_time"] + pd.Timedelta(
        minutes=cfg.source_observation_latency_minutes
    )
    if not checked["source_observation_earliest"].equals(expected_observation):
        raise RuntimeError("DEHR-72 source observation clock mismatch")
    release_share = (
        checked["absolute_release_position"] / checked["total_position"]
    )
    if checked["total_position"].le(0.0).any() or not release_share.between(
        0.0, 1.0
    ).all():
        raise RuntimeError("DEHR-72 source release magnitude is invalid")
    count_columns = [
        "option_count",
        "call_count",
        "put_count",
        "itm_call_count",
        "itm_put_count",
    ]
    for column in count_columns:
        values = checked[column].to_numpy(float)
        if (values < 0.0).any() or not np.equal(values, np.floor(values)).all():
            raise RuntimeError(f"DEHR-72 source {column} is not a count")
    if not np.array_equal(
        checked["option_count"].to_numpy(),
        (checked["call_count"] + checked["put_count"]).to_numpy(),
    ):
        raise RuntimeError("DEHR-72 source option count decomposition mismatch")
    if (
        checked["itm_call_count"].gt(checked["call_count"]).any()
        or checked["itm_put_count"].gt(checked["put_count"]).any()
    ):
        raise RuntimeError("DEHR-72 source ITM count exceeds type count")
    expiry = checked["expiry_time"].dt
    if not (
        expiry.hour.eq(8)
        & expiry.minute.eq(0)
        & expiry.second.eq(0)
        & expiry.microsecond.eq(0)
    ).all():
        raise RuntimeError("DEHR-72 source expiry is not exactly 08:00 UTC")
    event = checked["delivery_event_time"]
    if (
        event.lt(checked["expiry_time"]).any()
        or not event.dt.normalize().equals(checked["expiry_time"].dt.normalize())
    ):
        raise RuntimeError("DEHR-72 source delivery event clock is invalid")
    expected_delay = (event - checked["expiry_time"]).dt.total_seconds()
    if not np.allclose(
        checked["delivery_delay_seconds"],
        expected_delay,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("DEHR-72 source delivery delay mismatch")
    maximum_span = SourceConfig.maximum_event_row_span_seconds
    if not checked["maximum_event_row_span_seconds"].between(
        0.0, maximum_span
    ).all():
        raise RuntimeError("DEHR-72 source event row span is invalid")
    if not checked["largest_instrument_share"].between(0.0, 1.0).all():
        raise RuntimeError("DEHR-72 source instrument concentration is invalid")
    if not np.allclose(
        checked["call_position"] + checked["put_position"],
        checked["total_position"],
        rtol=1e-10,
        atol=1e-8,
    ):
        raise RuntimeError("DEHR-72 source position decomposition mismatch")
    if (
        checked["itm_call_position"].gt(checked["call_position"] + 1e-8).any()
        or checked["itm_put_position"].gt(checked["put_position"] + 1e-8).any()
    ):
        raise RuntimeError("DEHR-72 source ITM position exceeds type position")
    if not np.allclose(
        checked["itm_put_position"] - checked["itm_call_position"],
        checked["net_release_position"],
        rtol=1e-10,
        atol=1e-8,
    ):
        raise RuntimeError("DEHR-72 source net release decomposition mismatch")
    expected_absolute = checked["net_release_position"].abs()
    if not np.allclose(
        checked["absolute_release_position"], expected_absolute, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("DEHR-72 source absolute release mismatch")
    expected_side = np.sign(checked["net_release_position"]).astype(np.int8)
    if not np.array_equal(checked["release_side"].to_numpy(), expected_side):
        raise RuntimeError("DEHR-72 source release side mismatch")
    if not checked["release_side"].isin([-1, 0, 1]).all():
        raise RuntimeError("DEHR-72 source release side is outside {-1,0,1}")
    return checked


def load_delivery_source(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    aggregate, audit = validate_source_manifest_metadata(manifest)
    if sha256_file(SOURCE_DATA) != aggregate.get("sha256"):
        raise RuntimeError("DEHR-72 source aggregate hash mismatch")
    boundary = manifest.get("outcome_boundary", {})
    if boundary != {
        "binance_market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "post_delivery_return_or_pnl_loaded": False,
        "raw_deribit_rows_persisted": False,
    }:
        raise RuntimeError("DEHR-72 source outcome boundary mismatch")
    frame = validate_source_frame(pd.read_csv(SOURCE_DATA), cfg)
    if (
        aggregate.get("rows") != len(frame)
        or aggregate.get("columns") != SOURCE_COLUMNS
    ):
        raise RuntimeError("DEHR-72 source aggregate metadata mismatch")
    return frame, manifest


def prior_calendar_quantile(
    frame: pd.DataFrame,
    column: str,
    *,
    lookback_days: int,
    minimum: int,
    quantile: float,
) -> pd.Series:
    indexed = pd.Series(
        frame[column].to_numpy(float),
        index=pd.DatetimeIndex(frame["expiry_time"]),
    )
    result = indexed.rolling(
        f"{lookback_days}D", closed="left", min_periods=minimum
    ).quantile(quantile)
    return pd.Series(result.to_numpy(), index=frame.index)


def build_signal_panel(source: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    panel = source.sort_values("expiry_time", ignore_index=True).copy()
    panel["release_share"] = (
        panel["absolute_release_position"] / panel["total_position"]
    )
    panel["total_position_threshold"] = prior_calendar_quantile(
        panel,
        "total_position",
        lookback_days=cfg.reference_lookback_days,
        minimum=cfg.minimum_prior_expiries,
        quantile=cfg.total_position_quantile,
    )
    panel["release_share_threshold"] = prior_calendar_quantile(
        panel,
        "release_share",
        lookback_days=cfg.reference_lookback_days,
        minimum=cfg.minimum_prior_expiries,
        quantile=cfg.release_share_quantile,
    )
    panel["entry_time"] = panel["source_observation_earliest"] + pd.Timedelta(
        minutes=5 * cfg.entry_latency_bars
    )
    panel["exit_time"] = panel["entry_time"] + pd.Timedelta(
        minutes=5 * cfg.hold_bars
    )
    panel["thresholds_ready"] = panel[
        ["total_position_threshold", "release_share_threshold"]
    ].notna().all(axis=1)
    panel["eligible"] = panel["entry_time"].ge(
        pd.Timestamp(cfg.eligibility_start, tz="UTC")
    ) & panel["entry_time"].lt(
        pd.Timestamp(cfg.selection_end_exclusive, tz="UTC")
    )
    panel["candidate"] = (
        panel["eligible"]
        & panel["thresholds_ready"]
        & panel["release_side"].ne(0)
        & panel["total_position"].ge(panel["total_position_threshold"])
        & panel["release_share"].ge(panel["release_share_threshold"])
    )
    panel["side"] = panel["release_side"].where(
        panel["candidate"], 0
    ).astype(np.int8)
    return panel


def support_summary(
    schedule: pd.DataFrame,
    source: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    entries = schedule["entry_time"]
    train = entries.lt(pd.Timestamp("2022-01-01", tz="UTC"))
    test = ~train
    test_h1 = test & entries.dt.month.le(6)
    test_h2 = test & entries.dt.month.ge(7)
    counts = {
        "total_2020h2_2022": int(len(schedule)),
        "train_2020h2_2021": int(train.sum()),
        "train_2020h2": int((entries.dt.year.eq(2020) & train).sum()),
        "train_2021": int(entries.dt.year.eq(2021).sum()),
        "test_2022": int(test.sum()),
        "test_2022_h1": int(test_h1.sum()),
        "test_2022_h2": int(test_h2.sum()),
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

    entry_quarters = entries.dt.tz_convert(None).dt.to_period("Q").astype(str)
    quarter_counts = {
        key: int(value)
        for key, value in entry_quarters.value_counts().sort_index().items()
    }
    expected_quarters = [
        str(period) for period in pd.period_range("2020Q3", "2022Q4", freq="Q")
    ]
    entry_months = entries.dt.tz_convert(None).dt.to_period("M").astype(str)
    month_counts = {
        key: int(value)
        for key, value in entry_months.value_counts().sort_index().items()
    }
    expected_months = [
        str(period) for period in pd.period_range("2020-07", "2022-12", freq="M")
    ]
    maximum_month_share = (
        max(month_counts.values()) / len(schedule) if len(schedule) else 1.0
    )

    source_eligible = source.loc[
        source["expiry_time"].ge(pd.Timestamp(cfg.eligibility_start, tz="UTC"))
        & source["expiry_time"].lt(
            pd.Timestamp(cfg.selection_end_exclusive, tz="UTC")
        )
    ]
    source_months = (
        source_eligible["expiry_time"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
    )
    source_month_counts = {
        key: int(value)
        for key, value in source_months.value_counts().sort_index().items()
    }
    maximum_source_gap_days = (
        float(source_eligible["expiry_time"].diff().dt.total_seconds().max() / 86400)
        if len(source_eligible) > 1
        else float("inf")
    )
    source_support = {
        "first_expiry": source["expiry_time"].min().isoformat(),
        "last_expiry": source["expiry_time"].max().isoformat(),
        "eligible_expiries": int(len(source_eligible)),
        "monthly_counts": source_month_counts,
        "maximum_gap_days": maximum_source_gap_days,
    }
    checks = {
        "source_first": source["expiry_time"].min().normalize()
        <= pd.Timestamp(cfg.source_first_at_or_before, tz="UTC"),
        "source_last": source["expiry_time"].max().normalize()
        >= pd.Timestamp(cfg.source_last_at_or_after, tz="UTC"),
        "source_total": len(source_eligible) >= cfg.minimum_source_eligible_expiries,
        "source_each_month": all(
            source_month_counts.get(month, 0)
            >= cfg.minimum_source_expiries_per_month
            for month in expected_months
        ),
        "source_maximum_gap": maximum_source_gap_days
        <= cfg.maximum_source_gap_days,
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
        "active_months": sum(month in month_counts for month in expected_months)
        >= cfg.minimum_active_months,
        "side_all": side_checks["all"],
        "side_train": side_checks["train"],
        "side_test": side_checks["test"],
        "month_concentration": maximum_month_share <= cfg.maximum_month_share,
    }
    return {
        "counts": counts,
        "side_shares": side_shares,
        "quarter_counts": quarter_counts,
        "expected_quarters": expected_quarters,
        "month_counts": month_counts,
        "expected_months": expected_months,
        "active_months": sum(month in month_counts for month in expected_months),
        "maximum_month_share": float(maximum_month_share),
        "source_support": source_support,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def event_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    columns = [
        "expiry_time",
        "delivery_event_time",
        "source_observation_earliest",
        "entry_time",
        "exit_time",
        "side",
    ]
    for row in schedule[columns].to_dict(orient="records"):
        records.append(
            {
                "expiry_time": row["expiry_time"].isoformat(),
                "delivery_event_time": row["delivery_event_time"].isoformat(),
                "source_observation_earliest": row[
                    "source_observation_earliest"
                ].isoformat(),
                "entry_time": row["entry_time"].isoformat(),
                "exit_time": row["exit_time"].isoformat(),
                "side": int(row["side"]),
            }
        )
    return records


def event_clock_hash(
    events: list[dict[str, Any]],
    *,
    cfg: Config,
    preregistration_hash: str,
    source_manifest_hash: str,
    source_sha256: str,
) -> str:
    return canonical_hash(
        {
            "policy_id": POLICY_ID,
            "events": events,
            "config": asdict(cfg),
            "preregistration_hash": preregistration_hash,
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha256,
        }
    )


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _validate_config(cfg)
    preregistration = load_preregistration(cfg)
    source, source_manifest = load_delivery_source(cfg)
    panel = build_signal_panel(source, cfg)
    schedule = panel.loc[panel["candidate"]].reset_index(drop=True)
    summary = support_summary(schedule, source, cfg)
    events = event_records(schedule)
    source_sha = str(source_manifest["aggregate"]["sha256"])
    source_manifest_hash = str(source_manifest["manifest_hash"])
    clock_hash = event_clock_hash(
        events,
        cfg=cfg,
        preregistration_hash=str(preregistration["artifact_hash"]),
        source_manifest_hash=source_manifest_hash,
        source_sha256=source_sha,
    )
    core = {
        "protocol_version": "deribit_expiry_hedge_release_support_v2",
        "policy_id": POLICY_ID,
        "outcomes_opened": False,
        "source_incidence_opened": True,
        "source_clock_diagnostic_previously_opened": True,
        "preregistration_hash": preregistration["artifact_hash"],
        "source_manifest_hash": source_manifest_hash,
        "source_sha256": source_sha,
        "source_audit": {
            "aggregate_rows_parsed": int(len(source)),
            "binance_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "post_expiry_outcome_rows_loaded": 0,
            "rows_at_or_after_2023_loaded": 0,
        },
        "window_support": {
            "eligible_expiries": int(panel["eligible"].sum()),
            "threshold_ready_eligible_expiries": int(
                (panel["eligible"] & panel["thresholds_ready"]).sum()
            ),
            "candidate_expiries": int(len(schedule)),
        },
        "support_gate": summary,
        "event_clock_hash": clock_hash,
        "event_clock_written": bool(summary["passed"]),
        "sealed": [
            "all post-entry 2020H2-2022 BTC paths",
            "2023",
            "2024",
            "2025",
            "2026_ytd",
        ],
        "failure_action": (
            None
            if summary["passed"]
            else "reject before outcomes; no source-support or signal repair"
        ),
    }
    result = {**core, "result_hash": canonical_hash(core)}
    result_path = Path(cfg.support_output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    clock: dict[str, Any] | None = None
    if summary["passed"]:
        clock_core = {
            "protocol_version": "deribit_expiry_hedge_release_event_clock_v2",
            "policy_id": POLICY_ID,
            "outcomes_opened": False,
            "support_result_hash": result["result_hash"],
            "preregistration_hash": preregistration["artifact_hash"],
            "config": asdict(cfg),
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha,
            "event_clock_hash": clock_hash,
            "events": events,
        }
        clock = {**clock_core, "manifest_hash": canonical_hash(clock_core)}
        clock_path = Path(cfg.event_clock_output)
        clock_path.parent.mkdir(parents=True, exist_ok=True)
        clock_path.write_text(json.dumps(clock, indent=2, ensure_ascii=False) + "\n")
    else:
        Path(cfg.event_clock_output).unlink(missing_ok=True)
    return result, clock


def parse_args() -> tuple[Config, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-support",
        action="store_true",
        help="open only frozen pre-2023 source incidence and run support",
    )
    args = parser.parse_args()
    return Config(), bool(args.run_support)


def main() -> None:
    cfg, run = parse_args()
    payload = run_support(cfg)[0] if run else write_preregistration(cfg)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
