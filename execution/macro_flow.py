"""Frozen macro-flow target adapter and live configuration contract."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from execution.approved_portfolio_signals import build_macro_targets

MACRO_FLOW_SOURCE_PATH = Path(
    "configs/shadow/macro_flow_regime_switch_candidate_2026-09-06.json"
)
MACRO_FLOW_SOURCE_SHA256 = (
    "5c398879eb090a97423b85e7d8f268d6ba2f199b709bb25d728d8b77a8e2fcdc"
)
APPROVED_PORTFOLIO_PATH = Path(
    "configs/approved/g9_macro1_dollar_short05_2026-09-07.json"
)
APPROVED_PORTFOLIO_SHA256 = (
    "44fa87328c3299249f7d956705fa53a036e7271eea41eadb46a930228445821c"
)
MACRO_FLOW_MIN_HISTORY_BARS = 18_000


class MacroFlowContractError(ValueError):
    """Raised when live macro-flow configuration drifts from approval."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise MacroFlowContractError(
            f"macro-flow {field} must be {expected!r}, got {actual!r}"
        )


def validate_macro_flow_runtime_config(
    config: Mapping[str, Any],
    *,
    configured_side: str | None = None,
) -> None:
    """Validate the exact approved target formula and runtime clock."""

    for field, expected in (
        ("runtime_ready", True),
        ("enabled", True),
        ("live_authorized", True),
        ("research_only", False),
        ("target_maintenance", True),
    ):
        _require_equal(field, config.get(field), expected)
    _require_equal(
        "policy_type",
        str(config.get("policy_type", "")).lower(),
        "macro_flow",
    )
    _require_equal("symbol", str(config.get("symbol", "")).upper(), "BTCUSDT")
    _require_equal("side", str(config.get("side", "")).upper(), "AUTO")
    if configured_side is not None:
        _require_equal("configured sleeve side", configured_side.upper(), "AUTO")
    for field, expected in (
        ("minimum_feature_history_bars", MACRO_FLOW_MIN_HISTORY_BARS),
        ("decision_interval_minutes", 60),
        ("execution_offset_minutes", 5),
        ("maintenance_interval_minutes", 60),
        ("source_refresh_hours", 24),
    ):
        try:
            actual = int(config.get(field))
        except (TypeError, ValueError):
            actual = config.get(field)
        _require_equal(field, actual, expected)
    _require_equal("dollar_component_weight", config.get("dollar_component_weight"), 0.75)
    _require_equal("switch_component_weight", config.get("switch_component_weight"), 0.25)
    _require_equal("target_min", config.get("target_min"), -1.0)
    _require_equal("target_max", config.get("target_max"), 1.0)

    provenance = config.get("research_provenance")
    if not isinstance(provenance, Mapping):
        raise MacroFlowContractError("macro-flow research_provenance is missing")
    for label, path, expected_hash in (
        ("source", MACRO_FLOW_SOURCE_PATH, MACRO_FLOW_SOURCE_SHA256),
        ("approval", APPROVED_PORTFOLIO_PATH, APPROVED_PORTFOLIO_SHA256),
    ):
        _require_equal(
            f"research_provenance.{label}_path",
            str(provenance.get(f"{label}_path", "")),
            str(path),
        )
        _require_equal(
            f"research_provenance.{label}_sha256",
            str(provenance.get(f"{label}_sha256", "")),
            expected_hash,
        )
        if not path.is_file():
            raise MacroFlowContractError(f"macro-flow {label} artifact is missing: {path}")
        _require_equal(f"{label} artifact sha256", _sha256(path), expected_hash)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def score_macro_flow(
    market: pd.DataFrame,
    decision_bar_date: Any,
) -> dict[str, Any]:
    """Return the latest hourly target on its due slot or as a restart catch-up.

    The approved source target becomes effective at HH:05 UTC.  A live process
    that was unavailable at that exact instant must restore the latest held
    target, rather than remaining flat until the following hour.  Replays use
    the source HH:05 timestamp as the stable signal identity while scheduling
    execution on the current five-minute cycle; the signed ledger therefore
    applies each hourly target at most once.
    """

    decision = _utc(decision_bar_date)
    execution = decision + pd.Timedelta(minutes=5)
    source_execution = execution.floor("1h") + pd.Timedelta(minutes=5)
    if source_execution > execution:
        source_execution -= pd.Timedelta(hours=1)
    source_decision = source_execution - pd.Timedelta(minutes=5)
    signal_id = f"macro_flow:{source_execution.isoformat()}"
    targets = build_macro_targets(market, asof=execution)
    if source_execution not in targets.index:
        return {
            "name": "macro_flow",
            "emit": True,
            "kind": "target",
            "active": False,
            "ready": False,
            "target_fraction": 0.0,
            "decision_time": decision.isoformat(),
            "execution_time": execution.isoformat(),
            "source_decision_time": source_decision.isoformat(),
            "source_execution_time": source_execution.isoformat(),
            "signal_id": signal_id,
            "reason": "completed_hour_target_unavailable",
        }
    target = float(targets.loc[source_execution])
    catch_up = source_execution != execution
    return {
        "name": "macro_flow",
        "emit": True,
        "kind": "target",
        "active": target != 0.0,
        "ready": True,
        "target_fraction": target,
        "side": "LONG" if target > 0 else "SHORT" if target < 0 else "FLAT",
        "decision_time": decision.isoformat(),
        "execution_time": execution.isoformat(),
        "source_decision_time": source_decision.isoformat(),
        "source_execution_time": source_execution.isoformat(),
        "signal_id": signal_id,
        "reason": "target_catch_up" if catch_up else "target_ready",
    }
