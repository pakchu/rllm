"""Sequential strict evaluator for the frozen FLCC-1 family.

Evaluator freeze opens no outcome source. Stage 1 physically parses only the
2020-2022 execution/funding window. The 2023 window can be opened only by a
hash-bound Stage-1 report that passes every frozen gate and selects exactly one
candidate. 2024+ remains sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd

from training import evaluate_fiat_quote_participation_rotation as strict_engine
from training import federal_liquidity_component_concordance_clock as source_clock
from training import preregister_federal_liquidity_component_concordance as prereg


PREREGISTRATION_COMMIT = "8da2a57ae48a213bf03c16066cb5a35b492c43b6"
STATIC_INPUT_SHA256 = {
    "training/preregister_federal_liquidity_component_concordance.py": (
        "16a5a4fb5986c71f0c258179c246ac5dcb16dc8dcde81e80784a1f31d913d2f4"
    ),
    "training/federal_liquidity_component_concordance_clock.py": (
        "0fd6473135a6be712fa50c358d8b689c53bc0de1038a33955881045cfe01bf1d"
    ),
    "docs/federal-liquidity-component-concordance-preregistration-2026-07-17.md": (
        "b3694b9b73bf5388d391a50eb786bec66f1002e27ade766fc1d3aaf976321025"
    ),
    "results/federal_liquidity_component_concordance_preregistration_2026-07-17.json": (
        "3410443f009e63d4068d4d44c42e148799fba515b1934b01ecc38d7208ac54ee"
    ),
    "results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz": (
        "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c"
    ),
    "training/evaluate_fiat_quote_participation_rotation.py": (
        "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23"
    ),
}

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
CLOCK_LEDGER = Path(source_clock.DEFAULT_OUTPUT)
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path(prereg.FUNDING_PATH)
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST)
FUNDING_MANIFEST = Path(prereg.FUNDING_MANIFEST)
EVALUATOR_SOURCE = Path("training/evaluate_federal_liquidity_component_concordance.py")
EVALUATOR_FREEZE = Path(
    "results/federal_liquidity_component_concordance_evaluator_freeze_2026-07-17.json"
)
STAGE1_OUTPUT = Path(
    "results/federal_liquidity_component_concordance_stage1_2020_2022_2026-07-17.json"
)
STAGE1_DOC = Path(
    "docs/federal-liquidity-component-concordance-stage1-2020-2022-2026-07-17.md"
)
STAGE2_OUTPUT = Path(
    "results/federal_liquidity_component_concordance_stage2_2023_2026-07-17.json"
)
STAGE2_DOC = Path(
    "docs/federal-liquidity-component-concordance-stage2-2023-2026-07-17.md"
)

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def _utc_timestamp(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("FLCC-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


STAGE1: TimeWindow = (_utc_timestamp("2020-01-01"), _utc_timestamp("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2020": (_utc_timestamp("2020-01-01"), _utc_timestamp("2021-01-01")),
    "2021": (_utc_timestamp("2021-01-01"), _utc_timestamp("2022-01-01")),
    "2022": (_utc_timestamp("2022-01-01"), _utc_timestamp("2023-01-01")),
}
STAGE2: TimeWindow = (_utc_timestamp("2023-01-01"), _utc_timestamp("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc_timestamp("2023-01-01"), _utc_timestamp("2023-07-01")),
    "2023_h2": (_utc_timestamp("2023-07-01"), _utc_timestamp("2024-01-01")),
}

MECHANISM_CONTROLS = ("net_only", "component_concordance_only")
FALSIFICATION_CONTROLS = ("direction_flip", "one_release_delay", "random_side")
ALL_CLOCK_NAMES = ("primary", *MECHANISM_CONTROLS, *FALSIFICATION_CONTROLS)
CANDIDATE_IDS = tuple(spec.candidate_id for spec in source_clock.CANDIDATE_SPECS)
STAGE1_PROTOCOL = "federal_liquidity_component_concordance_stage1_v1"
STAGE1_ID = "stage1_2020_2022"
STAGE1_OPENED_WINDOWS = [STAGE1_ID]
STAGE1_SEALED_WINDOWS = ["stage2_2023", "2024", "2025", "2026_ytd"]
STAGE1_ROW_KEYS = frozenset(
    {
        "candidate_id",
        "primary",
        "stress_10bp_per_side",
        "subperiods",
        "controls",
        "entry_distribution",
        "falsification_control_details",
        "control_overall_qualification",
        "gates",
        "qualified",
    }
)
BASE_GATE_NAMES = frozenset(
    {
        "absolute_return_positive",
        "cagr_to_strict_mdd_at_least_3",
        "strict_mdd_at_most_15pct",
        "weekly_cluster_signflip_p_within_limit",
        "minimum_trades",
        "minimum_each_side_trades",
        "mean_gross_underlying_at_least_35bp",
        "stress_cost_absolute_return_positive",
        "each_subperiod_absolute_return_positive",
        "each_subperiod_minimum_trades",
        "single_entry_month_share_within_limit",
    }
)
STAGE1_GATE_NAMES = frozenset(
    BASE_GATE_NAMES
    | {
        "primary_ratio_strictly_beats_mechanism_controls",
        "falsification_controls_do_not_overall_qualify",
    }
)

EvaluationConfig = strict_engine.EvaluationConfig
simulate_schedule = strict_engine.simulate_schedule
_parse_market_window = strict_engine._parse_market_window
_parse_funding_window = strict_engine._parse_funding_window


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_evaluation_contract(registration: dict[str, Any]) -> None:
    policy = registration["execution_policy"]
    cfg = asdict(EvaluationConfig())
    expected = {
        "leverage": policy["leverage"],
        "base_cost_notional_per_side": policy["base_cost_notional_per_side"],
        "stress_cost_notional_per_side": policy["stress_cost_notional_per_side"],
        "cluster_draws": policy["weekly_cluster_draws"],
        "cluster_seed": policy["weekly_cluster_seed"],
        "mdd_denominator_floor": policy["strict_mdd_denominator_floor"],
    }
    if cfg != expected:
        raise ValueError(f"FLCC-1 strict-engine contract changed: {cfg} != {expected}")


def _verify_static_inputs() -> dict[str, Any]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"FLCC-1 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration, verify_sources=False)
    if registration.get("support_passed") is not True:
        raise ValueError("FLCC-1 source-only support did not pass")
    if registration.get("outcomes_opened") is not False:
        raise ValueError("FLCC-1 preregistration opened outcomes")
    _verify_evaluation_contract(registration)
    return registration


def _schedule_frame(events: list[source_clock.Event]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": event.candidate_id,
                "clock_name": event.clock_name,
                "feature_release_date": event.feature_release_date,
                "signal_release_date": event.signal_release_date,
                "signal_time": event.signal_time,
                "signal_day": event.signal_time,
                "entry_time": event.entry_time,
                "exit_time": event.exit_time,
                "side": event.side,
            }
            for event in events
        ]
    )
    for column in ("signal_time", "signal_day", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = frame["side"].astype("int8")
    return frame.sort_values("entry_time").reset_index(drop=True)


def load_schedules() -> dict[str, dict[str, pd.DataFrame]]:
    registration = _verify_static_inputs()
    source_rows = source_clock.read_source()
    rebuilt = source_clock.build_all_events(source_rows)
    ledger = source_clock.read_event_ledger(CLOCK_LEDGER)
    if rebuilt != ledger:
        raise ValueError("FLCC-1 rebuilt source clock differs from frozen ledger")
    schedules: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in source_clock.CANDIDATE_SPECS:
        schedules[spec.candidate_id] = {}
        for clock_name in ALL_CLOCK_NAMES:
            events = [
                event
                for event in ledger
                if event.candidate_id == spec.candidate_id
                and event.clock_name == clock_name
            ]
            frame = _schedule_frame(events)
            if frame.empty:
                raise ValueError(
                    f"FLCC-1 empty clock: {spec.candidate_id}/{clock_name}"
                )
            schedules[spec.candidate_id][clock_name] = frame

        expected = registration["source_only_support"][spec.candidate_id]
        primary = schedules[spec.candidate_id]["primary"]
        if len(primary) != expected["full_source_primary_events"]:
            raise ValueError(f"FLCC-1 {spec.candidate_id} primary count changed")
    return schedules


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "clock_name": str(row["clock_name"]),
                "feature_release_date": str(row["feature_release_date"]),
                "signal_release_date": str(row["signal_release_date"]),
                "signal_time": _timestamp(row["signal_time"]).isoformat(),
                "signal_day": _timestamp(row["signal_day"]).isoformat(),
                "entry_time": _timestamp(row["entry_time"]).isoformat(),
                "exit_time": _timestamp(row["exit_time"]).isoformat(),
                "side": int(row["side"]),
            }
        )
    return _canonical_hash(rows)


def _window_schedule(frame: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    start, end = window
    return frame.loc[frame["entry_time"].ge(start) & frame["exit_time"].lt(end)].copy()


def _entry_distribution(frame: pd.DataFrame, window: TimeWindow) -> dict[str, Any]:
    selected = _window_schedule(frame, window)
    months = selected["entry_time"].dt.strftime("%Y-%m").value_counts()
    maximum = int(months.max()) if len(months) else 0
    return {
        "trades": int(len(selected)),
        "longs": int(selected["side"].eq(1).sum()),
        "shorts": int(selected["side"].eq(-1).sum()),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(selected) if len(selected) else 0.0,
    }


def _schedule_freeze_record(frame: pd.DataFrame) -> dict[str, Any]:
    entries = [_timestamp(value) for value in frame["entry_time"]]
    exits = [_timestamp(value) for value in frame["exit_time"]]
    signals = [_timestamp(value) for value in frame["signal_time"]]
    return {
        "events": int(len(frame)),
        "schedule_hash": _schedule_hash(frame),
        "first_entry": min(entries).isoformat(),
        "last_exit": max(exits).isoformat(),
        "entry_exactly_five_minutes_after_signal": all(
            entry - signal == pd.Timedelta(minutes=5)
            for entry, signal in zip(entries, signals)
        ),
        "hold_exactly_five_days": all(
            exit_time - entry == pd.Timedelta(days=5)
            for entry, exit_time in zip(entries, exits)
        ),
        "globally_nonoverlapping": all(
            entry >= prior_exit for entry, prior_exit in zip(entries[1:], exits[:-1])
        ),
        "pre_2024_only": all(
            exit_time < _utc_timestamp("2024-01-01") for exit_time in exits
        ),
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    registration = _verify_static_inputs()
    schedules = load_schedules()
    records = {
        candidate_id: {
            clock_name: _schedule_freeze_record(frame)
            for clock_name, frame in candidate_schedules.items()
        }
        for candidate_id, candidate_schedules in schedules.items()
    }
    invariant_checks = {
        f"{candidate_id}/{clock_name}/{check}": bool(value)
        for candidate_id, clocks in records.items()
        for clock_name, record in clocks.items()
        for check, value in record.items()
        if isinstance(value, bool)
    }
    if not all(invariant_checks.values()):
        raise ValueError("FLCC-1 evaluator schedule invariant failed")
    core = {
        "protocol_version": "federal_liquidity_component_concordance_evaluator_v1",
        "as_of_date": "2026-07-17",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_fiat_quote_participation_rotation.py"
        ],
        "static_inputs": STATIC_INPUT_SHA256,
        "schedules": records,
        "invariant_checks": invariant_checks,
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": False,
        "outcomes_opened": False,
        "stage1_physical_window": [value.isoformat() for value in STAGE1],
        "stage2_physical_window": [value.isoformat() for value in STAGE2],
        "stage2_gate": (
            "requires a hash-valid passing Stage1 report selecting exactly one "
            "candidate from the frozen four-member family"
        ),
        "sealed": ["2023_until_stage1_pass", "2024", "2025", "2026_ytd"],
    }
    payload = _seal(core)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("FLCC-1 evaluator freeze hash mismatch")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FLCC-1 evaluator changed after freeze")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("FLCC-1 evaluator freeze opened outcomes")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("FLCC-1 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("FLCC-1 evaluator freeze parsed funding")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("FLCC-1 evaluator freeze simulated an outcome")
    schedules = load_schedules()
    for candidate_id, candidate_schedules in schedules.items():
        for clock_name, frame in candidate_schedules.items():
            expected = payload["schedules"][candidate_id][clock_name]
            if expected["schedule_hash"] != _schedule_hash(frame):
                raise ValueError(
                    f"FLCC-1 {candidate_id}/{clock_name} schedule changed after freeze"
                )
    return payload


def _verify_execution_manifests() -> None:
    registration = _load_json(PREREGISTRATION)
    source = registration["source_contract"]
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("FLCC-1 market manifest no longer binds the frozen source")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != source["funding_sha256"]:
        raise ValueError("FLCC-1 funding manifest no longer binds the frozen source")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("FLCC-1 funding source manifest opened an outcome")


def load_execution_window(
    window: TimeWindow,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _verify_execution_manifests()
    start, end = window
    market, market_diagnostics = _parse_market_window(MARKET, start, end)
    funding, funding_diagnostics = _parse_funding_window(FUNDING, start, end)
    return (
        market,
        funding,
        {
            "physical_window": [start.isoformat(), end.isoformat()],
            "market": market_diagnostics,
            "funding": funding_diagnostics,
            "full_market_file_sha256_not_recomputed": True,
            "full_funding_file_sha256_not_recomputed": True,
            "reason": "stop before parsing the sealed stage boundary",
        },
    )


def _simulate_subperiods(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    subperiods: dict[str, TimeWindow],
    *,
    cost_rate: float,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    return {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=start,
            period_end=end,
            cost_rate=cost_rate,
            cfg=cfg,
        )
        for name, (start, end) in subperiods.items()
    }


def _performance_gates(
    metrics: dict[str, Any],
    stress: dict[str, Any],
    subperiods: dict[str, Any],
    distribution: dict[str, Any],
    *,
    p_max: float,
    total_trades_min: int,
    each_side_min: int,
    subperiod_trade_mins: dict[str, int],
    month_share_max: float,
) -> dict[str, bool]:
    return {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": metrics["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": metrics["strict_mdd_pct"] <= 15.0,
        "weekly_cluster_signflip_p_within_limit": metrics["weekly_cluster_signflip"][
            "p_value_one_sided"
        ]
        <= p_max,
        "minimum_trades": metrics["trades"] >= total_trades_min,
        "minimum_each_side_trades": min(distribution["longs"], distribution["shorts"])
        >= each_side_min,
        "mean_gross_underlying_at_least_35bp": metrics["mean_gross_underlying_bp"]
        >= 35.0,
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "each_subperiod_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in subperiods.values()
        ),
        "each_subperiod_minimum_trades": all(
            subperiods[name]["trades"] >= minimum
            for name, minimum in subperiod_trade_mins.items()
        ),
        "single_entry_month_share_within_limit": (
            distribution["max_single_month_share"] <= month_share_max
        ),
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "weekly_cluster_signflip_p": metrics["weekly_cluster_signflip"][
            "p_value_one_sided"
        ],
        "weekly_clusters": metrics["weekly_cluster_signflip"]["cluster_count"],
    }


def _validate_headline(metrics: Any, *, label: str) -> dict[str, Any]:
    required = {
        "absolute_return_pct",
        "cagr_pct",
        "strict_mdd_pct",
        "cagr_to_strict_mdd",
        "trades",
        "mean_gross_underlying_bp",
        "weekly_cluster_signflip_p",
        "weekly_clusters",
    }
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ValueError(f"FLCC-1 malformed headline: {label}")
    for key in required:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"FLCC-1 nonnumeric headline: {label}/{key}")
        if not math.isfinite(float(value)):
            raise ValueError(f"FLCC-1 nonfinite headline: {label}/{key}")
    if int(metrics["trades"]) != metrics["trades"] or metrics["trades"] < 0:
        raise ValueError(f"FLCC-1 invalid trade count: {label}")
    if (
        int(metrics["weekly_clusters"]) != metrics["weekly_clusters"]
        or metrics["weekly_clusters"] < 0
    ):
        raise ValueError(f"FLCC-1 invalid weekly cluster count: {label}")
    if not 0.0 <= metrics["weekly_cluster_signflip_p"] <= 1.0:
        raise ValueError(f"FLCC-1 invalid p-value: {label}")
    return metrics


def _validate_distribution(distribution: Any, *, label: str) -> dict[str, Any]:
    required = {
        "trades",
        "longs",
        "shorts",
        "max_single_month_count",
        "max_single_month_share",
    }
    if not isinstance(distribution, dict) or set(distribution) != required:
        raise ValueError(f"FLCC-1 malformed entry distribution: {label}")
    for key in ("trades", "longs", "shorts", "max_single_month_count"):
        value = distribution[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"FLCC-1 invalid entry distribution: {label}/{key}")
    share = distribution["max_single_month_share"]
    if (
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or not math.isfinite(float(share))
        or not 0.0 <= share <= 1.0
    ):
        raise ValueError(f"FLCC-1 invalid month share: {label}")
    if distribution["longs"] + distribution["shorts"] != distribution["trades"]:
        raise ValueError(f"FLCC-1 side counts do not sum: {label}")
    return distribution


def _stored_performance_gates(
    metrics: dict[str, Any],
    stress: dict[str, Any],
    subperiods: dict[str, dict[str, Any]],
    distribution: dict[str, Any],
    *,
    p_max: float,
    total_trades_min: int,
    each_side_min: int,
    subperiod_trade_mins: dict[str, int],
    month_share_max: float,
) -> dict[str, bool]:
    return {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": metrics["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": metrics["strict_mdd_pct"] <= 15.0,
        "weekly_cluster_signflip_p_within_limit": metrics["weekly_cluster_signflip_p"]
        <= p_max,
        "minimum_trades": metrics["trades"] >= total_trades_min,
        "minimum_each_side_trades": min(distribution["longs"], distribution["shorts"])
        >= each_side_min,
        "mean_gross_underlying_at_least_35bp": metrics["mean_gross_underlying_bp"]
        >= 35.0,
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "each_subperiod_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in subperiods.values()
        ),
        "each_subperiod_minimum_trades": all(
            subperiods[name]["trades"] >= minimum
            for name, minimum in subperiod_trade_mins.items()
        ),
        "single_entry_month_share_within_limit": (
            distribution["max_single_month_share"] <= month_share_max
        ),
    }


def _recompute_stored_stage1_row(
    row: dict[str, Any], registration: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, bool]]:
    candidate_id = row.get("candidate_id")
    if candidate_id not in {spec.candidate_id for spec in source_clock.CANDIDATE_SPECS}:
        raise ValueError("FLCC-1 Stage1 row has an invalid candidate")
    primary = _validate_headline(row.get("primary"), label=f"{candidate_id}/primary")
    stress = _validate_headline(
        row.get("stress_10bp_per_side"), label=f"{candidate_id}/stress"
    )
    subperiods = row.get("subperiods")
    if not isinstance(subperiods, dict) or set(subperiods) != set(STAGE1_SUBPERIODS):
        raise ValueError(f"FLCC-1 malformed Stage1 subperiods: {candidate_id}")
    for name, metrics in subperiods.items():
        _validate_headline(metrics, label=f"{candidate_id}/{name}")
    controls = row.get("controls")
    if not isinstance(controls, dict) or set(controls) != set(ALL_CLOCK_NAMES) - {
        "primary"
    }:
        raise ValueError(f"FLCC-1 malformed Stage1 controls: {candidate_id}")
    for name, metrics in controls.items():
        _validate_headline(metrics, label=f"{candidate_id}/{name}")
    distributions = row.get("entry_distribution")
    if not isinstance(distributions, dict) or set(distributions) != set(
        ALL_CLOCK_NAMES
    ):
        raise ValueError(f"FLCC-1 malformed Stage1 distributions: {candidate_id}")
    for name, distribution in distributions.items():
        _validate_distribution(distribution, label=f"{candidate_id}/{name}")
    if primary["trades"] != distributions["primary"]["trades"]:
        raise ValueError(f"FLCC-1 primary trade count differs: {candidate_id}")
    for name, metrics in controls.items():
        if metrics["trades"] != distributions[name]["trades"]:
            raise ValueError(
                f"FLCC-1 control trade count differs: {candidate_id}/{name}"
            )

    control_details = row.get("falsification_control_details")
    if not isinstance(control_details, dict) or set(control_details) != set(
        FALSIFICATION_CONTROLS
    ):
        raise ValueError(f"FLCC-1 malformed falsification details: {candidate_id}")
    recomputed_control_gates: dict[str, dict[str, bool]] = {}
    for name, detail in control_details.items():
        if not isinstance(detail, dict) or set(detail) != {
            "stress_10bp_per_side",
            "subperiods",
            "gates",
        }:
            raise ValueError(
                f"FLCC-1 malformed falsification detail: {candidate_id}/{name}"
            )
        control_stress = _validate_headline(
            detail["stress_10bp_per_side"],
            label=f"{candidate_id}/{name}/stress",
        )
        if control_stress["trades"] != controls[name]["trades"]:
            raise ValueError(
                f"FLCC-1 control stress trade count differs: {candidate_id}/{name}"
            )
        control_subperiods = detail["subperiods"]
        if not isinstance(control_subperiods, dict) or set(control_subperiods) != set(
            STAGE1_SUBPERIODS
        ):
            raise ValueError(
                f"FLCC-1 malformed control subperiods: {candidate_id}/{name}"
            )
        for period_name, metrics in control_subperiods.items():
            _validate_headline(metrics, label=f"{candidate_id}/{name}/{period_name}")
        control_gates = detail["gates"]
        if (
            not isinstance(control_gates, dict)
            or set(control_gates) != BASE_GATE_NAMES
            or any(not isinstance(value, bool) for value in control_gates.values())
        ):
            raise ValueError(f"FLCC-1 malformed control gates: {candidate_id}/{name}")
        recomputed = _stored_performance_gates(
            controls[name],
            control_stress,
            control_subperiods,
            distributions[name],
            p_max=0.025,
            total_trades_min=90,
            each_side_min=40,
            subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
            month_share_max=0.08,
        )
        if control_gates != recomputed:
            raise ValueError(
                f"FLCC-1 control gates are not reproducible: {candidate_id}/{name}"
            )
        recomputed_control_gates[name] = recomputed

    support = registration["source_only_support"][candidate_id]["windows"]
    train_support = support["train"]
    primary_distribution = distributions["primary"]
    expected_distribution = {
        "trades": train_support["count"],
        "longs": train_support["long"],
        "shorts": train_support["short"],
        "max_single_month_count": train_support["max_single_month_count"],
        "max_single_month_share": train_support["max_single_month_share"],
    }
    if primary_distribution != expected_distribution:
        raise ValueError(f"FLCC-1 Stage1 source support differs: {candidate_id}")
    for year in STAGE1_SUBPERIODS:
        if subperiods[year]["trades"] != support[year]["count"]:
            raise ValueError(
                f"FLCC-1 Stage1 yearly support differs: {candidate_id}/{year}"
            )

    gates = _stored_performance_gates(
        primary,
        stress,
        subperiods,
        primary_distribution,
        p_max=0.025,
        total_trades_min=90,
        each_side_min=40,
        subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
        month_share_max=0.08,
    )
    gates["primary_ratio_strictly_beats_mechanism_controls"] = all(
        primary["cagr_to_strict_mdd"] > controls[name]["cagr_to_strict_mdd"]
        for name in MECHANISM_CONTROLS
    )
    control_qualification = {
        name: all(recomputed_control_gates[name].values())
        for name in FALSIFICATION_CONTROLS
    }
    gates["falsification_controls_do_not_overall_qualify"] = not any(
        control_qualification.values()
    )
    return gates, control_qualification


def _evaluate_candidate(
    candidate_id: str,
    schedules: dict[str, pd.DataFrame],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
    p_max: float,
    total_trades_min: int,
    each_side_min: int,
    subperiod_trade_mins: dict[str, int],
    month_share_max: float,
) -> dict[str, Any]:
    cfg = EvaluationConfig()
    base = {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=window[0],
            period_end=window[1],
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in schedules.items()
    }
    primary_stress = simulate_schedule(
        market,
        funding,
        schedules["primary"],
        period_start=window[0],
        period_end=window[1],
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    primary_subperiods = _simulate_subperiods(
        market,
        funding,
        schedules["primary"],
        subperiods,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    distributions = {
        name: _entry_distribution(schedule, window)
        for name, schedule in schedules.items()
    }
    gates = _performance_gates(
        base["primary"],
        primary_stress,
        primary_subperiods,
        distributions["primary"],
        p_max=p_max,
        total_trades_min=total_trades_min,
        each_side_min=each_side_min,
        subperiod_trade_mins=subperiod_trade_mins,
        month_share_max=month_share_max,
    )
    gates["primary_ratio_strictly_beats_mechanism_controls"] = all(
        base["primary"]["cagr_to_strict_mdd"] > base[name]["cagr_to_strict_mdd"]
        for name in MECHANISM_CONTROLS
    )
    control_details: dict[str, Any] = {}
    control_qualification: dict[str, bool] = {}
    for name in FALSIFICATION_CONTROLS:
        control_stress = simulate_schedule(
            market,
            funding,
            schedules[name],
            period_start=window[0],
            period_end=window[1],
            cost_rate=cfg.stress_cost_notional_per_side,
            cfg=cfg,
        )
        control_subperiods = _simulate_subperiods(
            market,
            funding,
            schedules[name],
            subperiods,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        control_gates = _performance_gates(
            base[name],
            control_stress,
            control_subperiods,
            distributions[name],
            p_max=p_max,
            total_trades_min=total_trades_min,
            each_side_min=each_side_min,
            subperiod_trade_mins=subperiod_trade_mins,
            month_share_max=month_share_max,
        )
        control_details[name] = {
            "stress_10bp_per_side": _headline(control_stress),
            "subperiods": {
                period_name: _headline(metrics)
                for period_name, metrics in control_subperiods.items()
            },
            "gates": control_gates,
        }
        control_qualification[name] = all(control_gates.values())
    gates["falsification_controls_do_not_overall_qualify"] = not any(
        control_qualification.values()
    )
    return {
        "candidate_id": candidate_id,
        "primary": _headline(base["primary"]),
        "stress_10bp_per_side": _headline(primary_stress),
        "subperiods": {
            name: _headline(metrics) for name, metrics in primary_subperiods.items()
        },
        "controls": {
            name: _headline(base[name]) for name in ALL_CLOCK_NAMES if name != "primary"
        },
        "entry_distribution": distributions,
        "falsification_control_details": control_details,
        "control_overall_qualification": control_qualification,
        "gates": gates,
        "qualified": all(gates.values()),
    }


def _stage1_selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    minimum_year_ratio = min(
        row["subperiods"][year]["cagr_to_strict_mdd"]
        for year in ("2020", "2021", "2022")
    )
    return (
        -minimum_year_ratio,
        -row["primary"]["cagr_to_strict_mdd"],
        -row["stress_10bp_per_side"]["absolute_return_pct"],
        row["candidate_id"],
    )


def _build_stage1_core(
    *,
    freeze_manifest_hash: str,
    schedules: dict[str, dict[str, pd.DataFrame]],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        _evaluate_candidate(
            candidate_id,
            candidate_schedules,
            market,
            funding,
            window=STAGE1,
            subperiods=STAGE1_SUBPERIODS,
            p_max=0.025,
            total_trades_min=90,
            each_side_min=40,
            subperiod_trade_mins={"2020": 20, "2021": 20, "2022": 20},
            month_share_max=0.08,
        )
        for candidate_id, candidate_schedules in schedules.items()
    ]
    qualified = sorted(
        (row for row in candidates if row["qualified"]),
        key=_stage1_selection_key,
    )
    selected = qualified[0]["candidate_id"] if qualified else None
    passed = selected is not None
    return {
        "protocol_version": STAGE1_PROTOCOL,
        "family_id": "FLCC-1",
        "stage": STAGE1_ID,
        "as_of_date": "2026-07-17",
        "evaluator_freeze_manifest_hash": freeze_manifest_hash,
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(EvaluationConfig()),
        "physical_source_diagnostics": diagnostics,
        "stage1_window": [value.isoformat() for value in STAGE1],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "qualified_candidate_ids": [row["candidate_id"] for row in qualified],
        "selected_candidate_id": selected,
        "stage1_passed": passed,
        "advance_to_stage2": passed,
        "2023_outcomes_opened": False,
        "2023_execution_rows_parsed": 0,
        "2023_funding_rows_parsed": 0,
        "opened_windows": STAGE1_OPENED_WINDOWS,
        "sealed_windows": STAGE1_SEALED_WINDOWS,
        "disposition": (
            "PASS_STAGE1_OPEN_2023_ONCE"
            if passed
            else "REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED"
        ),
    }


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window(STAGE1)
    core = _build_stage1_core(
        freeze_manifest_hash=freeze["manifest_hash"],
        schedules=schedules,
        market=market,
        funding=funding,
        diagnostics=diagnostics,
    )
    payload = _seal(core)
    STAGE1_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    STAGE1_DOC.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_DOC.write_text(render_stage_doc(payload))
    return payload


def _verified_passing_stage1(*, expected_freeze_hash: str) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("FLCC-1 Stage1 report is absent; 2023 remains sealed")
    payload = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("FLCC-1 Stage1 result hash mismatch")
    if (
        payload.get("protocol_version") != STAGE1_PROTOCOL
        or payload.get("family_id") != "FLCC-1"
        or payload.get("stage") != STAGE1_ID
    ):
        raise ValueError("FLCC-1 Stage1 result identity changed")
    if payload.get("evaluator_freeze_manifest_hash") != expected_freeze_hash:
        raise ValueError("FLCC-1 Stage1 is not bound to this evaluator freeze")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FLCC-1 Stage1 used another evaluator")
    if payload.get("config") != asdict(EvaluationConfig()):
        raise ValueError("FLCC-1 Stage1 evaluation configuration changed")
    if payload.get("stage1_window") != [value.isoformat() for value in STAGE1]:
        raise ValueError("FLCC-1 Stage1 logical window changed")
    if payload.get("opened_windows") != STAGE1_OPENED_WINDOWS:
        raise ValueError("FLCC-1 Stage1 opened-window contract changed")
    if payload.get("sealed_windows") != STAGE1_SEALED_WINDOWS:
        raise ValueError("FLCC-1 Stage1 sealed-window contract changed")
    diagnostics = payload.get("physical_source_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("physical_window") != [
        value.isoformat() for value in STAGE1
    ]:
        raise ValueError("FLCC-1 Stage1 physical window changed")
    if (
        diagnostics.get("full_market_file_sha256_not_recomputed") is not True
        or diagnostics.get("full_funding_file_sha256_not_recomputed") is not True
        or not isinstance(diagnostics.get("market"), dict)
        or not isinstance(diagnostics.get("funding"), dict)
    ):
        raise ValueError("FLCC-1 Stage1 physical diagnostics are incomplete")

    candidates = payload.get("candidates")
    if (
        payload.get("candidate_count") != len(CANDIDATE_IDS)
        or not isinstance(candidates, list)
        or len(candidates) != len(CANDIDATE_IDS)
    ):
        raise ValueError("FLCC-1 Stage1 candidate family is incomplete")
    registration = _verify_static_inputs()
    observed_ids: list[str] = []
    qualified_rows: list[dict[str, Any]] = []
    for raw_row in candidates:
        if not isinstance(raw_row, dict) or set(raw_row) != STAGE1_ROW_KEYS:
            raise ValueError("FLCC-1 Stage1 candidate row is malformed")
        row = cast(dict[str, Any], raw_row)
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise ValueError("FLCC-1 Stage1 candidate ID is malformed")
        observed_ids.append(candidate_id)
        stored_gates = row.get("gates")
        if (
            not isinstance(stored_gates, dict)
            or set(stored_gates) != STAGE1_GATE_NAMES
            or any(not isinstance(value, bool) for value in stored_gates.values())
        ):
            raise ValueError(f"FLCC-1 Stage1 gate set changed: {candidate_id}")
        stored_controls = row.get("control_overall_qualification")
        if (
            not isinstance(stored_controls, dict)
            or set(stored_controls) != set(FALSIFICATION_CONTROLS)
            or any(not isinstance(value, bool) for value in stored_controls.values())
        ):
            raise ValueError(
                f"FLCC-1 Stage1 control qualification changed: {candidate_id}"
            )
        recomputed_gates, recomputed_controls = _recompute_stored_stage1_row(
            row, registration
        )
        if stored_gates != recomputed_gates:
            raise ValueError(
                f"FLCC-1 Stage1 gates are not reproducible: {candidate_id}"
            )
        if stored_controls != recomputed_controls:
            raise ValueError(
                f"FLCC-1 Stage1 control gates are not reproducible: {candidate_id}"
            )
        qualified = row.get("qualified")
        if not isinstance(qualified, bool) or qualified != all(
            recomputed_gates.values()
        ):
            raise ValueError(
                f"FLCC-1 Stage1 qualification is not reproducible: {candidate_id}"
            )
        if qualified:
            qualified_rows.append(row)
    if tuple(observed_ids) != CANDIDATE_IDS:
        raise ValueError("FLCC-1 Stage1 candidate order or identity changed")

    qualified_rows.sort(key=_stage1_selection_key)
    expected_qualified_ids = [row["candidate_id"] for row in qualified_rows]
    if payload.get("qualified_candidate_ids") != expected_qualified_ids:
        raise ValueError("FLCC-1 Stage1 qualified ordering changed")
    selected = expected_qualified_ids[0] if expected_qualified_ids else None
    passed = selected is not None
    if payload.get("selected_candidate_id") != selected:
        raise ValueError("FLCC-1 Stage1 selected candidate is not reproducible")
    if payload.get("stage1_passed") is not passed:
        raise ValueError("FLCC-1 Stage1 pass flag is not reproducible")
    if payload.get("advance_to_stage2") is not passed:
        raise ValueError("FLCC-1 Stage1 advance flag is not reproducible")
    if payload.get("2023_outcomes_opened") is not False:
        raise ValueError("FLCC-1 Stage1 already opened 2023 outcomes")
    if payload.get("2023_execution_rows_parsed") != 0:
        raise ValueError("FLCC-1 Stage1 parsed 2023 execution outcomes")
    if payload.get("2023_funding_rows_parsed") != 0:
        raise ValueError("FLCC-1 Stage1 parsed 2023 funding outcomes")
    expected_disposition = (
        "PASS_STAGE1_OPEN_2023_ONCE"
        if passed
        else "REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED"
    )
    if payload.get("disposition") != expected_disposition:
        raise ValueError("FLCC-1 Stage1 disposition is not reproducible")
    if not passed:
        raise ValueError("FLCC-1 Stage1 failed; 2023 remains sealed")

    # A self-sealed report is not trusted. Re-run only the already-open Stage1
    # window and require byte-equivalent report semantics before Stage2 can be
    # parsed. This keeps 2023 physically sealed while defeating forged pass flags.
    schedules = load_schedules()
    market, funding, recomputed_diagnostics = load_execution_window(STAGE1)
    expected_payload = _seal(
        _build_stage1_core(
            freeze_manifest_hash=expected_freeze_hash,
            schedules=schedules,
            market=market,
            funding=funding,
            diagnostics=recomputed_diagnostics,
        )
    )
    if payload != expected_payload:
        raise ValueError("FLCC-1 Stage1 report is not reproducible from frozen sources")
    return payload


def evaluate_stage2() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1(expected_freeze_hash=freeze["manifest_hash"])
    selected = str(stage1["selected_candidate_id"])
    schedules = load_schedules()[selected]
    market, funding, diagnostics = load_execution_window(STAGE2)
    candidate = _evaluate_candidate(
        selected,
        schedules,
        market,
        funding,
        window=STAGE2,
        subperiods=STAGE2_SUBPERIODS,
        p_max=0.10,
        total_trades_min=20,
        each_side_min=7,
        subperiod_trade_mins={"2023_h1": 14, "2023_h2": 6},
        month_share_max=0.20,
    )
    stage1_selected = next(
        row for row in stage1["candidates"] if row["candidate_id"] == selected
    )
    mechanism_dominance = all(
        min(
            stage1_selected["primary"]["cagr_to_strict_mdd"],
            candidate["primary"]["cagr_to_strict_mdd"],
        )
        > min(
            stage1_selected["controls"][name]["cagr_to_strict_mdd"],
            candidate["controls"][name]["cagr_to_strict_mdd"],
        )
        for name in MECHANISM_CONTROLS
    )
    falsification_both_splits = any(
        stage1_selected["control_overall_qualification"][name]
        and candidate["control_overall_qualification"][name]
        for name in FALSIFICATION_CONTROLS
    )
    final_gates = {
        **candidate["gates"],
        "minimum_train_2023_primary_ratio_strictly_beats_controls": mechanism_dominance,
        "falsification_control_does_not_qualify_both_splits": (
            not falsification_both_splits
        ),
    }
    passed = all(final_gates.values())
    core = {
        "protocol_version": "federal_liquidity_component_concordance_stage2_v1",
        "as_of_date": "2026-07-17",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "stage1_manifest_hash": stage1["manifest_hash"],
        "selected_candidate_id": selected,
        "physical_source_diagnostics": diagnostics,
        "stage2_window": [value.isoformat() for value in STAGE2],
        "candidate": candidate,
        "final_gates": final_gates,
        "stage2_passed": passed,
        "advance_to_orthogonality_audit": passed,
        "sealed": ["2024", "2025", "2026_ytd"],
    }
    payload = _seal(core)
    STAGE2_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    STAGE2_DOC.write_text(render_stage_doc(payload))
    return payload


def render_stage_doc(report: dict[str, Any]) -> str:
    if "candidates" in report:
        rows = report["candidates"]
        lines = [
            "# FLCC-1 Stage1 — 2020–2022",
            "",
            "| Candidate | Abs return | CAGR | strict MDD | CAGR/MDD | Trades | p | Qualified |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            metric = row["primary"]
            lines.append(
                f"| {row['candidate_id']} | {metric['absolute_return_pct']:.2f}% | "
                f"{metric['cagr_pct']:.2f}% | {metric['strict_mdd_pct']:.2f}% | "
                f"{metric['cagr_to_strict_mdd']:.2f} | {metric['trades']} | "
                f"{metric['weekly_cluster_signflip_p']:.4f} | {row['qualified']} |"
            )
        lines.extend(
            [
                "",
                f"Stage1 passed: **{report['stage1_passed']}**",
                f"Selected candidate: **{report['selected_candidate_id']}**",
                "2023 outcomes opened: **False**",
                "",
            ]
        )
        return "\n".join(lines)

    candidate = report["candidate"]
    metric = candidate["primary"]
    return "\n".join(
        [
            "# FLCC-1 Stage2 — sealed 2023",
            "",
            f"Selected candidate: **{report['selected_candidate_id']}**",
            f"Absolute return: **{metric['absolute_return_pct']:.2f}%**",
            f"CAGR: **{metric['cagr_pct']:.2f}%**",
            f"Strict MDD: **{metric['strict_mdd_pct']:.2f}%**",
            f"CAGR / strict MDD: **{metric['cagr_to_strict_mdd']:.2f}**",
            f"Trades: **{metric['trades']}**",
            f"Stage2 passed: **{report['stage2_passed']}**",
            "",
        ]
    )


def _print_stage(report: dict[str, Any]) -> None:
    if "candidates" in report:
        for row in report["candidates"]:
            print(row["candidate_id"], json.dumps(row["primary"], sort_keys=True))
        print(
            f"stage1_passed={report['stage1_passed']} "
            f"selected={report['selected_candidate_id']}"
        )
    else:
        print(json.dumps(report["candidate"]["primary"], indent=2, sort_keys=True))
        print(f"stage2_passed={report['stage2_passed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--stage1", action="store_true")
    action.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_evaluator(), indent=2, sort_keys=True))
    elif args.stage1:
        _print_stage(evaluate_stage1())
    else:
        _print_stage(evaluate_stage2())


if __name__ == "__main__":
    main()
