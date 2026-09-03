"""Build the net-position-risk successor config for G9-OVERLAP-PORT-1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from training import optimize_gross9_overlap_portfolio as optimizer

POLICY_ID = "G9-OVERLAP-NET-PORT-1"
PROTOCOL_VERSION = "gross9_overlap_net_position_config_v1"
AS_OF_DATE = "2026-09-03"
SELECTION = Path("results/gross9_overlap_portfolio_train_selection_2026-09-03.json")
SELECTION_SHA256 = "a197cbb568e888e82edeca01e2108fad2588e42eb9be74fa9c213a92c33caa8c"
SELECTION_MANIFEST_HASH = "00663b4e9112d02c6a9639e53fb490404a95ab59a62ed730b5eb9f213f876c21"
UNIVERSE = Path("results/gross9_overlap_portfolio_universe_2026-09-03.json")
UNIVERSE_SHA256 = "e2a631cface501a1264d736c6635e64c2931667425b6abe873123d5e6c37ac8c"
CONFIG_OUTPUT = Path("configs/shadow/gross9_overlap_net_position_portfolio_2026-09-03.json")
RESULT_OUTPUT = Path("results/gross9_overlap_net_position_config_audit_2026-09-03.json")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON object required: {path}")
    core = {key: item for key, item in value.items() if key not in {"manifest_hash", "protocol_hash"}}
    observed = value.get("manifest_hash", value.get("protocol_hash"))
    if observed != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} artifact hash drift: {path}")
    return value


def net_exposure_metrics(clock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    events: dict[pd.Timestamp, list[tuple[int, float, float]]] = {}
    for row in clock.itertuples(index=False):
        signed = float(row.weight) * int(row.side)
        weight = abs(float(row.weight))
        events.setdefault(pd.Timestamp(row.entry_time), []).append((1, signed, weight))
        events.setdefault(pd.Timestamp(row.exit_time), []).append((-1, signed, weight))
    net = gross = 0.0
    last = start
    net_seconds = gross_seconds = 0.0
    max_abs_net = max_gross = 0.0
    for timestamp in sorted(events):
        elapsed = max(0.0, (timestamp - last).total_seconds())
        net_seconds += abs(net) * elapsed
        gross_seconds += gross * elapsed
        last = timestamp
        for kind, signed, weight in sorted(events[timestamp], key=lambda item: item[0]):
            if kind < 0:
                net -= signed
                gross -= weight
            else:
                net += signed
                gross += weight
        if abs(net) < 1e-12:
            net = 0.0
        if abs(gross) < 1e-12:
            gross = 0.0
        max_abs_net = max(max_abs_net, abs(net))
        max_gross = max(max_gross, gross)
    elapsed = max(0.0, (end - last).total_seconds())
    net_seconds += abs(net) * elapsed
    gross_seconds += gross * elapsed
    duration = max((end - start).total_seconds(), 1e-12)
    return {
        "max_abs_net_position": float(max_abs_net),
        "mean_abs_net_position": float(net_seconds / duration),
        "legacy_max_nonnet_gross": float(max_gross),
        "legacy_mean_nonnet_gross": float(gross_seconds / duration),
    }


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(SELECTION) != SELECTION_SHA256 or sha256_file(UNIVERSE) != UNIVERSE_SHA256:
        raise RuntimeError(f"{POLICY_ID} frozen source receipt drift")
    selection = load_json(SELECTION)
    universe = load_json(UNIVERSE)
    if selection.get("manifest_hash") != SELECTION_MANIFEST_HASH:
        raise RuntimeError(f"{POLICY_ID} selection manifest drift")
    rank1 = selection["authoritative_rank1"]
    weights = {key: float(value) for key, value in rank1["sleeve_weights"].items()}
    records = {row["sleeve_id"]: row for row in universe["sleeves"]}
    start = optimizer._utc(optimizer.TRAIN_PROXY_WINDOW[0])
    end = optimizer._utc(optimizer.TRAIN_PROXY_WINDOW[1])
    sleeves = {
        sleeve_id: optimizer.load_sleeve_clock(records[sleeve_id], "train", start, end, root=Path.cwd())
        for sleeve_id in weights
    }
    spec = optimizer.PortfolioSpec(weights=tuple(weights.items()), proxy_score=0.0, proxy_metrics={})
    clock = optimizer.build_portfolio_clock(spec, sleeves)
    net_risk = net_exposure_metrics(clock, start, end)
    net_risk["max_risk_reduction_vs_legacy_share"] = float(
        1.0 - net_risk["max_abs_net_position"] / net_risk["legacy_max_nonnet_gross"]
    )
    net_risk["mean_risk_reduction_vs_legacy_share"] = float(
        1.0 - net_risk["mean_abs_net_position"] / net_risk["legacy_mean_nonnet_gross"]
    )

    interval_rows: list[dict[str, Any]] = []
    pre_net_total = 0.0
    for sleeve_id, weight in weights.items():
        count = int(clock["sleeve"].eq(sleeve_id).sum())
        turnover = 2.0 * abs(weight) * count
        pre_net_total += turnover
        interval_rows.append(
            {
                "sleeve_id": sleeve_id,
                "weight": weight,
                "intervals": count,
                "hypothetical_pre_net_turnover_weight": turnover,
            }
        )
    for row in interval_rows:
        row["pre_net_turnover_share"] = row["hypothetical_pre_net_turnover_weight"] / pre_net_total

    risk = rank1["risk"]
    base = rank1["primary"]["base"]
    stress = rank1["primary"]["stress"]
    finalist_rows = selection["exact_finalists"]
    exposure_gate_names = ("mean_gross_exposure_cap", "max_gross_exposure_cap")
    exposure_gate_failures = sum(
        not all(bool(row["gates"][name]) for name in exposure_gate_names)
        for row in finalist_rows
    )
    turnover_gate_passes = sum(bool(row["gates"]["turnover_cap"]) for row in finalist_rows)
    selection_invariance = {
        "scope": "frozen Jul-Nov proxy search and 64 exact-ledger finalists only",
        "exact_finalists_checked": len(finalist_rows),
        "exact_score_formula_changed": False,
        "legacy_exposure_gate_failures": exposure_gate_failures,
        "turnover_gate_passes": turnover_gate_passes,
        "rank1_legacy_exposure_gates_passed": all(
            bool(rank1["gates"][name]) for name in exposure_gate_names
        ),
        "reason": (
            "all exact finalists already passed both legacy exposure gates, all failed the "
            "unchanged turnover cap, and exposure is absent from the exact score; netting "
            "therefore cannot change the frozen raw rank1"
        ),
    }
    annualized_net_turnover = float(risk["actual_net_turnover_weight_per_day"]) * 365.25
    turnover_cap_per_day = 100.0 / 365.25
    cost_detail = {
        "measurement_window_days": int((end - start).days),
        "atomic_transitions": int(base["transitions"]),
        "nonzero_net_execution_events": int(risk["nonzero_net_execution_events"]),
        "nonzero_net_execution_events_per_day": (
            float(risk["nonzero_net_execution_events"]) / float((end - start).days)
        ),
        "hypothetical_pre_net_turnover_weight": float(risk["turnover_weight"]),
        "actual_aggregate_net_turnover_weight": float(risk["actual_net_turnover_weight"]),
        "actual_aggregate_net_turnover_weight_per_day": float(risk["actual_net_turnover_weight_per_day"]),
        "annualized_aggregate_net_turnover_x": annualized_net_turnover,
        "annualized_turnover_cap_x": 100.0,
        "turnover_cap_multiple": annualized_net_turnover / 100.0,
        "turnover_excess_over_cap_x": annualized_net_turnover - 100.0,
        "turnover_cap_weight_per_day": turnover_cap_per_day,
        "turnover_excess_weight_per_day": float(risk["actual_net_turnover_weight_per_day"]) - turnover_cap_per_day,
        "netting_savings_share": float(risk["netting_savings_share"]),
        "base_cost_each_notional_side_bp": optimizer.BASE_COST_BP,
        "base_fee_cash": float(base["total_fees"]),
        "base_fee_pct_of_initial_equity": float(base["total_fees"]) / float(base["initial_equity"]) * 100.0,
        "base_funding_cash_received": float(base["total_funding"]),
        "base_fee_less_funding_cash": float(base["total_fees"] - base["total_funding"]),
        "base_fee_less_funding_pct_of_initial_equity": (
            float(base["total_fees"] - base["total_funding"])
            / float(base["initial_equity"])
            * 100.0
        ),
        "stress_cost_each_notional_side_bp": optimizer.STRESS_COST_BP,
        "stress_fee_cash": float(stress["total_fees"]),
        "stress_fee_pct_of_initial_equity": float(stress["total_fees"]) / float(stress["initial_equity"]) * 100.0,
        "stress_funding_cash_received": float(stress["total_funding"]),
        "stress_fee_less_funding_cash": float(stress["total_fees"] - stress["total_funding"]),
        "stress_fee_less_funding_pct_of_initial_equity": (
            float(stress["total_fees"] - stress["total_funding"])
            / float(stress["initial_equity"])
            * 100.0
        ),
        "max_single_sleeve_pre_net_turnover_share": float(risk["max_sleeve_turnover_share"]),
        "single_sleeve_share_cap": 0.40,
        "failed_gates": [name for name, passed in rank1["gates"].items() if not passed],
        "risk_cost_separation": (
            "opposite positions reduce position risk, but fees are charged whenever aggregate "
            "net quantity changes; small net exposure therefore does not imply low turnover"
        ),
        "per_sleeve": interval_rows,
    }
    config_core = {
        "name": "gross9_overlap_net_position_portfolio_2026_09_03",
        "policy_id": POLICY_ID,
        "successor_of": optimizer.POLICY_ID,
        "status": "terminal_train_reject_diagnostic_config_not_live",
        "as_of": AS_OF_DATE,
        "shadow_only": True,
        "live_capital_authorized": False,
        "order_submission_enabled": False,
        "position_risk": {
            "direction": "sign(sum(active sleeve side * sleeve weight))",
            "size": "abs(sum(active sleeve side * sleeve weight))",
            "long_short_offset_for_risk": True,
            "gross_risk_definition": "absolute aggregate signed position; long and short offset fully",
            "execution_cost_definition": "absolute aggregate net quantity change",
        },
        "sleeve_weights": weights,
        "selected_weight_sum": float(sum(weights.values())),
        "net_position_risk_metrics": net_risk,
        "selection_invariance_evidence": selection_invariance,
        "economic_snapshot": {
            "return_pct": float(base["absolute_return_pct"]),
            "cagr_to_strict_mdd": float(base["cagr_to_strict_mdd"]),
            "strict_mdd_pct": float(base["strict_mdd_pct"]),
            "stress_return_pct": float(stress["absolute_return_pct"]),
            "sleeve_intervals": int(base["intervals"]),
            "long_intervals": int(base["long_intervals"]),
            "short_intervals": int(base["short_intervals"]),
        },
        "operating_cost_failure": cost_detail,
        "source_selection": {
            "path": str(SELECTION),
            "sha256": SELECTION_SHA256,
            "manifest_hash": SELECTION_MANIFEST_HASH,
        },
        "evidence_boundary": {
            "new_market_or_funding_rows_opened": 0,
            "december_holdout_opened": False,
            "oos_opened": False,
        },
    }
    config = {**config_core, "protocol_hash": canonical_hash(config_core)}
    audit_core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "config": {"path": str(CONFIG_OUTPUT), "protocol_hash": config["protocol_hash"]},
        "source_selection": config["source_selection"],
        "risk_semantics_changed_by_user": True,
        "portfolio_weights_changed": False,
        "selection_rank_changed": False,
        "net_position_risk_metrics": net_risk,
        "selection_invariance_evidence": selection_invariance,
        "operating_cost_failure": cost_detail,
        "decision": "same_optimal_weights_under_net_position_risk; terminal_turnover_reject_remains",
        "evidence_boundary": config["evidence_boundary"],
    }
    audit = {**audit_core, "manifest_hash": canonical_hash(audit_core)}
    return config, audit


def run(
    config_output: Path = CONFIG_OUTPUT,
    result_output: Path = RESULT_OUTPUT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, audit = build()
    config_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit["config"]["path"] = str(config_output)
    audit["config"]["sha256"] = sha256_file(config_output)
    core = {key: value for key, value in audit.items() if key != "manifest_hash"}
    audit["manifest_hash"] = canonical_hash(core)
    result_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return config, audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-output", type=Path, default=CONFIG_OUTPUT)
    parser.add_argument("--result-output", type=Path, default=RESULT_OUTPUT)
    args = parser.parse_args(argv)
    config, audit = run(args.config_output, args.result_output)
    print(
        json.dumps(
            {
                "config": str(args.config_output),
                "decision": audit["decision"],
                "weights": config["sleeve_weights"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
