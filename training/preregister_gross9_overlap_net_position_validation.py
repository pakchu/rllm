"""Freeze sequential holdout/OOS validation for G9-OVERLAP-NET-PORT-1."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import build_gross9_overlap_net_position_config as net_config
from training import evaluate_gross9_async_active_veto_train_economics as train_sources
from training import evaluate_gross9_qtr_distill_economics as fixed_ledger
from training import optimize_gross9_overlap_portfolio as optimizer

POLICY_ID = net_config.POLICY_ID
PROTOCOL_VERSION = "gross9_overlap_net_position_sequential_validation_freeze_v1"
AS_OF_DATE = "2026-09-03"
CONFIG = net_config.CONFIG_OUTPUT
SELECTION = net_config.SELECTION
UNIVERSE = net_config.UNIVERSE
EVALUATOR = Path("training/evaluate_gross9_overlap_net_position_portfolio.py")
FREEZER = Path("training/preregister_gross9_overlap_net_position_validation.py")
DEFAULT_OUTPUT = Path(
    "results/gross9_overlap_net_position_validation_freeze_2026-09-03.json"
)

STAGES: dict[str, tuple[str, str, str]] = {
    "holdout_dec2023": (
        "train",
        "2023-12-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ),
    "test2024": (
        "test",
        "2024-01-01T00:00:00Z",
        "2025-01-01T00:00:00Z",
    ),
    "eval2025": (
        "eval",
        "2025-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ),
    "final2026": (
        "final",
        "2026-01-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
    ),
}


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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON object required: {path}")
    hash_key = "protocol_hash" if "protocol_hash" in value else "manifest_hash"
    core = {key: item for key, item in value.items() if key != hash_key}
    if value.get(hash_key) != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} artifact hash drift: {path}")
    return value


def count_gzip_csv_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _receipt(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    hash_key = "protocol_hash" if "protocol_hash" in value else "manifest_hash"
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        hash_key: value[hash_key],
    }


def build() -> dict[str, Any]:
    config = load_hashed_json(CONFIG)
    selection = load_hashed_json(SELECTION)
    universe = load_hashed_json(UNIVERSE)
    if not EVALUATOR.is_file():
        raise RuntimeError(f"{POLICY_ID} evaluator missing before freeze: {EVALUATOR}")
    if (
        config.get("policy_id") != POLICY_ID
        or config.get("status") != "train_selected_shadow_only_holdout_oos_unopened"
        or config.get("shadow_only") is not True
        or config.get("live_capital_authorized") is not False
        or config.get("order_submission_enabled") is not False
    ):
        raise RuntimeError(f"{POLICY_ID} config is not eligible for sequential validation")
    gate_policy = config.get("selection_gate_policy", {})
    if (
        gate_policy.get("waived_rejection_gates")
        != ["turnover_cap", "sleeve_turnover_share_cap"]
        or gate_policy.get("non_waived_gate_failures") != []
        or gate_policy.get("train_selected_after_user_waiver") is not True
    ):
        raise RuntimeError(f"{POLICY_ID} user waiver/config gate drift")
    weights = config.get("sleeve_weights", {})
    if weights != selection.get("authoritative_rank1", {}).get("sleeve_weights"):
        raise RuntimeError(f"{POLICY_ID} selected weights drift")
    if config.get("source_selection") != {
        "path": str(SELECTION),
        "sha256": sha256_file(SELECTION),
        "manifest_hash": selection["manifest_hash"],
    }:
        raise RuntimeError(f"{POLICY_ID} selection receipt drift")

    universe_records = {row["sleeve_id"]: row for row in universe["sleeves"]}
    selected_clocks: list[dict[str, Any]] = []
    for sleeve_id, weight in weights.items():
        record = universe_records.get(sleeve_id)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{POLICY_ID} selected sleeve absent from universe: {sleeve_id}")
        clock = record.get("clock", {})
        path = Path(str(clock.get("path", "")))
        if (
            not path.is_file()
            or sha256_file(path) != clock.get("sha256")
            or count_gzip_csv_rows(path) != int(clock.get("rows", -1))
        ):
            raise RuntimeError(f"{POLICY_ID} selected clock receipt drift: {sleeve_id}")
        selected_clocks.append(
            {
                "sleeve_id": sleeve_id,
                "weight": float(weight),
                "path": str(path),
                "sha256": clock["sha256"],
                "rows": int(clock["rows"]),
                "split_counts": record["split_counts"],
            }
        )

    holdout_gates = {
        "minimum_intervals": 8,
        "minimum_active_iso_weeks": 3,
        "absolute_return_positive": True,
        "stress_absolute_return_positive": True,
        "strict_mdd_max_pct": 8.0,
        "mean_abs_net_position_max": 0.85,
        "max_abs_net_position_max": 1.0,
    }
    oos_gates = {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_min": 3.0,
        "strict_mdd_max_pct": 15.0,
        "mean_exposure_weighted_gross_edge_min_bp": 20.0,
        "stress_absolute_return_positive": True,
        "stress_cagr_to_strict_mdd_min": 2.5,
        "each_calendar_half_positive": True,
        "weekly_cluster_signflip_one_sided_p_max": 0.10,
        "aggregate_net_signed_episode_min": {
            "test2024": 12,
            "eval2025": 12,
            "final2026": 8,
        },
        "aggregate_net_signed_episode_definition": (
            "nonzero sign episodes of sum(active sleeve side * sleeve weight) after atomic "
            "same-timestamp netting"
        ),
        "mean_abs_net_position_max": 0.85,
        "max_abs_net_position_max": 1.0,
    }
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "objective": "open the fixed rank1 holdout and OOS outcomes sequentially without reranking or repair",
        "frozen_inputs": {
            "config": _receipt(CONFIG, config),
            "train_selection": _receipt(SELECTION, selection),
            "universe": _receipt(UNIVERSE, universe),
            "selected_clocks": selected_clocks,
        },
        "sequence": {
            "stages": list(STAGES),
            "windows": {
                stage: {"split": split, "start": start, "end": end}
                for stage, (split, start, end) in STAGES.items()
            },
            "stop_on_first_failure": True,
            "rerank_repair_or_substitution_authorized": False,
            "final2026_window_rationale": (
                "inherits the existing Gross9 staged convention frozen before outcomes: "
                "2026-01-01 through the 2026-08-01 final exit open"
            ),
        },
        "gates": {
            "holdout_dec2023": holdout_gates,
            "holdout_gate_rationale": (
                "the one-month internal holdout uses the narrower shape/return/MDD gates "
                "predeclared by G9-OVERLAP-PORT-1; full OOS gates begin at test2024"
            ),
            "oos": oos_gates,
            "waived_cost_gates": ["turnover_cap", "sleeve_turnover_share_cap"],
            "cost_and_turnover_metrics_are_disclosure_only": True,
            "risk_definition": "abs(sum(active sleeve side * sleeve weight))",
        },
        "accounting": {
            "ledger": "fixed sleeve quantities with aggregate-net execution and funding",
            "base_cost_each_notional_side_bp": 6,
            "stress_cost_each_notional_side_bp": 10,
            "strict_mdd": "global high-water mark with held 5m favorable then adverse OHLC",
        },
        "source_plan": {
            "holdout_dec2023": "hash-bound gzip market and exact funding-mark prefix",
            "test2024_eval2025": "hash-bound gzip market plus PostgreSQL exact funding marks",
            "final2026": "PostgreSQL exact 1m-to-5m market aggregation plus exact funding marks",
            "extracted_rows_hashed_in_each_stage_artifact": True,
        },
        "implementation": {
            "evaluator": {"path": str(EVALUATOR), "sha256": sha256_file(EVALUATOR)},
            "freezer": {"path": str(FREEZER), "sha256": sha256_file(FREEZER)},
            "fixed_ledger": {
                "path": str(Path(fixed_ledger.__file__).relative_to(Path.cwd())),
                "sha256": sha256_file(fixed_ledger.__file__),
            },
            "optimizer_utilities": {
                "path": str(Path(optimizer.__file__).relative_to(Path.cwd())),
                "sha256": sha256_file(optimizer.__file__),
            },
            "net_risk_metrics": {
                "path": str(Path(net_config.__file__).relative_to(Path.cwd())),
                "sha256": sha256_file(net_config.__file__),
            },
            "hash_bound_source_loader": {
                "path": str(Path(train_sources.__file__).relative_to(Path.cwd())),
                "sha256": sha256_file(train_sources.__file__),
            },
        },
        "evidence_boundary": {
            "holdout_market_or_funding_rows_opened": 0,
            "oos_market_or_funding_rows_opened": 0,
            "holdout_outcomes_opened": False,
            "oos_outcomes_opened": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} validation freeze manifest drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} validation freeze identity drift")
    if value.get("sequence", {}).get("stages") != list(STAGES):
        raise RuntimeError(f"{POLICY_ID} validation sequence drift")
    if value.get("sequence", {}).get("stop_on_first_failure") is not True:
        raise RuntimeError(f"{POLICY_ID} sequential stop rule drift")
    if value.get("sequence", {}).get("rerank_repair_or_substitution_authorized") is not False:
        raise RuntimeError(f"{POLICY_ID} no-repair rule drift")
    if value.get("gates", {}).get("waived_cost_gates") != [
        "turnover_cap",
        "sleeve_turnover_share_cap",
    ]:
        raise RuntimeError(f"{POLICY_ID} cost waiver drift")
    implementation = value.get("implementation", {})
    expected_implementation_labels = {
        "evaluator",
        "freezer",
        "fixed_ledger",
        "optimizer_utilities",
        "net_risk_metrics",
        "hash_bound_source_loader",
    }
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != expected_implementation_labels
    ):
        raise RuntimeError(f"{POLICY_ID} implementation receipts missing")
    for label, record in implementation.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"{POLICY_ID} implementation receipt malformed: {label}")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"{POLICY_ID} implementation receipt drift: {label}")
    frozen_inputs = value.get("frozen_inputs", {})
    if not isinstance(frozen_inputs, Mapping) or set(frozen_inputs) != {
        "config",
        "train_selection",
        "universe",
        "selected_clocks",
    }:
        raise RuntimeError(f"{POLICY_ID} frozen input receipts missing")
    input_hash_keys = {
        "config": "protocol_hash",
        "train_selection": "manifest_hash",
        "universe": "manifest_hash",
    }
    loaded_inputs: dict[str, dict[str, Any]] = {}
    for label, hash_key in input_hash_keys.items():
        record = frozen_inputs.get(label, {})
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            hash_key,
        }:
            raise RuntimeError(f"{POLICY_ID} frozen input receipt malformed: {label}")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"{POLICY_ID} frozen input receipt drift: {label}")
        loaded = load_hashed_json(path)
        if loaded.get(hash_key) != record.get(hash_key):
            raise RuntimeError(f"{POLICY_ID} frozen input hash binding drift: {label}")
        loaded_inputs[label] = loaded
    selected_clocks = frozen_inputs.get("selected_clocks")
    if not isinstance(selected_clocks, list) or not selected_clocks:
        raise RuntimeError(f"{POLICY_ID} selected clock receipts missing")
    required_clock_keys = {
        "sleeve_id",
        "weight",
        "path",
        "sha256",
        "rows",
        "split_counts",
    }
    observed_weights: dict[str, float] = {}
    for record in selected_clocks:
        if not isinstance(record, Mapping) or set(record) != required_clock_keys:
            raise RuntimeError(f"{POLICY_ID} selected clock receipt malformed")
        path = Path(str(record.get("path", "")))
        rows = int(record.get("rows", -1))
        split_counts = record.get("split_counts")
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or count_gzip_csv_rows(path) != rows
            or not isinstance(split_counts, Mapping)
            or set(split_counts) != {"train", "test", "eval", "final"}
            or sum(int(count) for count in split_counts.values()) != rows
        ):
            raise RuntimeError(f"{POLICY_ID} selected clock receipt drift")
        sleeve_id = str(record.get("sleeve_id", ""))
        if not sleeve_id or sleeve_id in observed_weights:
            raise RuntimeError(f"{POLICY_ID} selected clock identity drift")
        observed_weights[sleeve_id] = float(record.get("weight", 0.0))
    if observed_weights != loaded_inputs["config"].get("sleeve_weights"):
        raise RuntimeError(f"{POLICY_ID} selected clock weights drift")
    boundary = value.get("evidence_boundary", {})
    if boundary.get("holdout_outcomes_opened") is not False or boundary.get("oos_outcomes_opened") is not False:
        raise RuntimeError(f"{POLICY_ID} freeze opened outcomes")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = build()
    validate(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = run(args.output)
    print(
        json.dumps(
            {
                "policy_id": POLICY_ID,
                "output": str(args.output),
                "manifest_hash": value["manifest_hash"],
                "outcomes_opened": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
