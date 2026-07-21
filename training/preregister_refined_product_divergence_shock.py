"""Emit the outcome-blind RPDS-576 preregistration artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "refined_product_divergence_shock_preregistration_v1"
POLICY_ID = "RPDS-576"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MECHANISM_DOCUMENT = Path(
    "docs/refined-product-divergence-shock-preregistration-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "089203b3213b1c6565ad82974ea6c807889bf360f9fcfc60b9b548edfa4e67b3"
)
GENERATOR_SOURCE = Path("training/preregister_refined_product_divergence_shock.py")
DEFAULT_OUTPUT = Path(
    "results/refined_product_divergence_shock_preregistration_2026-07-21.json"
)

SOURCE_BINDINGS = {
    "panel": {
        "path": Path(
            "data/eia_petroleum_stock_breadth_2019_2023/"
            "eia_petroleum_stock_breadth_2019_2023.csv.gz"
        ),
        "sha256": "26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b",
    },
    "source_manifest": {
        "path": Path("data/eia_petroleum_stock_breadth_2019_2023/source_manifest.json"),
        "sha256": "3969288900528d103016cdb0870a11269c1b352b9077faffdc61427f7fce29fb",
    },
    "build_manifest": {
        "path": Path("data/eia_petroleum_stock_breadth_2019_2023/build_manifest.json"),
        "sha256": "d6813b1a5677c9222a1197343900d6b03381f35ff9db8688892b77e4cd9c0661",
    },
}

COMPARATOR_BINDINGS = {
    "epsb": {
        "path": Path("results/eia_petroleum_stock_breadth_clocks_2026-07-17.csv.gz"),
        "sha256": "6c6470ba90e8bd826bb566e5952755dd8a872b29c1ba0643d29e08ab23e44400",
    },
    "live": {
        "path": Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
    },
    "far": {
        "path": Path("results/cchr_far_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "2203bdb6122fbbc4eaf28b0ddf626362a6cde1a1153ff13c74722eba340f3ccf",
    },
    "dtv": {
        "path": Path("results/cchr_dtv_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "798e442f8ff4867232079cd6b500f388326b42c920297d407abbd1c4c85df225",
    },
    "pdlh": {
        "path": Path("results/cchr_pdlh_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "5001efba77620c45a4784a71a7d5ab5a3127a4549be926a581e6597ed3e0c9fa",
    },
}


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_binding(binding: Mapping[str, Any]) -> None:
    actual = sha256_file(binding["path"])
    if actual != binding["sha256"]:
        raise RuntimeError(f"hash drift: {binding['path']}")


def _serialized_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(binding["path"]), "sha256": str(binding["sha256"])}
        for name, binding in bindings.items()
    }


def build_preregistration() -> dict[str, Any]:
    if sha256_file(MECHANISM_DOCUMENT) != MECHANISM_DOCUMENT_SHA256:
        raise RuntimeError("mechanism document hash drift")
    for binding in SOURCE_BINDINGS.values():
        verify_binding(binding)
    for binding in COMPARATOR_BINDINGS.values():
        verify_binding(binding)

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "mechanism_document": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "generator": {
            "path": str(GENERATOR_SOURCE),
            "sha256": sha256_file(GENERATOR_SOURCE),
        },
        "source_bindings": _serialized_bindings(SOURCE_BINDINGS),
        "source_contract": {
            "rows": 259,
            "complete_rows": 258,
            "quarantined_rows": 1,
            "release_years": [2019, 2020, 2021, 2022, 2023],
            "allowed_columns": [
                "release_date",
                "available_time_utc",
                "source_complete",
                "published_difference_consistent",
                "commercial_crude_change_mmbbl",
                "gasoline_change_mmbbl",
                "distillate_change_mmbbl",
            ],
            "availability": "next UTC calendar day 13:00 after release date",
            "zero_change_policy": "ineligible",
            "incomplete_or_inconsistent_policy": "quarantine and clear delayed state",
        },
        "signal": {
            "crude_sign": "sign(commercial_crude_change_mmbbl)",
            "gasoline_sign": "sign(gasoline_change_mmbbl)",
            "distillate_sign": "sign(distillate_change_mmbbl)",
            "predicate": "gasoline == distillate != 0 and crude == -gasoline",
            "side": "gasoline/distillate sign: build LONG, draw SHORT",
            "thresholds": [],
            "parameter_grid": [],
        },
        "execution": {
            "signal_time": "available_time_utc",
            "entry_delay_minutes": 5,
            "hold_five_minute_bars": 576,
            "hold_hours": 48,
            "leverage": 0.5,
            "global_nonoverlap": True,
            "split_containment": ["release", "signal", "entry", "exit"],
        },
        "splits": {
            "history": ["2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "train": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed": "2024 and later",
        },
        "support_gates": {
            "train": {
                "events_min": 24,
                "events_max": 75,
                "events_per_year_min": 5,
                "side_share_min": 0.25,
                "month_share_max": 0.25,
            },
            "selection": {
                "events_min": 8,
                "events_max": 24,
                "events_per_half_min": 3,
                "both_sides_required": True,
                "month_share_max": 0.25,
            },
        },
        "controls": [
            "direction_flip",
            "refined_only",
            "crude_only",
            "epsb_concordance_48h",
            "one_release_delay",
            "deterministic_random_side",
            "latency_plus_5m",
        ],
        "novelty": {
            "bindings": _serialized_bindings(COMPARATOR_BINDINGS),
            "comparison_start": "2020-01-01T00:00:00Z",
            "comparison_end_exclusive": "2024-01-01T00:00:00Z",
            "truncate_to_observed_comparator_events": False,
            "exact_entry_jaccard_max": 0.10,
            "one_to_one_tolerance_hours": 6,
            "maximum_bidirectional_containment_max": 0.25,
            "absolute_signed_exposure_correlation_max": 0.35,
            "epsb_primary_exact_release_overlap_required": 0,
            "epsb_primary_exact_entry_overlap_required": 0,
            "short_circuit_before_comparator_access_on_support_failure": True,
        },
        "later_outcome_contract": {
            "authorized": False,
            "sequential_opening": ["train_2020_2022", "selection_2023"],
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "full_calendar_cagr": True,
            "strict_intratrade_mdd": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_pct_max": 15.0,
            "control_ratio_margin_min": 0.25,
            "report_absolute_return_with_ratio": True,
        },
        "outcome_boundary": {
            "prefreeze_source_value_rows_read_for_schema": 1,
            "prefreeze_comparator_clock_rows_read_for_schema": 10,
            "rpds_predicate_evaluations": 0,
            "candidate_clock_rows_created": 0,
            "comparator_overlap_metrics_computed": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "authorization": {
            "next_action": "source support and novelty only",
            "outcome_evaluator": False,
            "post_2023_source_access": False,
            "threshold_or_hold_repair": False,
        },
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_preregistration()
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
