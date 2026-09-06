"""Outcome-blind preregistration for the fixed HVMCPAC-8 pairwise-AND battery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "HVMCPAC-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_multi_condition_pairwise_concordance_"
    "preregistration_2026-08-14.json"
)
COMPONENT_ORDER = ("CARSC-8", "HVTCCR-8", "HVTFR-8", "HVLZC-8")
CANDIDATE_FAMILY = tuple(
    f"{left}__AND__{right}"
    for index, left in enumerate(COMPONENT_ORDER)
    for right in COMPONENT_ORDER[index + 1 :]
)
FAMILY_SIZE = 6
FAMILYWISE_ALPHA = 0.10
BONFERRONI_RAW_P_MAX = FAMILYWISE_ALPHA / FAMILY_SIZE

COMPONENT_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "CARSC-8": {
        "preregistration": {"path": "results/cross_alt_return_synchrony_continuation_relay_preregistration_2026-08-10.json", "sha256": "fa9ed9a30e8cbd0532b690acd585d371c91c91a725e7c1b8f4f3187544d0c8e4"},
        "support": {"path": "results/cross_alt_return_synchrony_continuation_relay_support_2026-08-10.json", "sha256": "4302f99eae2772618c94af86af1f951efae1e7b0148c8c6432f2368e802b78a1"},
        "gross9": {"path": "results/cross_alt_return_synchrony_continuation_relay_gross9_novelty_2026-08-10.json", "sha256": "847be3fa882f345b487aa0fac363f7da2681b436df44294bbdc3b8ea54448d62"},
        "clock": {"path": "data/cross_alt_return_synchrony_continuation_relay_clocks_2023_2026.csv.gz", "sha256": "871d7201a04ae938294948d321abb23e99e7b745ddccc1149491940562b69e52"},
    },
    "HVTCCR-8": {
        "preregistration": {"path": "results/high_volatility_quote_turnover_concentration_continuation_relay_preregistration_2026-08-10.json", "sha256": "b3fd974ae3ba804679e63d30ea02ee6d0ad3246981e80b7308242d09d19e3d26"},
        "support": {"path": "results/high_volatility_quote_turnover_concentration_continuation_relay_support_2026-08-10.json", "sha256": "d874a611ae2ea65637fbbf7ca607062cc3df9c4d5989019bb23bf1abbb2713b3"},
        "gross9": {"path": "results/high_volatility_quote_turnover_concentration_continuation_relay_gross9_novelty_2026-08-10.json", "sha256": "910993f4a70ba770a9e66273630e14ca4df175864ba8f7009cb388c31fd8c28f"},
        "clock": {"path": "data/high_volatility_quote_turnover_concentration_continuation_relay_clocks_2023_2026.csv.gz", "sha256": "2485d9759fb4ebd9f247a87ddaa06c0fb4ec1ced100d4a82e3baac473ba4d294"},
    },
    "HVTFR-8": {
        "preregistration": {"path": "results/high_volatility_time_price_trend_fit_relay_preregistration_2026-08-10.json", "sha256": "2bbf372142c15f00714bfc564013a4de4e326f291e447491d8c3d28519d67745"},
        "support": {"path": "results/high_volatility_time_price_trend_fit_relay_support_2026-08-10.json", "sha256": "5547ab700e27c98f04b47fa07756775b0245ff18bb7e466d56f5f0348b570588"},
        "gross9": {"path": "results/high_volatility_time_price_trend_fit_relay_gross9_novelty_2026-08-10.json", "sha256": "890196f90cc3e9ffd58c4fe0d71dcee61681246b509e858c3f7591277a93e984"},
        "clock": {"path": "data/high_volatility_time_price_trend_fit_relay_clocks_2023_2026.csv.gz", "sha256": "f01c22a8409ef9b7f12b17d26b3a3b25564b3aad210a6c00fff789756cbe4151"},
    },
    "HVLZC-8": {
        "preregistration": {"path": "results/high_volatility_lempel_ziv_compressibility_relay_preregistration_2026-08-10.json", "sha256": "1aff82bc58760b8dfc8c14798c20085dc8e9387caaec71f478564855e6afda8b"},
        "support": {"path": "results/high_volatility_lempel_ziv_compressibility_relay_support_2026-08-10.json", "sha256": "d85fabbd757859f1c9fa38385d992b2a79bf6b9ebfd4ac32673f88e30d4bce3c"},
        "gross9": {"path": "results/high_volatility_lempel_ziv_compressibility_relay_gross9_novelty_2026-08-10.json", "sha256": "3ca73e95678052c3dc9677fa01b16e3576bc00f53cfef7ffb6dbaad40fed8e59"},
        "clock": {"path": "data/high_volatility_lempel_ziv_compressibility_relay_clocks_2023_2026.csv.gz", "sha256": "7d32fd2a4bb3ec02bae29966deb554410f325b2c32b38cb15b459ad2ae8d7ea3"},
    },
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_multi_condition_pairwise_concordance_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-14",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "combination_incidence_opened": False,
        "combination_outcomes_opened": False,
        "component_order": list(COMPONENT_ORDER),
        "component_artifacts": COMPONENT_ARTIFACTS,
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "construction": {
            "operator": "pairwise AND",
            "pair_definition": "all six unordered pairs from the four frozen components",
            "intersection": "retain only rows with exactly equal entry_time and exactly equal side",
            "timestamp_tolerance": "none",
            "side_tolerance": "none",
            "output_entry_time": "the shared exact component entry_time",
            "output_side": "the shared exact strict nonzero component side",
            "reservation": "apply the ordinary global half-open reservation after intersection; exit first on equal open",
            "component_formula_threshold_clock_mutability": "immutable",
        },
        "clock": {
            "entry": "exact shared component entry timestamp",
            "hold": "8 elapsed hours",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "gross9_novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": FAMILYWISE_ALPHA,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "train_only_selection": {
            "eligibility": "pair passes its frozen all-stage source-support gate and Gross9 novelty gate",
            "ranking_metric": "descending train base-cost CAGR divided by strict MDD",
            "tie_breaks": [
                "descending train base-cost absolute return",
                "ascending fixed candidate_family order",
            ],
            "winner_train_gate": "raw rank-one pair must pass every train economic gate",
            "freeze_deadline": "winner identity and complete train evidence are written before any test outcome is opened",
            "no_substitution": "if raw rank one fails train or any later gate, terminate; never substitute rank two or another pair",
            "future_reselection_or_repair": False,
        },
        "familywise_multiplicity": {
            "family": "all six pairwise-AND candidates, including pairs failing source or Gross9 gates",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 6 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "all_component_standalone_outcomes_known": True,
            "component_standalone_outcomes_used_to_change_components": False,
            "component_formulas_thresholds_clocks_frozen": True,
            "combination_incidence_opened": False,
            "combination_postentry_returns_or_pnl_opened": False,
            "test_outcomes_opened_before_winner_freeze": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence per pair: source support, Gross9 novelty, train-only ranking and rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no pair substitution, formula, threshold, side, clock, hold, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze the raw train rank one using synthetic/result rows supplied later."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("HVMCPAC-8 selection requires exactly six candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("HVMCPAC-8 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("HVMCPAC-8 selection requires the exact frozen family")
    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        if type(row.get("source_pass")) is not bool or type(row.get("gross9_pass")) is not bool:
            raise ValueError("HVMCPAC-8 pass flags must be booleans")
        if not (row["source_pass"] and row["gross9_pass"]):
            continue
        ratio, absolute_return = row.get("train_cagr_to_strict_mdd"), row.get("train_absolute_return")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)):
            raise ValueError("HVMCPAC-8 train ratio must be finite")
        if isinstance(absolute_return, bool) or not isinstance(absolute_return, (int, float)) or not math.isfinite(float(absolute_return)):
            raise ValueError("HVMCPAC-8 train return must be finite")
        if type(row.get("train_economic_pass")) is not bool:
            raise ValueError("HVMCPAC-8 train economic pass must be boolean")
        eligible.append((order, row, float(ratio), float(absolute_return)))
    if not eligible:
        raise RuntimeError("HVMCPAC-8 has no source-and-Gross9-pass pair")
    order, winner, ratio, absolute_return = sorted(
        eligible, key=lambda item: (-item[2], -item[3], item[0])
    )[0]
    if not winner["train_economic_pass"]:
        raise RuntimeError("HVMCPAC-8 raw rank one failed train; no substitution")
    return {
        "candidate": winner["candidate"],
        "family_order": order + 1,
        "train_cagr_to_strict_mdd": ratio,
        "train_absolute_return": absolute_return,
        "frozen_before_test": True,
        "substitution_authorized": False,
    }


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVMCPAC-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVMCPAC-8 candidate family drift")
    if value.get("component_artifacts") != COMPONENT_ARTIFACTS:
        raise RuntimeError("HVMCPAC-8 component artifact bindings drift")
    for component in COMPONENT_ORDER:
        for artifact_type in ("preregistration", "support", "gross9", "clock"):
            artifact = COMPONENT_ARTIFACTS[component][artifact_type]
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"HVMCPAC-8 {component} {artifact_type} artifact drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
