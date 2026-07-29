#!/usr/bin/env python3
"""Freeze the unopened CDP1/Gross9 portfolio-interaction experiment.

The standalone CDP1 signal and its 2022/2023 results are already exposed and
must not be changed.  This registration opens no Gross9 market payload and
freezes the only remaining question: whether the fixed CDP1 top1 adds
same-gross portfolio risk efficiency on 2022 and survives a top1-only 2023
interaction veto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "psim_d8_cdp1_gross9_same_gross_marginal_v1"
DEFAULT_OUTPUT = Path(
    "results/psim_d8_cdp1_gross9_marginal_preregistration_2026-07-29.json"
)

CDP_SELECTION = Path(
    "results/psim_d8_cross_protocol_disagreement_2022_selection_"
    "result_2026-07-29.json"
)
CDP_SELECTION_SHA256 = (
    "33cb78065b04a1103b048a0607a4d68f40ada58bd9911b49d1f4da49c73d2f4e"
)
CDP_VETO = Path(
    "results/psim_d8_cross_protocol_disagreement_2023_veto_"
    "result_2026-07-29.json"
)
CDP_VETO_SHA256 = (
    "44ed96d5f2cabc045aac3ecb8f828de548654047097ad445eb8cb868e5c721ce"
)
GROSS9_ANCHOR = Path("results/gross9_pre2025_authoritative_anchor_2026-07-28.json")
GROSS9_ANCHOR_SHA256 = (
    "329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875"
)
GROSS9_CONFIG = Path(
    "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json"
)
GROSS9_CONFIG_SHA256 = (
    "006f82e1f0affad9f96a08a6c600542feec4a0e1198ed99b8630627de4913450"
)
GROSS9_RESULT = Path("results/portfolio_rank7_capacity_update_2026-07-28.json")
GROSS9_RESULT_SHA256 = (
    "7260ba91698ac31838558d1af11701e5e251f9b49749a573c8d38610d5388756"
)

GROSS9_MARKET = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
)
GROSS9_MARKET_SHA256 = (
    "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
)
GROSS9_MARKET_BYTES = 66_696_659
GROSS9_MARKET_OI = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
)
GROSS9_MARKET_OI_SHA256 = (
    "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
)
GROSS9_MARKET_OI_BYTES = 72_898_508

BASELINE_WEIGHTS = {
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "rex_taker_low_range_position": 0.4,
    "cand_rex_veto_7": 1.6,
    "markov_transition_long": 2.0,
}
WEIGHT_GRID = (0.25, 0.50, 0.75, 1.00)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with (REPO_ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_open_metadata() -> None:
    expected = {
        CDP_SELECTION: CDP_SELECTION_SHA256,
        CDP_VETO: CDP_VETO_SHA256,
        GROSS9_ANCHOR: GROSS9_ANCHOR_SHA256,
        GROSS9_CONFIG: GROSS9_CONFIG_SHA256,
        GROSS9_RESULT: GROSS9_RESULT_SHA256,
    }
    for path, expected_hash in expected.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"frozen metadata drift: {path}")
    selection = json.loads((REPO_ROOT / CDP_SELECTION).read_text())
    veto = json.loads((REPO_ROOT / CDP_VETO).read_text())
    if selection.get("selected_top1") != "CDP_S50_G05":
        raise RuntimeError("CDP1 top1 drift")
    if veto.get("decision") != "pass":
        raise RuntimeError("CDP1 standalone future veto no longer passes")


def build_preregistration() -> dict[str, Any]:
    _validate_open_metadata()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": "2026-07-29",
        "stage": "pre_gross9_portfolio_interaction",
        "candidate_contract": {
            "family": "PSIM-D8-CDP1",
            "fixed_top1": "CDP_S50_G05",
            "signal_thresholds_or_state_repair_allowed": False,
            "standalone_selection_result": {
                "path": CDP_SELECTION.as_posix(),
                "sha256": CDP_SELECTION_SHA256,
            },
            "standalone_future_veto_result": {
                "path": CDP_VETO.as_posix(),
                "sha256": CDP_VETO_SHA256,
            },
            "unit_leverage": 0.5,
            "base_cost_per_side": 0.0006,
            "stress_cost_per_side": 0.0010,
            "funding": "exact_mark_funding_cashflows",
            "decision_clock": "daily D8 card decision_at 12:05 UTC",
            "entry_clock": "first 5m open one complete bar after decision_at",
            "hold_bars_5m": 288,
            "overlap": "forbidden_first_signal_wins",
        },
        "gross9_authority": {
            "weights": BASELINE_WEIGHTS,
            "gross": 9.0,
            "accounting": "same_btc_low_high_v1",
            "anchor": {
                "path": GROSS9_ANCHOR.as_posix(),
                "sha256": GROSS9_ANCHOR_SHA256,
            },
            "config": {
                "path": GROSS9_CONFIG.as_posix(),
                "sha256": GROSS9_CONFIG_SHA256,
            },
            "result": {
                "path": GROSS9_RESULT.as_posix(),
                "sha256": GROSS9_RESULT_SHA256,
            },
            "market": {
                "path": GROSS9_MARKET.as_posix(),
                "sha256": GROSS9_MARKET_SHA256,
                "bytes": GROSS9_MARKET_BYTES,
                "must_exist_and_match_before_any_interaction_metric": True,
            },
            "market_with_oi": {
                "path": GROSS9_MARKET_OI.as_posix(),
                "sha256": GROSS9_MARKET_OI_SHA256,
                "bytes": GROSS9_MARKET_OI_BYTES,
                "must_exist_and_match_before_any_interaction_metric": True,
            },
            "db_or_regenerated_cache_substitution_allowed": False,
        },
        "selection_contract": {
            "window": ["2022-01-01", "2023-01-01"],
            "candidate_weight_grid": list(WEIGHT_GRID),
            "grid_origin": (
                "inherited unchanged from the preregistered Gross9 addition "
                "grid used before CDP1 source or outcome access"
            ),
            "same_gross_comparator": (
                "Gross9 weights multiplied by (9 + candidate_weight) / 9"
            ),
            "ranking": [
                "cagr_to_strict_mdd_improvement_vs_same_gross_descending",
                "strict_mdd_reduction_vs_unscaled_gross9_descending",
                "candidate_weight_ascending",
            ],
            "top1_only": True,
            "gates": {
                "absolute_return_retention_vs_unscaled_gross9": 0.97,
                "minimum_cagr_to_strict_mdd_improvement_vs_same_gross": 0.05,
                "strict_mdd_not_above_unscaled_gross9": True,
                "base_and_stress_candidate_return_positive": True,
                "maximum_exact_entry_jaccard_vs_any_gross9_sleeve": 0.25,
                "gross_cap": 10.0,
            },
        },
        "future_veto_contract": {
            "window": ["2023-01-01", "2024-01-01"],
            "future_can_rerank": False,
            "future_can_select_rank2": False,
            "future_can_repair_candidate_or_weight": False,
            "gates": {
                "absolute_return_retention_vs_unscaled_gross9": 0.97,
                "minimum_cagr_to_strict_mdd_improvement_vs_same_gross": 0.05,
                "strict_mdd_not_above_unscaled_gross9": True,
                "base_and_stress_candidate_return_not_negative": True,
                "maximum_exact_entry_jaccard_vs_any_gross9_sleeve": 0.25,
            },
        },
        "portfolio_accounting_contract": {
            "bar_timing": "signal_t_entry_t_plus_1_open",
            "costs": "candidate entry and exit costs charged on exact notional",
            "funding": "candidate exact settlement mark cashflows; Gross9 frozen replay",
            "drawdown": (
                "same-BTC OHLC upper-before-lower strict intraposition MDD "
                "on the shared 5m clock"
            ),
            "overlap_controls": [
                "exact entry-position Jaccard against each Gross9 sleeve",
                "occupied-bar Jaccard against each Gross9 sleeve",
                "daily marked-return Pearson and Spearman correlations",
            ],
        },
        "access_boundary": {
            "gross9_market_opened_or_hashed": False,
            "gross9_market_with_oi_opened_or_hashed": False,
            "gross9_events_rebuilt": False,
            "2022_portfolio_interaction_metrics_computed": False,
            "2023_portfolio_interaction_metrics_computed": False,
        },
        "failure_actions": {
            "missing_or_drifting_authority": "PAUSE_WITHOUT_SUBSTITUTION",
            "no_passing_2022_cell": "TERMINAL_REJECT_CDP1_NO_REPAIR",
            "failed_2023_veto": "TERMINAL_REJECT_CDP1_NO_REPAIR",
            "passing_2023_veto": (
                "PROMOTE_ONLY_AFTER_DETERMINISTIC_REPLAY_TESTS_AND_CRITIC_PASS"
            ),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_preregistration()
    target = path if path.is_absolute() else REPO_ROOT / path
    rendered = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"preregistration drift: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "fixed_top1": payload["candidate_contract"]["fixed_top1"],
                "manifest_hash": payload["manifest_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
