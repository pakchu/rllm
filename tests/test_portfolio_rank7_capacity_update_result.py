from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.portfolio_opt_added_alpha_update import (
    FAMILIES,
    SLEEVES,
    SPLIT_BOUNDS,
    Config,
    forward_veto,
    json_hash,
    pre2025_row_sort_key,
    pre2025_selection_key,
    valid_weights,
)


RESULT = Path("results/portfolio_rank7_capacity_update_2026-07-28.json")
CANDIDATE = Path(
    "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json"
)
DOCS = Path("docs/portfolio-rank7-capacity-update-2026-07-28.md")
EXPECTED_WEIGHTS = {
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "rex_taker_low_range_position": 0.4,
    "cand_rex_veto_7": 1.6,
    "markov_transition_long": 2.0,
}


def load_result() -> tuple[dict, Config]:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    config = Config(**payload["config"])
    return payload, config


def test_rank7_capacity_result_is_frozen_pre2025_top1() -> None:
    payload, config = load_result()
    selected = payload["frozen_pre2025_top1"]

    assert payload["future_used_for_allocation_ranking"] is False
    assert payload["future_can_only_veto_frozen_rank1"] is True
    assert selected["weights"] == EXPECTED_WEIGHTS
    assert selected["gross"] == 9.0
    assert valid_weights(selected["weights"], config)
    assert selected["future_veto_passed"] is True
    assert forward_veto(selected["stats"], config)

    reconstructed = []
    for row in payload["top_pre2025"]:
        reconstructed.append(
            {
                **row,
                "selection_key": pre2025_selection_key(row["stats"], config),
            }
        )
    ranked = sorted(reconstructed, key=pre2025_row_sort_key, reverse=True)
    assert ranked[0]["weights"] == selected["weights"]
    assert [row["weights"] for row in ranked] == [
        row["weights"] for row in reconstructed
    ]


def test_rank7_capacity_is_evidence_bound_without_duplicate_sleeve() -> None:
    payload, config = load_result()
    evidence = payload["candidate_universe"]["rank7_capacity_extension"]

    assert SLEEVES.count("frozen_annual_rank7") == 1
    assert evidence["selection_uses_only_pre_2025_windows"] is True
    assert evidence["future_repair_or_reselection"] is False
    assert evidence["base_leverage"] == 0.5
    assert evidence["selected_leverage"] == 1.5
    assert evidence["selected_multiplier"] == config.rank7_family_gross_cap == 3.0
    assert evidence["duplicate_sleeve_created"] is False
    assert evidence["report_only_metrics_used_for_allocation"] is False
    assert payload["family_gross_caps"]["rank7"] == 3.0
    assert all(
        cap == 2.0
        for family, cap in payload["family_gross_caps"].items()
        if family != "rank7"
    )


def test_rank7_capacity_result_preserves_gross8_gates_and_improves_ratio() -> None:
    payload, config = load_result()
    selected = payload["frozen_pre2025_top1"]["stats"]
    previous = payload["comparison_portfolio"]["stats"]

    assert config.gross_cap == 10.0
    assert config.min_nonzero_weight == 0.25
    assert config.weight_step == 0.05
    assert config.train_mdd_cap == 40.0
    assert config.test_mdd_cap == 20.0
    assert config.future_mdd_cap == 20.0
    assert config.min_test_trades == 80
    assert config.min_test_ratio == 3.0
    assert config.min_future_ratio == 3.0
    assert config.cost_rate == 0.0006

    for split in SPLIT_BOUNDS:
        assert selected[split]["absolute_return_pct"] > previous[split][
            "absolute_return_pct"
        ]
        assert selected[split]["cagr_to_strict_mdd"] > previous[split][
            "cagr_to_strict_mdd"
        ]
    assert selected["train"]["strict_mdd_pct"] <= config.train_mdd_cap
    assert selected["test2024"]["strict_mdd_pct"] <= config.test_mdd_cap
    assert selected["test2024"]["trades"] >= config.min_test_trades
    for split in ("eval2025", "ytd2026"):
        assert selected[split]["strict_mdd_pct"] <= config.future_mdd_cap
        assert selected[split]["cagr_to_strict_mdd"] >= config.min_future_ratio


def test_rank7_capacity_protocol_hash_and_shadow_candidate_match() -> None:
    payload, config = load_result()
    constraints = {
        key: value
        for key, value in payload["config"].items()
        if key
        in {
            "gross_cap",
            "family_gross_cap",
            "min_nonzero_weight",
            "weight_step",
            "train_mdd_cap",
            "test_mdd_cap",
            "future_mdd_cap",
            "min_test_trades",
            "min_test_ratio",
            "min_future_ratio",
            "random_samples",
            "seed",
            "seed_count",
            "refinement_rounds",
            "refinement_top_n",
            "refinement_patience",
            "cost_rate",
        }
    }
    constraints["family_gross_cap_overrides"] = {
        "rank7": config.rank7_family_gross_cap
    }
    expected_hash = json_hash(
        {
            "sleeves": SLEEVES,
            "families": FAMILIES,
            "splits": SPLIT_BOUNDS,
            "constraints": constraints,
            "input_sha256": {
                name: record["sha256"]
                for name, record in payload["input_provenance"].items()
            },
            "accounting_version": payload["accounting_version"],
            "selection": (
                "train+test2024 only; exact multi-seed beam refinement; "
                "tie=lower gross then lexicographic weights; future veto cannot rerank"
            ),
        }
    )
    assert payload["protocol_hash"] == expected_hash

    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert candidate["status"] == "forward_shadow_candidate_not_live"
    assert candidate["weights"] == EXPECTED_WEIGHTS
    assert candidate["gross_weight"] == 9.0
    assert candidate["protocol_hash"] == payload["protocol_hash"]
    assert candidate["source_result"] == str(RESULT)
    assert np.isclose(
        candidate["rank7_capacity_extension"]["selected_multiplier"],
        config.rank7_family_gross_cap,
    )
    assert "capacity-bound result" in DOCS.read_text(encoding="utf-8")
