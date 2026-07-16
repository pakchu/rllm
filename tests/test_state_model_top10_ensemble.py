from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import training.portfolio_opt_added_alpha_update as base
import training.portfolio_opt_state_ensemble_update as optimizer
from training.state_model_top10_ensemble import (
    STRICT_MAJORITY_COUNT,
    load_pre_evaluation_top10,
    strict_majority_mask,
)


def test_strict_majority_requires_six_of_ten() -> None:
    masks = [np.array([index < 6, index < 5, True]) for index in range(10)]

    actual = strict_majority_mask(masks)

    assert STRICT_MAJORITY_COUNT == 6
    assert actual.tolist() == [True, False, True]


def test_strict_majority_rejects_non_top10_shape() -> None:
    with pytest.raises(ValueError, match="exactly 10"):
        strict_majority_mask([np.ones(3, dtype=bool)] * 9)
    with pytest.raises(ValueError, match="one market grid"):
        strict_majority_mask(
            [np.ones(3, dtype=bool)] * 9 + [np.ones(4, dtype=bool)]
        )


def test_loader_uses_frozen_selected_order_without_future_fields(tmp_path) -> None:
    selected = [
        {
            "signal_hash": f"hash-{rank}",
            "parameter": rank,
            "train": {
                "ratio": 12 - rank,
                "cagr_pct": 12 - rank,
            },
            "test2024": {
                "ratio": 12 - rank,
                "cagr_pct": 12 - rank,
                "return_pct": 12 - rank,
            },
            "eval2025": {"ratio": 100 - rank},
        }
        for rank in range(12)
    ]
    source = tmp_path / "scan.json"
    source.write_text(
        json.dumps(
            {
                "protocol": "train/test rank; eval2025/2026 report-only diagnostics",
                "selected": selected,
            }
        )
    )

    rows, metadata = load_pre_evaluation_top10(source, "kalman")

    assert [row["parameter"] for row in rows] == list(range(10))
    assert metadata["future_fields_used"] is False
    assert metadata["rows_read"] == 10
    assert metadata["pre_evaluation_order_verified"] is True


def test_optimizer_universe_patch_is_scoped_and_family_capped() -> None:
    original = (base.SLEEVES, base.NEW_SLEEVES, base.FAMILIES, base.feature_frame, base.split_arrays)

    with optimizer.patched_portfolio_universe(optimizer.DEFAULT_CONFIG):
        assert base.SLEEVES == optimizer.SLEEVES
        assert base.NEW_SLEEVES == optimizer.NEW_SLEEVES
        assert all(
            base.FAMILIES[name] == "funding_premium"
            for name in optimizer.ENSEMBLE_SLEEVES
        )
        assert base.feature_frame is not original[3]
        assert base.split_arrays is not original[4]

    assert (base.SLEEVES, base.NEW_SLEEVES, base.FAMILIES, base.feature_frame, base.split_arrays) == original


def test_rejected_rank1_is_not_exposed_as_candidate_weights() -> None:
    previous = {
        "weights": {"previous": 1.0},
        "gross": 1.0,
    }
    rejected = {
        "weights": {"bocpd_top10_strict_majority_long": 1.75},
        "gross": 1.75,
        "future_veto_passed": False,
    }
    report = {
        "frozen_pre2025_top1": rejected,
        "previous_added_alpha_best": {"frozen_pre2025_top1": previous},
        "replace_previous_candidate": False,
        "deployment_disposition": "retain_previous_added_alpha_shadow_candidate",
        "selected_state_ensemble_sleeves": ["bocpd_top10_strict_majority_long"],
        "protocol_hash": "hash",
    }

    candidate = optimizer.build_candidate_config(report, optimizer.DEFAULT_CONFIG)

    assert candidate["weights"] == previous["weights"]
    assert candidate["gross_weight"] == previous["gross"]
    assert candidate["selected_state_ensemble_sleeves"] == []
    assert candidate["rejected_frozen_rank1"]["weights"] == rejected["weights"]
    assert candidate["rejected_frozen_rank1"]["state_ensemble_sleeves"] == [
        "bocpd_top10_strict_majority_long"
    ]


def test_committed_state_ensemble_result_retains_previous_candidate() -> None:
    result = json.loads(
        Path("results/portfolio_state_ensemble_update_2026-07-16.json").read_text()
    )
    candidate = json.loads(
        Path("configs/shadow/portfolio_state_ensemble_candidate_2026-07-16.json").read_text()
    )
    portfolio_pool = json.loads(Path("research/pools/portfolio_pool.json").read_text())
    previous = result["previous_added_alpha_best"]["frozen_pre2025_top1"]
    rejected = result["frozen_pre2025_top1"]

    assert result["future_used_for_allocation_ranking"] is False
    assert result["future_can_only_veto_frozen_rank1"] is True
    assert rejected["future_veto_passed"] is False
    assert result["replace_previous_candidate"] is False
    assert candidate["weights"] == previous["weights"]
    assert candidate["selected_state_ensemble_sleeves"] == []
    assert candidate["rejected_frozen_rank1"]["weights"] == rejected["weights"]
    state_audit = result["source_validation"]["state_model_top10_ensembles"]
    assert state_audit["future_fields_used"] is False
    assert all(
        len(family["members"]) == 10
        for family in state_audit["families"].values()
    )
    pool_entry = next(
        entry
        for entry in portfolio_pool["entries"]
        if entry["id"] == "state_ensemble_update_rejected_20260716"
    )
    assert pool_entry["status"] == "rejected"
    assert pool_entry["protocol_hash"] == result["protocol_hash"]
    assert pool_entry["construction_recipe"]["retained_portfolio"] == (
        "added_alpha_gross800_shadow_20260716"
    )
